#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return None
    return pd.read_csv(path)


def save_plot(
    fig: plt.Figure,
    path: Path,
    layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    if layout_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=layout_rect)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if np.isnan(default):
        return out
    return out.fillna(default)


@dataclass
class Config:
    input_tables_dir: Path
    output_dir: Path
    min_segments_per_show: int = 15
    birth_threshold_pct: float = 1.0
    similarity_top_k: int = 50
    random_seed: int = 42
    excluded_topics: tuple[int, ...] = ()
    multi_topic_threshold: float = 0.08
    multi_topic_min_topics: int = 2


class AdvancedTopicAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.output_dir / "tables")
        ensure_dir(cfg.output_dir / "figures")
        self.rng = np.random.default_rng(cfg.random_seed)
        self.segments_topics = None
        self.topic_info = None
        self.segment_topic_probs_long = None
        self.show_year_topic_distribution = None
        self.work = None
        self.weighted_topic_year = None
        self.weighted_topic_decade = None
        self.multi_topic_long = None
        self.n_excluded_topics_segments = 0

    def load(self):
        tdir = self.cfg.input_tables_dir
        self.segments_topics = read_csv(tdir / "segments_topics.csv", required=True)
        self.topic_info = read_csv(tdir / "topic_info.csv", required=False)
        self.segment_topic_probs_long = read_csv(
            tdir / "segment_topic_probs_long.csv", required=False
        )
        self.show_year_topic_distribution = read_csv(
            tdir / "show_year_topic_distribution.csv", required=False
        )

        for col in ["show", "year", "topic_final"]:
            if col not in self.segments_topics.columns:
                raise ValueError(f"`segments_topics.csv` missing column: {col}")

        self.segments_topics["year"] = safe_numeric(self.segments_topics["year"])
        self.segments_topics = self.segments_topics.dropna(subset=["year"]).copy()
        self.segments_topics["year"] = self.segments_topics["year"].astype(int)

        if "max_topic_probability" not in self.segments_topics.columns:
            self.segments_topics["max_topic_probability"] = 1.0
        else:
            self.segments_topics["max_topic_probability"] = safe_numeric(
                self.segments_topics["max_topic_probability"], default=1.0
            ).clip(0, 1)

    def build_working_table(self):
        df = self.segments_topics.copy()
        df = df[df["topic_final"] != -1].copy()
        if self.cfg.excluded_topics:
            before = len(df)
            df = df[~df["topic_final"].isin(self.cfg.excluded_topics)].copy()
            self.n_excluded_topics_segments = before - len(df)
        df["topic_weight"] = df["max_topic_probability"].fillna(1.0).clip(0, 1)
        df["decade"] = (df["year"] // 10) * 10
        self.work = df

    def _attach_topic_names(
        self,
        df: pd.DataFrame,
        topic_col: str,
        output_col: str,
    ) -> pd.DataFrame:
        if self.topic_info is None:
            return df
        required = {"Topic", "Name"}
        if not required.issubset(self.topic_info.columns):
            return df
        mapping = (
            self.topic_info[["Topic", "Name"]]
            .dropna(subset=["Topic", "Name"])
            .copy()
            .assign(Topic=lambda x: x["Topic"].astype(int))
            .drop_duplicates("Topic")
            .rename(columns={"Topic": topic_col, "Name": output_col})
        )
        return df.merge(mapping, on=topic_col, how="left")

    def _topic_short_label(self, topic_id: int, topic_name: object, max_terms: int = 3) -> str:
        if pd.isna(topic_name):
            return f"T{int(topic_id)}"
        text = str(topic_name).strip()
        prefix = f"{int(topic_id)}_"
        if text.startswith(prefix):
            text = text[len(prefix) :]
        text = " ".join(text.replace("_", " ").split())
        if not text:
            return f"T{int(topic_id)}"
        terms = text.split()
        short = " ".join(terms[:max_terms])
        return f"T{int(topic_id)} {short}"

    @staticmethod
    def _brand_cmap() -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            "datarisas_teal",
            ["#edf9f8", "#c8efea", "#6ed9cf", "#25b7ad", "#0c8f87"],
        )

    @staticmethod
    def _brand_contrast_cmap() -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list(
            "datarisas_teal_orange",
            ["#edf9f8", "#98ddd7", "#27bdb2", "#f0ad1f", "#d94801"],
        )

    def analyze_soft_topic_prevalence(self):
        w = self.work.copy()

        topic_year = (
            w.groupby(["year", "topic_final"], as_index=False)["topic_weight"]
            .sum()
            .rename(columns={"topic_weight": "weighted_mass"})
        )
        year_total = topic_year.groupby("year", as_index=False)["weighted_mass"].sum().rename(
            columns={"weighted_mass": "year_mass"}
        )
        topic_year = topic_year.merge(year_total, on="year", how="left")
        topic_year["prevalence_pct"] = (
            100.0 * topic_year["weighted_mass"] / topic_year["year_mass"]
        )
        if "topic_name_final" in self.work.columns:
            names = self.work[["topic_final", "topic_name_final"]].drop_duplicates(
                "topic_final"
            )
            topic_year = topic_year.merge(names, on="topic_final", how="left")
        self.weighted_topic_year = topic_year
        topic_year.to_csv(
            self.cfg.output_dir / "tables" / "soft_topic_prevalence_by_year.csv",
            index=False,
        )

        topic_decade = (
            w.groupby(["decade", "topic_final"], as_index=False)["topic_weight"]
            .sum()
            .rename(columns={"topic_weight": "weighted_mass"})
        )
        decade_total = (
            topic_decade.groupby("decade", as_index=False)["weighted_mass"]
            .sum()
            .rename(columns={"weighted_mass": "decade_mass"})
        )
        topic_decade = topic_decade.merge(decade_total, on="decade", how="left")
        topic_decade["prevalence_pct"] = (
            100.0 * topic_decade["weighted_mass"] / topic_decade["decade_mass"]
        )
        if "topic_name_final" in self.work.columns:
            names = self.work[["topic_final", "topic_name_final"]].drop_duplicates(
                "topic_final"
            )
            topic_decade = topic_decade.merge(names, on="topic_final", how="left")
        self.weighted_topic_decade = topic_decade
        topic_decade.to_csv(
            self.cfg.output_dir / "tables" / "soft_topic_prevalence_by_decade.csv",
            index=False,
        )
        return topic_year, topic_decade

    def plot_top_topics_over_time(self, top_n: int = 10):
        t = self.weighted_topic_year.copy()
        top_topics = (
            t.groupby("topic_final")["weighted_mass"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        p = t[t["topic_final"].isin(top_topics)].pivot_table(
            index="year", columns="topic_final", values="prevalence_pct", fill_value=0.0
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        p.plot(ax=ax)
        ax.set_title(f"Soft topic prevalence over time (top {top_n})")
        ax.set_xlabel("Year")
        ax.set_ylabel("Prevalence (%)")
        ax.grid(alpha=0.25)
        save_plot(
            fig,
            self.cfg.output_dir / "figures" / "soft_topic_prevalence_top_topics.png",
        )

    def plot_top_topics_over_decades(self, top_n: int = 10):
        t = self.weighted_topic_decade.copy()
        t["topic_short"] = t.apply(
            lambda r: self._topic_short_label(int(r["topic_final"]), r.get("topic_name_final")),
            axis=1,
        )

        brand_cmap = self._brand_cmap()

        top_topics = (
            t.groupby("topic_final")["weighted_mass"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        p = t[t["topic_final"].isin(top_topics)].pivot_table(
            index="decade", columns="topic_final", values="prevalence_pct", fill_value=0.0
        )
        label_map = (
            t[["topic_final", "topic_short"]]
            .drop_duplicates("topic_final")
            .set_index("topic_final")["topic_short"]
            .to_dict()
        )

        fig, ax = plt.subplots(figsize=(13, 7), facecolor="#eff8f7")
        ax.set_facecolor("#ffffff")
        ax.grid(color="#d6ecea", alpha=0.8, linewidth=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#bdddd9")
        ax.spines["bottom"].set_color("#bdddd9")

        color_positions = np.linspace(0.18, 0.98, max(len(top_topics), 1))
        color_lookup = {
            topic_id: brand_cmap(pos) for topic_id, pos in zip(top_topics, color_positions, strict=False)
        }
        for topic_id in top_topics:
            series = p[topic_id].sort_index()
            ax.plot(
                series.index,
                series.values,
                color=color_lookup[topic_id],
                linewidth=2.2,
                marker="o",
                markersize=5,
                alpha=0.95,
            )
            if len(series):
                ax.text(
                    float(series.index.max()) + 0.3,
                    float(series.iloc[-1]),
                    label_map.get(topic_id, f"T{topic_id}"),
                    fontsize=9,
                    color="#155b58",
                    va="center",
                )

        ax.set_title(
            f"Temas mas relevantes por decada (top {top_n})",
            fontsize=18,
            color="#0f6661",
            pad=16,
            fontweight="bold",
        )
        ax.set_xlabel("Decada", fontsize=12, color="#2d3d3f")
        ax.set_ylabel("Prevalencia (%)", fontsize=12, color="#2d3d3f")
        ax.tick_params(colors="#2d3d3f")
        ax.margins(x=0.03)
        fig.text(
            0.5,
            0.015,
            "Cada linea representa un topico. Etiquetas al extremo derecho para lectura rapida.",
            ha="center",
            fontsize=10,
            color="#587272",
        )
        save_plot(
            fig,
            self.cfg.output_dir
            / "figures"
            / "soft_topic_prevalence_by_decade_top_topics.png",
            layout_rect=(0.0, 0.06, 1.0, 1.0),
        )

        all_topics = (
            t.pivot_table(
                index="topic_final",
                columns="decade",
                values="prevalence_pct",
                aggfunc="sum",
                fill_value=0.0,
            )
            .sort_index(axis=1)
            .copy()
        )
        all_topics["mean_prevalence"] = all_topics.mean(axis=1)
        all_topics = all_topics.sort_values("mean_prevalence", ascending=False).drop(
            columns=["mean_prevalence"]
        )

        y_labels = [
            label_map.get(int(topic_id), f"T{int(topic_id)}")
            for topic_id in all_topics.index.to_list()
        ]
        mat = all_topics.to_numpy()
        decades = [int(v) for v in all_topics.columns.to_list()]

        heat_h = max(8.5, 0.34 * len(y_labels) + 2.2)
        fig_h, ax_h = plt.subplots(figsize=(15, heat_h), facecolor="#eff8f7")
        ax_h.set_facecolor("#ffffff")
        im = ax_h.imshow(mat, aspect="auto", cmap=brand_cmap, interpolation="nearest")
        ax_h.set_xticks(np.arange(len(decades)))
        ax_h.set_xticklabels(decades, fontsize=10, color="#2d3d3f")
        ax_h.set_yticks(np.arange(len(y_labels)))
        ax_h.set_yticklabels(y_labels, fontsize=8.5, color="#2d3d3f")
        ax_h.tick_params(axis="both", which="both", length=0)
        ax_h.set_xlabel("Decada", fontsize=12, color="#2d3d3f")
        ax_h.set_ylabel("Topicos (filtrados)", fontsize=12, color="#2d3d3f")
        ax_h.set_title(
            "Mapa completo de prevalencia tematica por decada",
            fontsize=18,
            color="#0f6661",
            pad=16,
            fontweight="bold",
        )
        for side in ["top", "right", "left", "bottom"]:
            ax_h.spines[side].set_visible(False)
        cbar = fig_h.colorbar(im, ax=ax_h, pad=0.01)
        cbar.set_label("Prevalencia (%)", color="#2d3d3f")
        cbar.ax.yaxis.set_tick_params(color="#2d3d3f")
        plt.setp(cbar.ax.get_yticklabels(), color="#2d3d3f")
        fig_h.text(
            0.5,
            0.01,
            "Filas ordenadas por prevalencia promedio. Etiqueta corta por topico para lectura periodistica.",
            ha="center",
            fontsize=10,
            color="#587272",
        )
        save_plot(
            fig_h,
            self.cfg.output_dir
            / "figures"
            / "soft_topic_prevalence_by_decade_all_topics.png",
            layout_rect=(0.0, 0.04, 1.0, 1.0),
        )

        ranking = (
            t.sort_values(["decade", "prevalence_pct"], ascending=[True, False])
            .groupby("decade", as_index=False)
            .head(10)
            .copy()
        )
        ranking["rank"] = ranking.groupby("decade").cumcount() + 1
        ranking = ranking[
            ["decade", "rank", "topic_final", "topic_short", "prevalence_pct"]
        ].rename(columns={"topic_short": "topic_label"})
        ranking.to_csv(
            self.cfg.output_dir / "tables" / "topic_ranking_top10_by_decade.csv",
            index=False,
        )

        ranking["ranked_topic"] = ranking.apply(
            lambda r: f"#{int(r['rank'])} {r['topic_label']} ({float(r['prevalence_pct']):.1f}%)",
            axis=1,
        )
        ranking_wide = (
            ranking.pivot(index="rank", columns="decade", values="ranked_topic")
            .sort_index()
            .sort_index(axis=1)
        )
        ranking_wide.to_csv(
            self.cfg.output_dir / "tables" / "topic_ranking_top10_by_decade_wide.csv"
        )

        table_df = ranking_wide.fillna("-")
        fig_t, ax_t = plt.subplots(
            figsize=(2.2 + 2.35 * table_df.shape[1], 1.7 + 0.55 * table_df.shape[0]),
            facecolor="#eff8f7",
        )
        ax_t.axis("off")
        table = ax_t.table(
            cellText=table_df.values.tolist(),
            rowLabels=[f"#{int(r)}" for r in table_df.index.to_list()],
            colLabels=[str(int(c)) for c in table_df.columns.to_list()],
            loc="center",
            cellLoc="left",
            rowLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.2)
        table.scale(1.0, 1.35)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#22b8ad")
                cell.set_text_props(color="white", weight="bold")
                cell.set_edgecolor("#d2ece9")
            elif col == -1:
                cell.set_facecolor("#d6f1ee")
                cell.set_text_props(color="#155b58", weight="bold")
                cell.set_edgecolor("#d2ece9")
            else:
                cell.set_edgecolor("#d2ece9")
                cell.set_facecolor("#ffffff" if row % 2 == 0 else "#f4fbfa")
                cell.set_text_props(color="#2c4042")

        ax_t.set_title(
            "Ranking de topicos por decada (Top 10)",
            fontsize=18,
            color="#0f6661",
            pad=12,
            fontweight="bold",
        )
        fig_t.text(
            0.5,
            0.025,
            "Tabla comparativa para seguir como cambia el ranking tematico entre decadas.",
            ha="center",
            fontsize=10,
            color="#587272",
        )
        save_plot(
            fig_t,
            self.cfg.output_dir / "figures" / "topic_ranking_top10_by_decade_table.png",
            layout_rect=(0.0, 0.05, 1.0, 1.0),
        )

    def analyze_topic_lifecycle(self):
        t = self.weighted_topic_year[["year", "topic_final", "prevalence_pct"]].copy()
        rows = []
        for topic_id, g in t.groupby("topic_final"):
            g = g.sort_values("year")
            active = g[g["prevalence_pct"] >= self.cfg.birth_threshold_pct]
            rows.append(
                {
                    "topic_final": int(topic_id),
                    "birth_year": int(active["year"].min()) if len(active) else np.nan,
                    "active_years": int(active["year"].nunique()) if len(active) else 0,
                    "volatility_mean_abs_diff": float(
                        g["prevalence_pct"].diff().abs().mean(skipna=True)
                    ),
                    "mean_prevalence_pct": float(g["prevalence_pct"].mean()),
                    "max_prevalence_pct": float(g["prevalence_pct"].max()),
                }
            )
        out = pd.DataFrame(rows).sort_values(
            ["birth_year", "mean_prevalence_pct"], ascending=[True, False]
        )
        out.to_csv(self.cfg.output_dir / "tables" / "topic_lifecycle_metrics.csv", index=False)
        return out

    def analyze_topic_lifecycle_by_decade(self):
        t = self.weighted_topic_decade[
            ["decade", "topic_final", "prevalence_pct"]
        ].copy()
        rows = []
        for topic_id, g in t.groupby("topic_final"):
            g = g.sort_values("decade")
            active = g[g["prevalence_pct"] >= self.cfg.birth_threshold_pct]
            rows.append(
                {
                    "topic_final": int(topic_id),
                    "birth_decade": int(active["decade"].min()) if len(active) else np.nan,
                    "active_decades": int(active["decade"].nunique()) if len(active) else 0,
                    "volatility_mean_abs_diff": float(
                        g["prevalence_pct"].diff().abs().mean(skipna=True)
                    ),
                    "mean_prevalence_pct": float(g["prevalence_pct"].mean()),
                    "max_prevalence_pct": float(g["prevalence_pct"].max()),
                }
            )
        out = pd.DataFrame(rows).sort_values(
            ["birth_decade", "mean_prevalence_pct"], ascending=[True, False]
        )
        out.to_csv(
            self.cfg.output_dir / "tables" / "topic_lifecycle_metrics_by_decade.csv",
            index=False,
        )
        return out

    def analyze_show_similarity(self):
        w = self.work.copy()
        show_sizes = w.groupby("show").size().rename("n_segments").reset_index()
        eligible = show_sizes[show_sizes["n_segments"] >= self.cfg.min_segments_per_show][
            "show"
        ]
        sw = w[w["show"].isin(eligible)].copy()

        show_topic = (
            sw.groupby(["show", "topic_final"], as_index=False)["topic_weight"]
            .sum()
            .rename(columns={"topic_weight": "mass"})
        )
        show_total = show_topic.groupby("show", as_index=False)["mass"].sum().rename(
            columns={"mass": "show_mass"}
        )
        show_topic = show_topic.merge(show_total, on="show", how="left")
        show_topic["p_topic"] = show_topic["mass"] / show_topic["show_mass"]

        M = show_topic.pivot_table(
            index="show", columns="topic_final", values="p_topic", fill_value=0.0
        )
        if M.shape[0] < 2:
            raise ValueError("Not enough shows for similarity analysis.")

        sim = cosine_similarity(M.values)
        sim_df = pd.DataFrame(sim, index=M.index, columns=M.index)
        sim_df.to_csv(self.cfg.output_dir / "tables" / "show_similarity_matrix.csv")

        tri_i, tri_j = np.triu_indices_from(sim, k=1)
        pairs = pd.DataFrame(
            {
                "show_a": [M.index[i] for i in tri_i],
                "show_b": [M.index[j] for j in tri_j],
                "cosine_similarity": sim[tri_i, tri_j],
            }
        ).sort_values("cosine_similarity", ascending=False)
        pairs.to_csv(self.cfg.output_dir / "tables" / "show_similarity_pairs.csv", index=False)
        pairs.head(self.cfg.similarity_top_k).to_csv(
            self.cfg.output_dir / "tables" / "show_similarity_pairs_topk.csv", index=False
        )

        pca = PCA(n_components=2, random_state=self.cfg.random_seed)
        coords = pca.fit_transform(M.values)
        coords_df = (
            pd.DataFrame(coords, index=M.index, columns=["x", "y"])
            .reset_index()
            .rename(columns={"index": "show"})
        )
        coords_df = coords_df.merge(show_sizes, on="show", how="left")
        coords_df.to_csv(self.cfg.output_dir / "tables" / "show_topic_space_pca.csv", index=False)

        coords_df["distance_center"] = np.hypot(
            coords_df["x"] - coords_df["x"].mean(),
            coords_df["y"] - coords_df["y"].mean(),
        )
        point_sizes = 45 + np.sqrt(coords_df["n_segments"]) * 13
        coord_lookup = coords_df.set_index("show")[["x", "y"]]

        def in_view(x: float, y: float, xlim, ylim) -> bool:
            x_ok = True if xlim is None else (xlim[0] <= x <= xlim[1])
            y_ok = True if ylim is None else (ylim[0] <= y <= ylim[1])
            return x_ok and y_ok

        def draw_pca_figure(
            output_name: str,
            title: str,
            caption: str,
            xlim=None,
            ylim=None,
            label_limit: int | None = 28,
            edge_limit: int = 12,
            use_contrast_cmap: bool = True,
            label_with_background: bool = True,
        ) -> None:
            fig, ax = plt.subplots(figsize=(14, 10), facecolor="#eff8f7")
            ax.set_facecolor("#ffffff")
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            ax.spines["left"].set_color("#bdddd9")
            ax.spines["bottom"].set_color("#bdddd9")
            ax.tick_params(colors="#2d3d3f")
            ax.grid(color="#d6ecea", alpha=0.8, linewidth=0.8, zorder=0)

            if xlim is not None:
                ax.set_xlim(xlim[0], xlim[1])
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])

            edge_rows = []
            for _, edge in pairs.iterrows():
                if edge["show_a"] not in coord_lookup.index or edge["show_b"] not in coord_lookup.index:
                    continue
                x1, y1 = coord_lookup.loc[edge["show_a"]]
                x2, y2 = coord_lookup.loc[edge["show_b"]]
                if not (in_view(x1, y1, xlim, ylim) and in_view(x2, y2, xlim, ylim)):
                    continue
                edge_rows.append(edge)
                if len(edge_rows) >= edge_limit:
                    break
            top_edges = pd.DataFrame(edge_rows)

            if not top_edges.empty:
                sim_min = float(top_edges["cosine_similarity"].min())
                sim_max = float(top_edges["cosine_similarity"].max())
                sim_denom = sim_max - sim_min if sim_max > sim_min else 1.0
                for _, edge in top_edges.iterrows():
                    x1, y1 = coord_lookup.loc[edge["show_a"]]
                    x2, y2 = coord_lookup.loc[edge["show_b"]]
                    sim_scaled = (float(edge["cosine_similarity"]) - sim_min) / sim_denom
                    ax.plot(
                        [x1, x2],
                        [y1, y2],
                        color="#22b8ad",
                        linestyle=(0, (5, 4)),
                        alpha=0.18 + 0.35 * sim_scaled,
                        linewidth=0.9 + 1.8 * sim_scaled,
                        zorder=1,
                    )

            view_mask = coords_df.apply(
                lambda r: in_view(float(r["x"]), float(r["y"]), xlim, ylim),
                axis=1,
            )
            plot_df = coords_df[view_mask].copy()
            if plot_df.empty:
                plot_df = coords_df.copy()

            vmin = float(np.nanpercentile(plot_df["n_segments"], 8))
            vmax = float(np.nanpercentile(plot_df["n_segments"], 96))
            norm = None
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                norm = mcolors.PowerNorm(gamma=0.72, vmin=vmin, vmax=vmax)

            scatter = ax.scatter(
                plot_df["x"],
                plot_df["y"],
                s=point_sizes.loc[plot_df.index],
                c=plot_df["n_segments"],
                cmap=self._brand_contrast_cmap() if use_contrast_cmap else self._brand_cmap(),
                norm=norm,
                alpha=0.90,
                edgecolors="#ffffff",
                linewidths=1.0,
                zorder=3,
            )
            cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
            cbar.set_label("Cantidad de chistes", color="#2d3d3f")
            cbar.ax.yaxis.set_tick_params(color="#2d3d3f")
            plt.setp(cbar.ax.get_yticklabels(), color="#2d3d3f")

            outliers = plot_df.nlargest(min(10, len(plot_df)), "distance_center")
            if not outliers.empty:
                ax.scatter(
                    outliers["x"],
                    outliers["y"],
                    s=point_sizes.loc[outliers.index] * 1.5,
                    facecolors="none",
                    edgecolors="#0f6661",
                    linewidths=1.1,
                    alpha=0.55,
                    zorder=4,
                )

            shows_in_edges = set(top_edges.get("show_a", pd.Series(dtype=str))).union(
                set(top_edges.get("show_b", pd.Series(dtype=str)))
            )
            plot_df["label_score"] = (
                plot_df["n_segments"].rank(pct=True)
                + plot_df["distance_center"].rank(pct=True)
                + plot_df["show"].isin(shows_in_edges).astype(float) * 0.75
            )
            ranked_labels = plot_df.sort_values("label_score", ascending=False).copy()
            label_df = (
                ranked_labels
                if label_limit is None
                else ranked_labels.head(label_limit).copy()
            )
            x_mid = float(plot_df["x"].mean())
            y_mid = float(plot_df["y"].mean())
            for i, (_, row) in enumerate(label_df.iterrows()):
                sx = 1 if row["x"] >= x_mid else -1
                sy = 1 if row["y"] >= y_mid else -1
                dx = sx * (8 + (i % 3) * 2)
                dy = sy * (7 + ((i // 3) % 3) * 2)
                ax.annotate(
                    row["show"],
                    (row["x"], row["y"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=9,
                    color="#203033",
                    bbox=(
                        {
                            "boxstyle": "round,pad=0.2",
                            "fc": "#f4fbfa",
                            "ec": "none",
                            "alpha": 0.9,
                        }
                        if label_with_background
                        else None
                    ),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "mutation_scale": 7,
                        "color": "#6b8f8c",
                        "lw": 0.55,
                        "alpha": 0.75,
                    },
                    zorder=5,
                )

            size_q = np.quantile(plot_df["n_segments"], [0.25, 0.5, 0.75])
            legend_handles = []
            for value in sorted({int(round(v)) for v in size_q if not np.isnan(v)}):
                legend_handles.append(
                    ax.scatter(
                        [],
                        [],
                        s=45 + np.sqrt(value) * 13,
                        c="#22b8ad",
                        alpha=0.5,
                        edgecolors="#ffffff",
                        linewidths=1.0,
                        label=f"{value} chistes",
                    )
                )
            if legend_handles:
                legend = ax.legend(
                    handles=legend_handles,
                    title="Tamano de punto",
                    loc="lower left",
                    frameon=True,
                    framealpha=0.92,
                    facecolor="#f4fbfa",
                    edgecolor="#d2ece9",
                )
                plt.setp(legend.get_title(), color="#2d3d3f")
                plt.setp(legend.get_texts(), color="#2d3d3f")

            ax.axhline(0, color="#bdddd9", linestyle="--", linewidth=0.9, alpha=0.8, zorder=2)
            ax.axvline(0, color="#bdddd9", linestyle="--", linewidth=0.9, alpha=0.8, zorder=2)
            ax.set_title(
                title,
                fontsize=20,
                color="#0f6661",
                pad=20,
                fontweight="bold",
            )
            ax.set_xlabel("Eje tematico 1 (PCA)", fontsize=12, color="#2d3d3f")
            ax.set_ylabel("Eje tematico 2 (PCA)", fontsize=12, color="#2d3d3f")
            fig.text(
                0.5,
                0.012,
                caption,
                ha="center",
                fontsize=10,
                color="#587272",
            )
            save_plot(
                fig,
                self.cfg.output_dir / "figures" / output_name,
                layout_rect=(0.0, 0.06, 1.0, 1.0),
            )

        draw_pca_figure(
            output_name="show_topic_space_pca.png",
            title="Mapa de cercania entre comediantes por estilo tematico",
            caption=(
                "Puntos mas cercanos representan rutinas con mezcla de topicos similar. "
                "Las lineas azules punteadas marcan las parejas mas parecidas."
            ),
            label_limit=28,
            edge_limit=12,
            use_contrast_cmap=True,
        )

        x_lo, x_hi = np.quantile(coords_df["x"], [0.10, 0.85])
        y_lo, y_hi = np.quantile(coords_df["y"], [0.10, 0.85])
        x_span = float(x_hi - x_lo)
        y_span = float(y_hi - y_lo)
        x_pad = 0.12 * x_span if x_span > 0 else 0.05
        y_pad = 0.12 * y_span if y_span > 0 else 0.05
        zoom_xlim = (float(x_lo - x_pad), 0.05)
        zoom_ylim = (float(y_lo - y_pad), float(y_hi + y_pad))
        if zoom_xlim[0] >= zoom_xlim[1]:
            zoom_xlim = (zoom_xlim[1] - 0.2, zoom_xlim[1])

        draw_pca_figure(
            output_name="show_topic_space_pca_zoom_cluster.png",
            title="Zoom al nucleo donde se concentra la mayoria",
            caption=(
                "Acercamiento del area mas densa para leer mejor las diferencias finas "
                "entre comediantes con estilos tematicos cercanos."
            ),
            xlim=zoom_xlim,
            ylim=zoom_ylim,
            label_limit=None,
            edge_limit=18,
            use_contrast_cmap=True,
            label_with_background=False,
        )

        return sim_df, pairs

    def analyze_multi_topic_segments(self):
        if self.segment_topic_probs_long is None:
            return None

        probs = self.segment_topic_probs_long.copy()
        required = {"segment_idx", "p_topic"}
        if not required.issubset(probs.columns):
            return None

        topic_col = None
        for candidate in ["topic_id_assumed", "topic_id_matrix"]:
            if candidate in probs.columns and probs[candidate].notna().any():
                topic_col = candidate
                break
        if topic_col is None:
            return None

        probs["segment_idx"] = safe_numeric(probs["segment_idx"])
        probs["topic_id"] = safe_numeric(probs[topic_col])
        probs["p_topic"] = safe_numeric(probs["p_topic"])
        probs = probs.dropna(subset=["segment_idx", "topic_id", "p_topic"]).copy()
        probs["segment_idx"] = probs["segment_idx"].astype(int)
        probs["topic_id"] = probs["topic_id"].astype(int)
        probs = probs[probs["p_topic"] >= float(self.cfg.multi_topic_threshold)].copy()
        if self.cfg.excluded_topics:
            probs = probs[~probs["topic_id"].isin(self.cfg.excluded_topics)].copy()
        if probs.empty:
            return {
                "n_segments_with_soft_topics": 0,
                "n_segments_multi_topic": 0,
                "multi_topic_threshold": float(self.cfg.multi_topic_threshold),
                "multi_topic_min_topics": int(self.cfg.multi_topic_min_topics),
                "excluded_topics_applied": list(self.cfg.excluded_topics),
            }

        probs = probs.rename(columns={"segment_idx": "segment_row_idx"})

        segment_index = self.segments_topics.reset_index(drop=True).copy()
        segment_index.insert(0, "segment_row_idx", np.arange(len(segment_index), dtype=int))
        segment_index["decade"] = (segment_index["year"] // 10) * 10
        metadata_cols = [
            "segment_row_idx",
            "show",
            "year",
            "decade",
            "topic_final",
            "topic_name_final",
        ]
        if "model_text" in segment_index.columns:
            metadata_cols.append("model_text")
        probs = probs.merge(segment_index[metadata_cols], on="segment_row_idx", how="left")
        probs = self._attach_topic_names(probs, topic_col="topic_id", output_col="topic_name")
        probs = probs.sort_values(["segment_row_idx", "p_topic"], ascending=[True, False]).copy()
        probs["topic_rank"] = probs.groupby("segment_row_idx").cumcount() + 1

        self.multi_topic_long = probs
        probs.to_csv(
            self.cfg.output_dir / "tables" / "segment_topics_soft_assignments.csv",
            index=False,
        )

        self._export_multitopic_prevalence(probs)

        base = (
            probs.groupby("segment_row_idx", as_index=False)
            .agg(
                n_topics_above_threshold=("topic_id", "size"),
                probability_mass_above_threshold=("p_topic", "sum"),
            )
            .copy()
        )
        top1 = probs[probs["topic_rank"] == 1][
            ["segment_row_idx", "topic_id", "topic_name", "p_topic"]
        ].rename(
            columns={
                "topic_id": "top1_topic_id",
                "topic_name": "top1_topic_name",
                "p_topic": "top1_p_topic",
            }
        )
        top2 = probs[probs["topic_rank"] == 2][
            ["segment_row_idx", "topic_id", "topic_name", "p_topic"]
        ].rename(
            columns={
                "topic_id": "top2_topic_id",
                "topic_name": "top2_topic_name",
                "p_topic": "top2_p_topic",
            }
        )
        candidate_meta = segment_index[metadata_cols].copy()
        if "model_text" in candidate_meta.columns:
            candidate_meta["text_preview"] = candidate_meta["model_text"].astype(str).str.slice(
                0, 160
            )
            candidate_meta = candidate_meta.drop(columns=["model_text"])

        multi = base.merge(top1, on="segment_row_idx", how="left")
        multi = multi.merge(top2, on="segment_row_idx", how="left")
        multi = multi.merge(candidate_meta, on="segment_row_idx", how="left")
        multi = multi[
            multi["n_topics_above_threshold"] >= int(self.cfg.multi_topic_min_topics)
        ].copy()
        multi["topic_prob_gap_top1_top2"] = (
            multi["top1_p_topic"] - multi["top2_p_topic"]
        )
        multi = multi.sort_values(
            ["n_topics_above_threshold", "top2_p_topic", "top1_p_topic"],
            ascending=[False, False, False],
        )
        multi.to_csv(self.cfg.output_dir / "tables" / "multi_topic_segments.csv", index=False)

        n_soft = int(base.shape[0])
        n_multi = int(multi.shape[0])
        return {
            "n_segments_with_soft_topics": n_soft,
            "n_segments_multi_topic": n_multi,
            "share_multi_topic_over_soft": (float(n_multi / n_soft) if n_soft else 0.0),
            "multi_topic_threshold": float(self.cfg.multi_topic_threshold),
            "multi_topic_min_topics": int(self.cfg.multi_topic_min_topics),
            "topic_id_source": topic_col,
            "excluded_topics_applied": list(self.cfg.excluded_topics),
        }

    def _export_multitopic_prevalence(self, probs: pd.DataFrame):
        year = (
            probs.groupby(["year", "topic_id"], as_index=False)["p_topic"]
            .sum()
            .rename(columns={"p_topic": "weighted_mass"})
        )
        year_total = year.groupby("year", as_index=False)["weighted_mass"].sum().rename(
            columns={"weighted_mass": "year_mass"}
        )
        year = year.merge(year_total, on="year", how="left")
        year["prevalence_pct"] = 100.0 * year["weighted_mass"] / year["year_mass"]
        if "topic_name" in probs.columns:
            names = probs[["topic_id", "topic_name"]].drop_duplicates("topic_id")
            year = year.merge(names, on="topic_id", how="left")
        year.to_csv(
            self.cfg.output_dir / "tables" / "soft_topic_prevalence_by_year_multitopic.csv",
            index=False,
        )

        decade = (
            probs.groupby(["decade", "topic_id"], as_index=False)["p_topic"]
            .sum()
            .rename(columns={"p_topic": "weighted_mass"})
        )
        decade_total = (
            decade.groupby("decade", as_index=False)["weighted_mass"]
            .sum()
            .rename(columns={"weighted_mass": "decade_mass"})
        )
        decade = decade.merge(decade_total, on="decade", how="left")
        decade["prevalence_pct"] = 100.0 * decade["weighted_mass"] / decade["decade_mass"]
        if "topic_name" in probs.columns:
            names = probs[["topic_id", "topic_name"]].drop_duplicates("topic_id")
            decade = decade.merge(names, on="topic_id", how="left")
        decade.to_csv(
            self.cfg.output_dir
            / "tables"
            / "soft_topic_prevalence_by_decade_multitopic.csv",
            index=False,
        )

    def build_summary(self, lifecycle, lifecycle_decade, pairs, multi_topic_summary):
        n_assigned_topic = int((self.segments_topics["topic_final"] != -1).sum())
        summary = {
            "n_segments_total": int(len(self.segments_topics)),
            "n_segments_assigned_topic": n_assigned_topic,
            "excluded_topics": list(self.cfg.excluded_topics),
            "n_segments_excluded_topics": int(self.n_excluded_topics_segments),
            "n_segments_analyzed": int(len(self.work)),
            "n_topics_detected": int(self.work["topic_final"].nunique()),
            "year_min": int(self.work["year"].min()),
            "year_max": int(self.work["year"].max()),
            "decade_min": int(self.work["decade"].min()),
            "decade_max": int(self.work["decade"].max()),
            "n_shows": int(self.work["show"].nunique()),
            "topic_birth_threshold_pct": float(self.cfg.birth_threshold_pct),
            "top_similarity_pairs_preview": pairs.head(10).to_dict(orient="records"),
            "topic_lifecycle_preview": lifecycle.head(10).to_dict(orient="records"),
            "topic_lifecycle_decade_preview": lifecycle_decade.head(10).to_dict(
                orient="records"
            ),
            "multi_topic_summary": multi_topic_summary,
            "note": (
                "Influence analysis intentionally removed due to insufficient evidence "
                "for causal claims."
            ),
        }
        with open(self.cfg.output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def run(self):
        self.load()
        self.build_working_table()
        self.analyze_soft_topic_prevalence()
        self.plot_top_topics_over_time(top_n=10)
        self.plot_top_topics_over_decades(top_n=10)
        lifecycle = self.analyze_topic_lifecycle()
        lifecycle_decade = self.analyze_topic_lifecycle_by_decade()
        _, pairs = self.analyze_show_similarity()
        multi_topic_summary = self.analyze_multi_topic_segments()
        summary = self.build_summary(lifecycle, lifecycle_decade, pairs, multi_topic_summary)

        print("\n=== Advanced analysis completed (without influence) ===")
        print(f"Output directory: {self.cfg.output_dir}")
        print(f"Segments total: {summary['n_segments_total']}")
        print(f"Assigned topic: {summary['n_segments_assigned_topic']}")
        print(f"Topics detected: {summary['n_topics_detected']}")
        print(f"Time span (year): {summary['year_min']} - {summary['year_max']}")
        print(f"Time span (decade): {summary['decade_min']} - {summary['decade_max']}")
        if multi_topic_summary is not None:
            print(
                "Multi-topic segments: "
                f"{multi_topic_summary['n_segments_multi_topic']} "
                f"(threshold >= {multi_topic_summary['multi_topic_threshold']})"
            )
        print(f"Shows: {summary['n_shows']}")
        print("\nTop similarity pairs:")
        print(pairs.head(10).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(
        description="Advanced analysis for topic-modeling outputs (without influence)."
    )
    p.add_argument("--input-tables-dir", type=str, default="outputs/topic_modeling/tables")
    p.add_argument("--output-dir", type=str, default="outputs/advanced_analysis")
    p.add_argument("--min-segments-per-show", type=int, default=15)
    p.add_argument("--birth-threshold-pct", type=float, default=1.0)
    p.add_argument("--similarity-top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--exclude-topics",
        type=str,
        default="3,6,17,22",
        help='Comma-separated topic IDs to exclude from analysis (e.g., "3,6,14").',
    )
    p.add_argument(
        "--multi-topic-threshold",
        type=float,
        default=0.08,
        help=(
            "Minimum per-topic probability used in multi-topic analysis from "
            "segment_topic_probs_long.csv."
        ),
    )
    p.add_argument(
        "--multi-topic-min-topics",
        type=int,
        default=2,
        help="Minimum number of topics above threshold to flag a segment as multi-topic.",
    )
    return p.parse_args()


def parse_topic_ids(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return tuple(sorted(set(values)))


def main():
    a = parse_args()
    cfg = Config(
        input_tables_dir=Path(a.input_tables_dir),
        output_dir=Path(a.output_dir),
        min_segments_per_show=a.min_segments_per_show,
        birth_threshold_pct=a.birth_threshold_pct,
        similarity_top_k=a.similarity_top_k,
        random_seed=a.seed,
        excluded_topics=parse_topic_ids(a.exclude_topics),
        multi_topic_threshold=float(a.multi_topic_threshold),
        multi_topic_min_topics=int(a.multi_topic_min_topics),
    )
    AdvancedTopicAnalyzer(cfg).run()


if __name__ == "__main__":
    main()
