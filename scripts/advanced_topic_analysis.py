#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def safe_numeric(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors='coerce')
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

class AdvancedTopicAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        ensure_dir(cfg.output_dir)
        ensure_dir(cfg.output_dir / 'tables')
        ensure_dir(cfg.output_dir / 'figures')
        self.rng = np.random.default_rng(cfg.random_seed)
        self.segments_topics = None
        self.topic_info = None
        self.segment_topic_probs_long = None
        self.show_year_topic_distribution = None
        self.work = None
        self.weighted_topic_year = None
        self.n_excluded_topics_segments = 0

    def load(self):
        tdir = self.cfg.input_tables_dir
        self.segments_topics = read_csv(tdir / 'segments_topics.csv', required=True)
        self.topic_info = read_csv(tdir / 'topic_info.csv', required=False)
        self.segment_topic_probs_long = read_csv(tdir / 'segment_topic_probs_long.csv', required=False)
        self.show_year_topic_distribution = read_csv(tdir / 'show_year_topic_distribution.csv', required=False)

        for col in ['show', 'year', 'topic_final']:
            if col not in self.segments_topics.columns:
                raise ValueError(f"`segments_topics.csv` missing column: {col}")

        self.segments_topics['year'] = safe_numeric(self.segments_topics['year'])
        self.segments_topics = self.segments_topics.dropna(subset=['year']).copy()
        self.segments_topics['year'] = self.segments_topics['year'].astype(int)

        if 'max_topic_probability' not in self.segments_topics.columns:
            self.segments_topics['max_topic_probability'] = 1.0
        else:
            self.segments_topics['max_topic_probability'] = safe_numeric(
                self.segments_topics['max_topic_probability'], default=1.0
            ).clip(0, 1)

    def build_working_table(self):
        df = self.segments_topics.copy()
        df = df[df['topic_final'] != -1].copy()
        if self.cfg.excluded_topics:
            before = len(df)
            df = df[~df['topic_final'].isin(self.cfg.excluded_topics)].copy()
            self.n_excluded_topics_segments = before - len(df)
        df['topic_weight'] = df['max_topic_probability'].fillna(1.0).clip(0, 1)
        df['decade'] = (df['year'] // 10) * 10
        self.work = df

    def analyze_soft_topic_prevalence(self):
        w = self.work.copy()
        topic_year = (
            w.groupby(['year', 'topic_final'], as_index=False)['topic_weight']
            .sum()
            .rename(columns={'topic_weight': 'weighted_mass'})
        )
        year_total = topic_year.groupby('year', as_index=False)['weighted_mass'].sum().rename(
            columns={'weighted_mass': 'year_mass'}
        )
        topic_year = topic_year.merge(year_total, on='year', how='left')
        topic_year['prevalence_pct'] = 100.0 * topic_year['weighted_mass'] / topic_year['year_mass']

        if 'topic_name_final' in self.work.columns:
            names = self.work[['topic_final', 'topic_name_final']].drop_duplicates('topic_final')
            topic_year = topic_year.merge(names, on='topic_final', how='left')

        self.weighted_topic_year = topic_year
        topic_year.to_csv(self.cfg.output_dir / 'tables' / 'soft_topic_prevalence_by_year.csv', index=False)
        return topic_year

    def plot_top_topics_over_time(self, top_n=10):
        t = self.weighted_topic_year.copy()
        top_topics = (
            t.groupby('topic_final')['weighted_mass'].sum()
            .sort_values(ascending=False).head(top_n).index.tolist()
        )
        p = t[t['topic_final'].isin(top_topics)].pivot_table(
            index='year', columns='topic_final', values='prevalence_pct', fill_value=0.0
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        p.plot(ax=ax)
        ax.set_title(f"Soft topic prevalence over time (top {top_n})")
        ax.set_xlabel('Year')
        ax.set_ylabel('Prevalence (%)')
        ax.grid(alpha=0.25)
        save_plot(fig, self.cfg.output_dir / 'figures' / 'soft_topic_prevalence_top_topics.png')

    def analyze_topic_lifecycle(self):
        t = self.weighted_topic_year[['year', 'topic_final', 'prevalence_pct']].copy()
        rows = []
        for topic_id, g in t.groupby('topic_final'):
            g = g.sort_values('year')
            active = g[g['prevalence_pct'] >= self.cfg.birth_threshold_pct]
            rows.append({
                'topic_final': int(topic_id),
                'birth_year': int(active['year'].min()) if len(active) else np.nan,
                'active_years': int(active['year'].nunique()) if len(active) else 0,
                'volatility_mean_abs_diff': float(g['prevalence_pct'].diff().abs().mean(skipna=True)),
                'mean_prevalence_pct': float(g['prevalence_pct'].mean()),
                'max_prevalence_pct': float(g['prevalence_pct'].max()),
            })
        out = pd.DataFrame(rows).sort_values(['birth_year', 'mean_prevalence_pct'], ascending=[True, False])
        out.to_csv(self.cfg.output_dir / 'tables' / 'topic_lifecycle_metrics.csv', index=False)
        return out

    def analyze_show_similarity(self):
        w = self.work.copy()
        show_sizes = w.groupby('show').size().rename('n_segments').reset_index()
        eligible = show_sizes[show_sizes['n_segments'] >= self.cfg.min_segments_per_show]['show']
        sw = w[w['show'].isin(eligible)].copy()

        show_topic = sw.groupby(['show', 'topic_final'], as_index=False)['topic_weight'].sum().rename(columns={'topic_weight': 'mass'})
        show_total = show_topic.groupby('show', as_index=False)['mass'].sum().rename(columns={'mass': 'show_mass'})
        show_topic = show_topic.merge(show_total, on='show', how='left')
        show_topic['p_topic'] = show_topic['mass'] / show_topic['show_mass']

        M = show_topic.pivot_table(index='show', columns='topic_final', values='p_topic', fill_value=0.0)
        if M.shape[0] < 2:
            raise ValueError('Not enough shows for similarity analysis.')

        sim = cosine_similarity(M.values)
        sim_df = pd.DataFrame(sim, index=M.index, columns=M.index)
        sim_df.to_csv(self.cfg.output_dir / 'tables' / 'show_similarity_matrix.csv')

        tri_i, tri_j = np.triu_indices_from(sim, k=1)
        pairs = pd.DataFrame({
            'show_a': [M.index[i] for i in tri_i],
            'show_b': [M.index[j] for j in tri_j],
            'cosine_similarity': sim[tri_i, tri_j]
        }).sort_values('cosine_similarity', ascending=False)
        pairs.to_csv(self.cfg.output_dir / 'tables' / 'show_similarity_pairs.csv', index=False)
        pairs.head(self.cfg.similarity_top_k).to_csv(self.cfg.output_dir / 'tables' / 'show_similarity_pairs_topk.csv', index=False)

        pca = PCA(n_components=2, random_state=self.cfg.random_seed)
        coords = pca.fit_transform(M.values)
        coords_df = pd.DataFrame(coords, index=M.index, columns=['x', 'y']).reset_index().rename(columns={'index': 'show'})
        coords_df = coords_df.merge(show_sizes, on='show', how='left')
        coords_df.to_csv(self.cfg.output_dir / 'tables' / 'show_topic_space_pca.csv', index=False)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(coords_df['x'], coords_df['y'], s=20 + np.sqrt(coords_df['n_segments']) * 10, alpha=0.8)
        for _, r in coords_df.iterrows():
            ax.text(r['x'], r['y'], r['show'], fontsize=8, alpha=0.8)
        ax.set_title('Show similarity space (topic-distribution PCA)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(alpha=0.25)
        save_plot(fig, self.cfg.output_dir / 'figures' / 'show_topic_space_pca.png')

        return sim_df, pairs

    def build_summary(self, lifecycle, pairs):
        n_assigned_topic = int((self.segments_topics['topic_final'] != -1).sum())
        summary = {
            'n_segments_total': int(len(self.segments_topics)),
            'n_segments_assigned_topic': n_assigned_topic,
            'excluded_topics': list(self.cfg.excluded_topics),
            'n_segments_excluded_topics': int(self.n_excluded_topics_segments),
            'n_segments_analyzed': int(len(self.work)),
            'n_topics_detected': int(self.work['topic_final'].nunique()),
            'year_min': int(self.work['year'].min()),
            'year_max': int(self.work['year'].max()),
            'n_shows': int(self.work['show'].nunique()),
            'topic_birth_threshold_pct': float(self.cfg.birth_threshold_pct),
            'top_similarity_pairs_preview': pairs.head(10).to_dict(orient='records'),
            'topic_lifecycle_preview': lifecycle.head(10).to_dict(orient='records'),
            'note': 'Influence analysis intentionally removed due to insufficient evidence for causal claims.'
        }
        with open(self.cfg.output_dir / 'analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def run(self):
        self.load()
        self.build_working_table()
        self.analyze_soft_topic_prevalence()
        self.plot_top_topics_over_time(top_n=10)
        lifecycle = self.analyze_topic_lifecycle()
        _, pairs = self.analyze_show_similarity()
        summary = self.build_summary(lifecycle, pairs)

        print("\n=== Advanced analysis completed (without influence) ===")
        print(f"Output directory: {self.cfg.output_dir}")
        print(f"Segments total: {summary['n_segments_total']}")
        print(f"Assigned topic: {summary['n_segments_assigned_topic']}")
        print(f"Topics detected: {summary['n_topics_detected']}")
        print(f"Time span: {summary['year_min']} - {summary['year_max']}")
        print(f"Shows: {summary['n_shows']}")
        print("\nTop similarity pairs:")
        print(pairs.head(10).to_string(index=False))

def parse_args():
    p = argparse.ArgumentParser(description='Advanced analysis for topic-modeling outputs (without influence).')
    p.add_argument('--input-tables-dir', type=str, default='outputs/topic_modeling/tables')
    p.add_argument('--output-dir', type=str, default='outputs/advanced_analysis')
    p.add_argument('--min-segments-per-show', type=int, default=15)
    p.add_argument('--birth-threshold-pct', type=float, default=1.0)
    p.add_argument('--similarity-top-k', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--exclude-topics',
        type=str,
        default='3,6,17,22',
        help='Comma-separated topic IDs to exclude from analysis (e.g., "3,6,14").',
    )
    return p.parse_args()

def parse_topic_ids(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    values = []
    for item in raw.split(','):
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
    )
    AdvancedTopicAnalyzer(cfg).run()

if __name__ == '__main__':
    main()
