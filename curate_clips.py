#!/usr/bin/env python3
"""
Streamlit interface for curating audio clips dataset.

This interface allows you to:
- Play audio clips
- Edit transcriptions
- Select/deselect clips for the final dataset
- Auto-save all changes to disk
"""

import json
import os
from pathlib import Path
import streamlit as st
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Configuration
METADATA_FILE = './clips_metadata.json'
CURATED_METADATA_FILE = './clips_metadata_curated.json'


def load_metadata(file_path: str) -> List[Dict]:
    """Load clips metadata from JSON file."""
    if not os.path.exists(file_path):
        st.error(f"Metadata file not found: {file_path}")
        st.info("Please run `python3 process_clips.py` first to generate the clips.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add 'selected' field if not present
    for item in data:
        if 'selected' not in item:
            item['selected'] = True

    return data


def save_metadata(data: List[Dict], file_path: str) -> None:
    """Save clips metadata to JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metadata to {file_path}")


def get_selected_clips(data: List[Dict]) -> List[Dict]:
    """Filter and return only selected clips."""
    return [clip for clip in data if clip.get('selected', True)]


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'metadata' not in st.session_state:
        st.session_state.metadata = load_metadata(METADATA_FILE)

    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

    if 'clips_per_page' not in st.session_state:
        st.session_state.clips_per_page = 10

    if 'search_filter' not in st.session_state:
        st.session_state.search_filter = ""

    if 'show_only_selected' not in st.session_state:
        st.session_state.show_only_selected = False


def update_clip(index: int, field: str, value) -> None:
    """Update a specific field of a clip and save to disk immediately."""
    st.session_state.metadata[index][field] = value
    save_metadata(st.session_state.metadata, METADATA_FILE)
    logger.info(f"Updated clip {index}, field '{field}'")


def export_selected_clips():
    """Export only selected clips to a new file."""
    selected = get_selected_clips(st.session_state.metadata)
    save_metadata(selected, CURATED_METADATA_FILE)
    return len(selected)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Clip Curation Interface",
        page_icon="🎤",
        layout="wide"
    )

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🎤 Chilean Humor Clip Curation Interface")
    st.markdown("---")

    # Check if metadata exists
    if not st.session_state.metadata:
        st.error("No metadata found. Please run the clip processing script first.")
        st.code("python3 process_clips.py", language="bash")
        return

    # Sidebar - Statistics and Controls
    with st.sidebar:
        st.header("📊 Statistics")

        total_clips = len(st.session_state.metadata)
        selected_clips = len(get_selected_clips(st.session_state.metadata))

        st.metric("Total Clips", total_clips)
        st.metric("Selected Clips", selected_clips)
        st.metric("Excluded Clips", total_clips - selected_clips)

        st.markdown("---")

        st.header("⚙️ Settings")

        # Clips per page
        clips_per_page = st.selectbox(
            "Clips per page",
            options=[5, 10, 20, 50, 100],
            index=1
        )
        st.session_state.clips_per_page = clips_per_page

        # Filter options
        st.markdown("---")
        st.header("🔍 Filters")

        show_only_selected = st.checkbox(
            "Show only selected clips",
            value=st.session_state.show_only_selected
        )
        st.session_state.show_only_selected = show_only_selected

        search_filter = st.text_input(
            "Search in transcripts",
            value=st.session_state.search_filter,
            placeholder="Enter search term..."
        )
        st.session_state.search_filter = search_filter

        # Export button
        st.markdown("---")
        st.header("💾 Export")

        if st.button("Export Selected Clips", type="primary", use_container_width=True):
            count = export_selected_clips()
            st.success(f"✅ Exported {count} selected clips to:\n`{CURATED_METADATA_FILE}`")

        # Bulk actions
        st.markdown("---")
        st.header("🔧 Bulk Actions")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Select All", use_container_width=True):
                for i in range(len(st.session_state.metadata)):
                    st.session_state.metadata[i]['selected'] = True
                save_metadata(st.session_state.metadata, METADATA_FILE)
                st.rerun()

        with col2:
            if st.button("Deselect All", use_container_width=True):
                for i in range(len(st.session_state.metadata)):
                    st.session_state.metadata[i]['selected'] = False
                save_metadata(st.session_state.metadata, METADATA_FILE)
                st.rerun()

    # Apply filters
    filtered_metadata = st.session_state.metadata

    if st.session_state.show_only_selected:
        filtered_metadata = [
            (i, clip) for i, clip in enumerate(st.session_state.metadata)
            if clip.get('selected', True)
        ]
    else:
        filtered_metadata = list(enumerate(st.session_state.metadata))

    if st.session_state.search_filter:
        search_term = st.session_state.search_filter.lower()
        filtered_metadata = [
            (i, clip) for i, clip in filtered_metadata
            if search_term in clip.get('transcript', '').lower()
        ]

    # Pagination
    total_filtered = len(filtered_metadata)
    total_pages = (total_filtered + clips_per_page - 1) // clips_per_page

    if total_filtered == 0:
        st.warning("No clips match the current filters.")
        return

    st.info(f"Showing {total_filtered} clips (filtered from {total_clips} total)")

    # Page navigation
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])

    with col1:
        if st.button("⏮️ First", disabled=st.session_state.current_page == 0):
            st.session_state.current_page = 0
            st.rerun()

    with col2:
        if st.button("◀️ Prev", disabled=st.session_state.current_page == 0):
            st.session_state.current_page -= 1
            st.rerun()

    with col3:
        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page {st.session_state.current_page + 1} of {total_pages}</div>", unsafe_allow_html=True)

    with col4:
        if st.button("Next ▶️", disabled=st.session_state.current_page >= total_pages - 1):
            st.session_state.current_page += 1
            st.rerun()

    with col5:
        if st.button("Last ⏭️", disabled=st.session_state.current_page >= total_pages - 1):
            st.session_state.current_page = total_pages - 1
            st.rerun()

    st.markdown("---")

    # Display clips for current page
    start_idx = st.session_state.current_page * clips_per_page
    end_idx = min(start_idx + clips_per_page, total_filtered)

    page_clips = filtered_metadata[start_idx:end_idx]

    for original_idx, clip in page_clips:
        with st.container():
            # Header row: checkbox and clip info
            col1, col2 = st.columns([1, 11])

            with col1:
                selected = st.checkbox(
                    "Include",
                    value=clip.get('selected', True),
                    key=f"selected_{original_idx}",
                    label_visibility="collapsed"
                )
                if selected != clip.get('selected', True):
                    update_clip(original_idx, 'selected', selected)
                    st.rerun()

            with col2:
                # Clip header
                audio_filename = os.path.basename(clip['audio_path'])
                routine_id = clip.get('routine_id', 'N/A')
                duration = clip.get('duration', 0)

                status_emoji = "✅" if clip.get('selected', True) else "❌"

                st.markdown(f"### {status_emoji} Clip #{original_idx + 1}: `{audio_filename}`")
                st.caption(f"Routine ID: {routine_id} | Duration: {duration:.1f}s | Time: {clip.get('start_time', 0):.1f}s - {clip.get('end_time', 0):.1f}s")

            # Audio player and transcript
            col1, col2 = st.columns([1, 2])

            with col1:
                # Audio player
                audio_path = clip['audio_path']
                if os.path.exists(audio_path):
                    with open(audio_path, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                else:
                    st.error(f"Audio file not found: `{audio_path}`")

            with col2:
                # Editable transcript
                transcript = st.text_area(
                    "Transcript",
                    value=clip.get('transcript', ''),
                    height=150,
                    key=f"transcript_{original_idx}",
                    label_visibility="collapsed",
                    placeholder="Transcript text..."
                )

                # Check if transcript changed and save immediately
                if transcript != clip.get('transcript', ''):
                    update_clip(original_idx, 'transcript', transcript)

            st.markdown("---")


if __name__ == '__main__':
    main()
