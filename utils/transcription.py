import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import os

def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    if 'v=' in url:
        return parse_qs(urlparse(url).query).get('v', [None])[0]
    return url.split('/')[-1]

def get_transcript(video_id: str) -> str:
    """Fetch transcript using YouTube API"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript])
    except Exception as e:
        st.error(f"Error fetching transcript: {str(e)}")
        return None

def save_transcript(text: str, filename: str = "transcript.txt") -> None:
    """Save transcript to file"""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
        st.success(f"Transcript saved to {filename}")
    except Exception as e:
        st.error(f"Error saving transcript: {str(e)}")