import streamlit as st
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI
import tempfile
import os
import re

# --- Configuration ---
st.set_page_config(page_title="NVIDIA AI Book Reader", layout="wide", page_icon="📗")

# --- Persistence Logic ---
KEY_FILE = ".env"

def load_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f:
            return f.read().strip()
    return ""

def save_key(key):
    with open(KEY_FILE, "w") as f:
        f.write(key)

# --- Custom Styling ---
def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700;900&display=swap');
        html, body, [class*="css"] { font-family: 'Merriweather', serif; }
        .streamlit-expanderHeader { font-size: 1.2rem; font-weight: 700; color: #FAFAFA; background-color: #2D2D2D; border-radius: 8px; }
        
        .summary-box { 
            background-color: #2D2D2D; 
            padding: 20px; 
            border-radius: 8px; 
            border: 1px solid #76b900; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
            line-height: 1.6; 
        }

        /* Target HTML <b> tags for bolding */
        b, strong { 
            color: #76b900 !important; 
            font-weight: 700; 
        }
        
        h1, h2, h3 { font-weight: 900 !important; letter-spacing: -0.5px; }
        </style>
    """, unsafe_allow_html=True)

apply_custom_style()

# --- Helper Functions ---

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text()

def extract_title(soup, filename):
    """
    Smart Title Extractor:
    1. Looks for H1/H2 tags.
    2. Looks for elements with class='title' or 'chapter'.
    3. Fallback: Uses the first short line of text.
    """
    # 1. Standard Header Search
    header = soup.find(['h1', 'h2', 'h3'])
    if header:
        return header.get_text().strip()[:60]
    
    # 2. Class-based Search (e.g. <p class="chapter-title">)
    for tag in soup.find_all(True, class_=re.compile(r'(title|chapter|head)', re.I)):
        text = tag.get_text().strip()
        if 3 < len(text) < 60: # Valid title length
            return text

    # 3. First Line Heuristic (Grab first significant line)
    text_content = soup.get_text().strip()
    if text_content:
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        if lines:
            first_line = lines[0]
            # If the first line is short (like "Chapter 1"), assume it's the title
            if len(first_line) < 50:
                return first_line

    # 4. Fallback to clean filename
    return f"Section ({filename.split('/')[-1]})"

def parse_epub(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        book = epub.read_epub(tmp_path)
        chapters = []
        junk_pattern = re.compile(r'(cover|title|copyright|toc|contents|dedication|ack)', re.IGNORECASE)

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                if junk_pattern.search(item.get_name()):
                    continue
                
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                word_count = len(text.split())

                # Strict Length Filter
                if word_count > 300:
                    real_title = extract_title(soup, item.get_name())
                    chapters.append({'title': real_title, 'content': text, 'words': word_count})
        return chapters
    except Exception as e:
        st.error(f"Error parsing EPUB: {e}")
        return []
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def calculate_time(word_count, wpm):
    minutes = word_count / wpm
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

def summarize_text(text, api_key):
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        
        # PROMPT: Use HTML tags for bolding
        prompt = f"""
        Summarize this book excerpt.
        CRITICAL INSTRUCTION: Do NOT use Markdown asterisks (**). 
        Instead, use HTML <b> tags (e.g. <b>Name</b>) for <b>Characters</b>, <b>Locations</b>, and <b>Themes</b>.
        
        TEXT: {text[:40000]} 
        """
        
        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"NVIDIA API Error: {str(e)}"

# --- Main App Interface ---

st.title("📗 NVIDIA AI Book Reader")
st.caption("Powered by NVIDIA NIM (Llama 3.1 70B)")

# --- Settings ---
with st.expander("⚙️ Settings & API Key", expanded=True):
    col1, col2 = st.columns([2, 1])
    with col1:
        saved_key = load_key()
        api_key_input = st.text_input("NVIDIA API Key", value=saved_key, type="password", placeholder="nvapi-...")
        st.markdown("Get 1,000 free credits at [build.nvidia.com](https://build.nvidia.com/explore/discover)")
        
        save_choice = st.checkbox("Remember my API Key", value=bool(saved_key))
        if save_choice and api_key_input:
            if api_key_input != saved_key:
                save_key(api_key_input)
                st.toast("Key saved!", icon="💾")
        elif not save_choice and os.path.exists(KEY_FILE):
            os.remove(KEY_FILE)
            
    with col2:
        wpm = st.number_input("Reading Speed (WPM)", 50, 1000, 250, 10)

st.divider()

# --- Upload & Process ---
uploaded_file = st.file_uploader("Drop EPUB file here", type=["epub"])

if uploaded_file:
    if "summaries" not in st.session_state:
        st.session_state.summaries = {}
        
    with st.spinner("Analyzing book structure..."):
        chapters = parse_epub(uploaded_file)
        
    if chapters:
        total_words = sum(ch['words'] for ch in chapters)
        m1, m2, m3 = st.columns(3)
        m1.metric("Real Chapters", len(chapters))
        m2.metric("Total Words", f"{total_words:,}")
        m3.metric("Est. Time", calculate_time(total_words, wpm))
        
        st.write("---")
        
        st.subheader("Select Chapters")
        
        # Mapping titles for slider
        chapter_map = {i: ch['title'] for i, ch in enumerate(chapters)}
        
        start_idx, end_idx = st.select_slider(
            "Select Range:",
            options=range(len(chapters)),
            value=(0, min(2, len(chapters)-1)), 
            format_func=lambda x: chapter_map[x]
        )
        
        selected_chapters = chapters[start_idx : end_idx + 1]
        
        if st.button("Generate Summaries", type="primary", use_container_width=True):
            if not api_key_input:
                st.error("Please enter NVIDIA API Key in settings.")
            else:
                progress_bar = st.progress(0)
                status = st.empty()
                
                for i, chapter in enumerate(selected_chapters):
                    idx = start_idx + i
                    status.text(f"Processing: {chapter['title']}...")
                    summary = summarize_text(chapter['content'], api_key_input)
                    st.session_state.summaries[idx] = summary
                    progress_bar.progress((i + 1) / len(selected_chapters))
                
                status.empty()
                st.rerun()

        if st.session_state.summaries:
            st.write("---")
            st.subheader("Analysis Results")
            full_text_export = f"Analysis for {uploaded_file.name}\n\n"
            
            for idx, text in st.session_state.summaries.items():
                title = chapters[idx]['title']
                st.markdown(f"### {title}")
                # Render HTML safely
                st.markdown(f'<div class="summary-box">{text}</div>', unsafe_allow_html=True)
                full_text_export += f"--- {title} ---\n{text}\n\n"
            
            st.download_button(
                label="Download Summaries (.txt)",
                data=full_text_export,
                file_name=f"{uploaded_file.name}_summary.txt",
                mime="text/plain",
                type="primary"
            )