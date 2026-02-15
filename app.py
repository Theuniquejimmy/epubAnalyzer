import streamlit as st
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI
import tempfile
import os
import re
import json

# --- Configuration ---
st.set_page_config(page_title="NVIDIA AI Book Reader", layout="wide", page_icon="📗")

# --- Persistence Logic (JSON) ---
CONFIG_FILE = "config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"api_key": "", "wpm": 250}

def save_config(api_key, wpm):
    config = {"api_key": api_key, "wpm": wpm}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

user_config = load_config()

# --- Custom Styling ---
def apply_custom_style():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@300;400;700;900&display=swap');
        html, body, [class*="css"] { font-family: 'Merriweather', serif; }
        .streamlit-expanderHeader { font-size: 1.1rem; font-weight: 700; color: #FAFAFA; background-color: #2D2D2D; border-radius: 8px; }
        
        .summary-box { 
            background-color: #2D2D2D; 
            padding: 30px; 
            border-radius: 8px; 
            border: 1px solid #76b900; 
            margin-bottom: 20px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.3); 
            line-height: 1.8; 
            font-size: 1.1rem;
            color: #E0E0E0;
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
    header = soup.find(['h1', 'h2', 'h3'])
    if header: return header.get_text().strip()[:60]
    for tag in soup.find_all(True, class_=re.compile(r'(title|chapter|head)', re.I)):
        text = tag.get_text().strip()
        if 3 < len(text) < 60: return text
    text_content = soup.get_text().strip()
    if text_content:
        lines = [line.strip() for line in text_content.split('\n') if line.strip()]
        if lines and len(lines[0]) < 50: return lines[0]
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
                if junk_pattern.search(item.get_name()): continue
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                word_count = len(text.split())
                if word_count > 300:
                    real_title = extract_title(soup, item.get_name())
                    chapters.append({'title': real_title, 'content': text, 'words': word_count})
        return chapters
    except Exception as e:
        st.error(f"Error parsing EPUB: {e}")
        return []
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def calculate_time(word_count, wpm):
    minutes = word_count / wpm
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m" if hours > 0 else f"{mins}m"

def summarize_batch(text, api_key, title_list):
    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        
        # --- PLOT-FOCUSED + BOLDING PROMPT ---
        prompt = f"""
        You are a plot summarizer. Read the following text (covering chapters: {', '.join(title_list)}).
        
        INSTRUCTIONS:
        1. Write a chronological summary of the **PLOT EVENTS**. (Step 1 -> Step 2 -> Step 3).
        2. Keep language simple and direct. Do not write a review.
        3. **FORMATTING RULE:** You MUST use HTML <b> tags to bold the following EVERY TIME they appear:
           - <b>Character Names</b> (e.g. <b>Harry</b>, <b>Hermione</b>)
           - <b>Specific Locations</b> (e.g. <b>Hogwarts</b>)
           - <b>Key Themes</b> (e.g. <b>Betrayal</b>)
        
        TEXT CONTENT:
        {text[:150000]} 
        """
        
        completion = client.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes stories chronologically and bolds names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3, # Low temp = factual, focused on the instructions
            top_p=1,
            max_tokens=2048,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"NVIDIA API Error: {str(e)}"

# --- Main App Interface ---

st.title("📗 NVIDIA AI Book Reader")
st.caption("Context-Aware Analysis (Llama 3.1 70B)")

# --- Settings (Collapsed by default) ---
with st.expander("⚙️ Settings & API Key", expanded=False):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key_input = st.text_input(
            "NVIDIA API Key", 
            value=user_config.get("api_key", ""), 
            type="password", 
            placeholder="nvapi-..."
        )
        st.markdown("Get 1,000 free credits at [build.nvidia.com](https://build.nvidia.com/explore/discover)")
            
    with col2:
        wpm_input = st.number_input(
            "Reading Speed (WPM)", 
            min_value=50, 
            max_value=1000, 
            value=int(user_config.get("wpm", 250)), 
            step=10
        )
    
    if st.button("Save Settings"):
        save_config(api_key_input, wpm_input)
        st.success("Settings saved!", icon="💾")
        user_config["api_key"] = api_key_input
        user_config["wpm"] = wpm_input

st.divider()

# --- Upload & Process ---
uploaded_file = st.file_uploader("Drop EPUB file here", type=["epub"])

if uploaded_file:
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
        
    with st.spinner("Analyzing book structure..."):
        chapters = parse_epub(uploaded_file)
        
    if chapters:
        total_words = sum(ch['words'] for ch in chapters)
        m1, m2, m3 = st.columns(3)
        m1.metric("Real Chapters", len(chapters))
        m2.metric("Total Words", f"{total_words:,}")
        
        current_wpm = wpm_input if wpm_input else 250
        m3.metric("Est. Time", calculate_time(total_words, current_wpm))
        
        st.write("---")
        
        st.subheader("Select Scope")
        
        chapter_map = {i: ch['title'] for i, ch in enumerate(chapters)}
        start_idx, end_idx = st.select_slider(
            "Select Range to Analyze as One Block:",
            options=range(len(chapters)),
            value=(0, min(4, len(chapters)-1)), 
            format_func=lambda x: chapter_map[x]
        )
        
        selected_chapters = chapters[start_idx : end_idx + 1]
        
        batch_words = sum(ch['words'] for ch in selected_chapters)
        batch_titles = [ch['title'] for ch in selected_chapters]
        
        st.info(f"Selected **{len(selected_chapters)} chapters** ({batch_words:,} words). The AI will read this as a single continuous story.")
        
        if st.button("Generate Plot Summary", type="primary", use_container_width=True):
            key_to_use = api_key_input
            
            if not key_to_use:
                st.error("Please enter NVIDIA API Key in settings.")
            else:
                st.session_state.analysis_result = None 
                
                combined_text = "\n\n".join([ch['content'] for ch in selected_chapters])
                
                with st.spinner(f"Reading {len(selected_chapters)} chapters and extracting plot points..."):
                    summary = summarize_batch(combined_text, key_to_use, batch_titles)
                    st.session_state.analysis_result = {
                        "range": f"{batch_titles[0]} - {batch_titles[-1]}",
                        "text": summary
                    }

        # Display Result
        if st.session_state.analysis_result:
            st.write("---")
            st.subheader(f"Plot Summary: {st.session_state.analysis_result['range']}")
            st.markdown(f'<div class="summary-box">{st.session_state.analysis_result["text"]}</div>', unsafe_allow_html=True)
            
            st.download_button(
                label="Download Analysis (.txt)",
                data=st.session_state.analysis_result["text"],
                file_name=f"analysis_summary.txt",
                mime="text/plain",
                type="primary"
            )