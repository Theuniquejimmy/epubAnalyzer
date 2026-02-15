import streamlit as st
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI
import tempfile
import os
import re
import json
from collections import defaultdict

# --- Configuration ---
st.set_page_config(page_title="NVIDIA AI Book Reader", layout="wide", page_icon="📗")

# --- Persistence Logic ---
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

        b, strong { color: #76b900 !important; font-weight: 700; }
        em, i { color: #A0C0FF; font-style: italic; }
        u { text-decoration-color: #76b900; text-decoration-thickness: 2px; }
        h1, h2, h3 { font-weight: 900 !important; letter-spacing: -0.5px; }
        
        .preview-text {
            font-family: 'Courier New', monospace;
            background-color: #1E1E1E;
            padding: 15px;
            border-left: 4px solid #76b900;
            border-radius: 4px;
            font-size: 0.85rem;
            height: 150px;
            overflow-y: auto;
            color: #ccc;
        }
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
    return f"Section"

def get_metadata(book):
    try: title = book.get_metadata('DC', 'title')[0][0]
    except: title = "Unknown Title"
    try: creator = book.get_metadata('DC', 'creator')[0][0]
    except: creator = "Unknown Author"
    return title, creator

def calculate_reading_time(word_count, wpm):
    """Returns a formatted string of reading time."""
    if wpm <= 0: wpm = 250
    minutes = word_count / wpm
    if minutes < 60:
        return f"{int(minutes)} min"
    else:
        hours = int(minutes // 60)
        mins = int(minutes % 60)
        return f"{hours}h {mins}m"

def parse_epub_to_pages(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        book = epub.read_epub(tmp_path)
        meta_title, meta_author = get_metadata(book)
        
        all_pages = []
        chapter_map = [] 
        junk_pattern = re.compile(r'(cover|copyright|dedication|ack)', re.IGNORECASE)

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                if junk_pattern.search(item.get_name()): continue
                
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                raw_text = soup.get_text()
                
                if len(raw_text) < 200: continue 
                
                chapter_title = extract_title(soup, item.get_name())
                
                if not all_pages or all_pages[-1]['chapter'] != chapter_title:
                    current_page_idx = len(all_pages) + 1
                    chapter_map.append({"title": chapter_title, "start_page": current_page_idx})
                
                chunk_size = 1500 
                for i in range(0, len(raw_text), chunk_size):
                    chunk_text = raw_text[i : i + chunk_size]
                    all_pages.append({
                        'chapter': chapter_title,
                        'content': chunk_text,
                        'id': len(all_pages) + 1
                    })
                    
        return all_pages, chapter_map, meta_title, meta_author
    except Exception as e:
        st.error(f"Error parsing EPUB: {e}")
        return [], [], "Unknown", "Unknown"
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

def clear_results():
    st.session_state.analysis_result = None

def update_slider_from_dropdowns():
    start_title = st.session_state.start_chapter_select
    end_title = st.session_state.end_chapter_select
    
    titles = [c['title'] for c in st.session_state.chapter_map]
    
    try:
        start_idx = titles.index(start_title)
        end_idx = titles.index(end_title)
    except ValueError:
        return 
    
    if end_idx < start_idx:
        end_idx = start_idx
    
    start_page = st.session_state.chapter_map[start_idx]['start_page']
    
    if end_idx < len(st.session_state.chapter_map) - 1:
        end_page = st.session_state.chapter_map[end_idx + 1]['start_page'] - 1
    else:
        end_page = len(st.session_state.pages)
        
    if end_page < start_page: end_page = start_page
    
    st.session_state.page_slider = (start_page, end_page)
    clear_results()

# --- AI FUNCTIONS (Segmented to guarantee scope) ---

def summarize_segmented(selected_pages, api_key, book_title, book_author):
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    
    chapter_dict = defaultdict(list)
    for p in selected_pages:
        chapter_dict[p['chapter']].append(p['content'])
        
    full_summary = ""
    progress_bar = st.progress(0)
    total_chapters = len(chapter_dict)
    
    for idx, (chap_title, content_list) in enumerate(chapter_dict.items()):
        progress_bar.progress((idx) / total_chapters)
        chap_text = "\n".join(content_list)
        
        prompt = f"""
        You are summarizing **{chap_title}** from **{book_title}**.
        Write a **Detailed Narrative Summary** (3-5 sentences).
        No bullets. Narrative paragraph only. Bold names.
        
        TEXT:
        {chap_text[:30000]}
        """
        try:
            completion = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, top_p=1, max_tokens=1024, stream=False
            )
            full_summary += f"### {chap_title}\n{completion.choices[0].message.content}\n\n"
        except Exception as e:
            full_summary += f"### {chap_title}\n[Error: {str(e)}]\n\n"

    progress_bar.progress(100)
    return full_summary

def xray_segmented(selected_pages, api_key, book_title, book_author):
    """
    Runs X-Ray analysis chapter-by-chapter to ensure full scope coverage.
    """
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
    
    chapter_dict = defaultdict(list)
    for p in selected_pages:
        chapter_dict[p['chapter']].append(p['content'])
        
    full_xray = ""
    progress_bar = st.progress(0)
    total_chapters = len(chapter_dict)
    
    for idx, (chap_title, content_list) in enumerate(chapter_dict.items()):
        progress_bar.progress((idx) / total_chapters)
        chap_text = "\n".join(content_list)
        
        prompt = f"""
        Perform a **Deep X-Ray Analysis** on **{chap_title}** of **{book_title}**.
        
        REQUIRED STRUCTURE (Markdown):
        ### Characters (In this chapter)
        - **Name** — [Role/Action in this specific chapter]
        
        ### Timeline
        - **Scene:** [What happened in this chapter?]

        ### Re-immersion
        * **Conflict:** [Current tension]

        FORMATTING: Use `_` and `<u>`. **Bold** names. Keep it concise.
        
        TEXT:
        {chap_text[:30000]}
        """
        try:
            completion = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "system", "content": "You are an expert literary analyst."}, {"role": "user", "content": prompt}],
                temperature=0.4, top_p=1, max_tokens=2048, stream=False
            )
            # Add Header
            full_xray += f"## 🔎 {chap_title}\n{completion.choices[0].message.content}\n\n---\n\n"
        except Exception as e:
            full_xray += f"### {chap_title}\n[Error: {str(e)}]\n\n"

    progress_bar.progress(100)
    return full_xray

# --- Main App Interface ---

st.title("📗 NVIDIA AI Book Reader")
st.caption("Literary Assistant (Llama 3.1 70B)")

# --- Settings ---
with st.expander("⚙️ Settings & API Key", expanded=False):
    col1, col2 = st.columns([2, 1])
    with col1:
        api_key_input = st.text_input("NVIDIA API Key", value=user_config.get("api_key", ""), type="password", placeholder="nvapi-...")
        st.markdown("Get 1,000 free credits at [build.nvidia.com](https://build.nvidia.com/explore/discover)")
    with col2:
        wpm_input = st.number_input("Reading Speed (WPM)", min_value=50, max_value=1000, value=int(user_config.get("wpm", 250)), step=10)
    
    if st.button("Save Settings"):
        save_config(api_key_input, wpm_input)
        st.success("Settings saved!", icon="💾")
        user_config["api_key"] = api_key_input
        user_config["wpm"] = wpm_input

st.divider()

# --- Upload & Process ---
uploaded_file = st.file_uploader("Drop EPUB file here", type=["epub"])

if uploaded_file:
    # 1. NEW FILE DETECTION
    if st.session_state.get("current_file_name") != uploaded_file.name:
        st.session_state.pages = []
        st.session_state.chapter_map = []
        st.session_state.analysis_result = None
        st.session_state.current_file_name = uploaded_file.name
        if "page_slider" in st.session_state:
            del st.session_state.page_slider
        
    # 2. PARSING
    if not st.session_state.pages:
        with st.spinner("Breaking book into pages..."):
            pages, chap_map, title, author = parse_epub_to_pages(uploaded_file)
            st.session_state.pages = pages
            st.session_state.chapter_map = chap_map
            st.session_state.book_meta = {"title": title, "author": author}
            
    if st.session_state.pages:
        total_pages = len(st.session_state.pages)
        
        # Calculate Total Reading Time
        total_words = sum([len(p['content'].split()) for p in st.session_state.pages])
        total_time_str = calculate_reading_time(total_words, wpm_input)
        
        # METRICS
        m1, m2 = st.columns(2)
        m1.metric("Book Title", st.session_state.book_meta['title'])
        m2.metric("Total Virtual Pages", f"{total_pages} ({total_time_str})")
        st.write("---")
        
        # --- SCOPE SELECTORS ---
        chapter_titles = [c['title'] for c in st.session_state.chapter_map]
        
        if chapter_titles:
            c1, c2 = st.columns(2)
            with c1:
                st.selectbox(
                    "Start Chapter:", 
                    options=chapter_titles, 
                    index=0,
                    key="start_chapter_select",
                    on_change=update_slider_from_dropdowns
                )
            with c2:
                last_idx = len(chapter_titles) - 1
                st.selectbox(
                    "End Chapter:", 
                    options=chapter_titles, 
                    index=last_idx,
                    key="end_chapter_select",
                    on_change=update_slider_from_dropdowns
                )
        
        # --- SLIDER ---
        if "page_slider" not in st.session_state:
             st.session_state.page_slider = (1, min(10, total_pages))

        sel_start, sel_end = st.slider(
            "Fine-Tune Page Range",
            min_value=1,
            max_value=total_pages,
            key="page_slider", 
            on_change=clear_results
        )
        
        selected_pages = st.session_state.pages[sel_start-1 : sel_end]
        
        # Calculate Selection Time
        sel_words = sum([len(p['content'].split()) for p in selected_pages])
        sel_time_str = calculate_reading_time(sel_words, wpm_input)
        
        # --- PREVIEW ---
        st.subheader("Selection Preview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Start (Page {sel_start})**")
            if selected_pages:
                start_preview = selected_pages[0]['content'][:400].replace("\n", " ") + "..."
            else:
                start_preview = "(No pages selected)"
            st.markdown(f'<div class="preview-text">{start_preview}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f"**End (Page {sel_end})**")
            if selected_pages:
                end_preview = "..." + selected_pages[-1]['content'][-400:].replace("\n", " ")
            else:
                end_preview = "(No pages selected)"
            st.markdown(f'<div class="preview-text">{end_preview}</div>', unsafe_allow_html=True)
            
        st.caption(f"Analyzing {len(selected_pages)} pages (Approx. {sel_time_str} reading time).")
        st.write("---")

        # --- ACTIONS ---
        col_sum, col_xray = st.columns(2)
        
        with col_sum:
            if st.button("Story Recap", type="primary", use_container_width=True):
                if not api_key_input:
                    st.error("Missing API Key")
                else:
                    st.session_state.analysis_result = None
                    with st.spinner("Analyzing chapter by chapter..."):
                        res = summarize_segmented(selected_pages, api_key_input, st.session_state.book_meta['title'], st.session_state.book_meta['author'])
                        st.session_state.analysis_result = {"type": "Recap", "text": res}

        with col_xray:
            if st.button("X-Ray Analysis", type="secondary", use_container_width=True):
                if not api_key_input:
                    st.error("Missing API Key")
                else:
                    st.session_state.analysis_result = None
                    with st.spinner("Scanning Segmented X-Ray (this guarantees full coverage)..."):
                        # NOW USES SEGMENTED X-RAY
                        res = xray_segmented(selected_pages, api_key_input, st.session_state.book_meta['title'], st.session_state.book_meta['author'])
                        st.session_state.analysis_result = {"type": "X-Ray", "text": res}

        if st.session_state.analysis_result:
            st.write("---")
            st.subheader(f"Result: {st.session_state.analysis_result['type']}")
            st.markdown(f'<div class="summary-box">{st.session_state.analysis_result["text"]}</div>', unsafe_allow_html=True)