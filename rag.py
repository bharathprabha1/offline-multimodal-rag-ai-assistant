import os
import json
import re
import threading
import sys
import tempfile
import subprocess
import shutil
import time
import datetime
import queue
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
from functools import partial
from PIL import Image, ImageTk
from collections import Counter

# --- MODERN GUI LIBRARY ---
try:
    import customtkinter as ctk
except ImportError:
    print("Please run: pip install customtkinter")
    sys.exit(1)

# --- VISUALIZATION & AI LIBRARIES ---
try:
    # IMPORTING YOUR NEW MODULE HERE
    import rag_visualizer

    import numpy as np
    import pyttsx3
    import PyPDF2
    import soundfile as sf
    import sounddevice as sd
    from docx import Document
    from vosk import Model, KaldiRecognizer
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from llama_cpp import Llama
    from deep_translator import GoogleTranslator

    # Matplotlib Integration
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

except ImportError as e:
    print(f"CRITICAL ERROR: Missing library. {e}")
    print(
        "Run: pip install numpy pyttsx3 PyPDF2 soundfile sounddevice python-docx vosk sentence-transformers scikit-learn llama-cpp-python deep-translator matplotlib wordcloud networkx")
    sys.exit(1)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILENAME)
VOSK_PATH = "vosk-model-small-en-us-0.15"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
HISTORY_FILE = "chat_history.json"

# --- GLOBAL OBJECTS ---
embedder = None
llm = None
chunks = []
embeddings = []
last_response_text = ""
current_engine = None
is_speaking = False
is_listening = False
audio_queue = queue.Queue()
is_busy = False

# --- SET THEME ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ==============================================================================
# PART 1: HISTORY MANAGER
# ==============================================================================
class HistoryManager:
    def __init__(self):
        self.filepath = HISTORY_FILE
        self.history_data = self.load_history()
        self.start_new_session()

    def start_new_session(self):
        self.current_session_id = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.current_session_id not in self.history_data:
            self.history_data[self.current_session_id] = {
                "title": f"New Chat {datetime.datetime.now().strftime('%H:%M')}",
                "messages": []
            }

    def load_history(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def save_message(self, role, text):
        if self.current_session_id not in self.history_data:
            self.history_data[self.current_session_id] = {"title": "New Chat", "messages": []}

        session = self.history_data[self.current_session_id]
        if role == "user" and len(session["messages"]) == 0:
            session["title"] = text[:25] + "..." if len(text) > 25 else text

        session["messages"].append({"role": role, "text": text, "timestamp": str(datetime.datetime.now())})

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.history_data, f, indent=4)

    def get_sessions(self):
        sessions = [(k, v["title"]) for k, v in self.history_data.items()]
        return sorted(sessions, key=lambda x: x[0], reverse=True)

    def get_session_messages(self, session_id):
        return self.history_data.get(session_id, {}).get("messages", [])


history_manager = HistoryManager()


# ==============================================================================
# PART 2: INGESTION LOGIC
# ==============================================================================
def clean_text(t):
    return re.sub(r"\s+", " ", t).strip()


def process_audio(p):
    if not os.path.exists(VOSK_PATH): return ""
    model = Model(VOSK_PATH)
    rec = KaldiRecognizer(model, 16000)
    try:
        data, sr = sf.read(p, dtype="int16")
        if len(data.shape) > 1: data = data.mean(axis=1).astype("int16")
        results = []
        for i in range(0, len(data), 4000):
            if rec.AcceptWaveform(data[i:i + 4000].tobytes()):
                results.append(json.loads(rec.Result()).get("text", ""))
        results.append(json.loads(rec.FinalResult()).get("text", ""))
        return clean_text(" ".join(results))
    except:
        return ""


def run_ingestion(target_folder, status_callback):
    global embedder, chunks, embeddings
    local_cache_dir = os.path.join(target_folder, "cache")
    chunks_file = os.path.join(local_cache_dir, "chunks.json")
    embeddings_file = os.path.join(local_cache_dir, "embeddings.npy")

    status_callback("Loading Embedder...")
    if embedder is None:
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    status_callback(f"Scanning folder: {os.path.basename(target_folder)}")
    new_chunks = []
    corpus = []

    try:
        files = os.listdir(target_folder)
    except Exception as e:
        status_callback(f"Error accessing folder: {e}")
        return False

    if not files:
        status_callback("Folder is empty!")
        return False

    os.makedirs(local_cache_dir, exist_ok=True)

    for i, f in enumerate(files):
        p = os.path.join(target_folder, f)
        text = ""
        if not os.path.isfile(p): continue
        status_callback(f"Reading: {f}...")

        try:
            if f.lower().endswith(".pdf"):
                try:
                    r = PyPDF2.PdfReader(open(p, "rb"))
                    text = " ".join(pg.extract_text() or "" for pg in r.pages)
                except:
                    text = ""
            elif f.lower().endswith(".docx"):
                try:
                    d = Document(p)
                    text = " ".join(x.text for x in d.paragraphs)
                except:
                    text = ""
            elif f.lower().endswith(".txt"):
                try:
                    text = open(p, encoding="utf-8", errors="ignore").read()
                except:
                    text = ""
            elif f.lower().endswith(".wav"):
                text = process_audio(p)

            text = clean_text(text)
            if not text: continue

            words = text.split()
            chunk_size = 500
            for j in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[j:j + chunk_size])
                if len(chunk_text) > 30:
                    new_chunks.append({"source": f, "text": chunk_text})
                    corpus.append(chunk_text)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if new_chunks:
        status_callback("Updating AI Memory...")
        vecs = embedder.encode(corpus)
        chunks = new_chunks
        embeddings = vecs

        with open(chunks_file, "w", encoding='utf-8') as f:
            json.dump(new_chunks, f)
        np.save(embeddings_file, vecs)

        status_callback(f"Success! Learned {len(new_chunks)} segments.")
        return True
    else:
        status_callback("No valid text found in files.")
        return False


def load_llm(callback):
    global llm
    if not os.path.exists(MODEL_PATH):
        callback(f"Error: Model missing at {MODEL_PATH}")
        return False
    if llm is None:
        try:
            callback("Loading Brain (Mistral-7B)...")
            llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=6, n_gpu_layers=-1, verbose=False)
            callback("Brain Loaded (Fast Mode).")
            return True
        except Exception as e:
            callback(f"LLM Crash: {e}")
            return False
    return True


# ==============================================================================
# PART 3: GENERATION & LOGIC
# ==============================================================================
def speak_worker(text, on_finish):
    global current_engine, is_speaking
    try:
        import pythoncom
        pythoncom.CoInitialize()
        current_engine = pyttsx3.init()
        current_engine.setProperty('rate', 150)
        current_engine.say(text)
        current_engine.runAndWait()
    except Exception as e:
        print(f"Speech Error: {e}")
    finally:
        is_speaking = False
        if on_finish: on_finish()
        try:
            pythoncom.CoUninitialize()
        except:
            pass


def toggle_speech(button_widget):
    global is_speaking, current_engine, last_response_text, is_busy
    if is_busy: return

    if is_speaking:
        try:
            if current_engine: current_engine.stop()
        except:
            pass
        is_speaking = False
        button_widget.configure(text="🔊 Speak Answer", fg_color="#4b5320", hover_color="#5f6b2e")
    else:
        if not last_response_text or len(last_response_text.strip()) == 0:
            return
        is_speaking = True
        button_widget.configure(text="⏹ Stop Speaking", fg_color="#8B0000", hover_color="#A52A2A")

        def reset_ui():
            button_widget.configure(text="🔊 Speak Answer", fg_color="#4b5320", hover_color="#5f6b2e")

        threading.Thread(target=speak_worker, args=(last_response_text, reset_ui), daemon=True).start()


# ==============================================================================
# PART 4: MODERN GUI CLASS
# ==============================================================================
def create_gradient_image(width, height, color1, color2):
    base = Image.new('RGB', (width, height), color1)
    top = Image.new('RGB', (width, height), color2)
    mask = Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return base


class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Offline RAG: Voice & Visual AI")
        self.geometry("1200x700")
        self.minsize(1000, 600)

        # Background
        self.bg_image_data = create_gradient_image(1300, 800, "#0f0c29", "#302b63")
        self.bg_image = ctk.CTkImage(light_image=self.bg_image_data, dark_image=self.bg_image_data, size=(1300, 800))
        self.bg_label = ctk.CTkLabel(self, image=self.bg_image, text="")
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.grid_columnconfigure(1, weight=3)  # Chat
        self.grid_columnconfigure(2, weight=2)  # Visuals
        self.grid_rowconfigure(0, weight=1)

        # --- SIDEBAR ---
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=15, fg_color="#1a1a2e", border_width=1,
                                    border_color="#333")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar.grid_rowconfigure(8, weight=1)

        self.logo = ctk.CTkLabel(self.sidebar, text="RAG BRAIN",
                                 font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"), text_color="#00d4ff")
        self.logo.grid(row=0, column=0, padx=20, pady=(20, 15))

        self.btn_select = ctk.CTkButton(self.sidebar, text="📂 Select Folder", command=self.select_folder, height=35,
                                        fg_color="#16213e", hover_color="#0f3460")
        self.btn_select.grid(row=1, column=0, padx=15, pady=8, sticky="ew")

        self.btn_process = ctk.CTkButton(self.sidebar, text="⚙️ Process Docs", command=self.start_processing, height=35,
                                         fg_color="#E59937", hover_color="#D68826")
        self.btn_process.grid(row=2, column=0, padx=15, pady=8, sticky="ew")

        self.lbl_lang = ctk.CTkLabel(self.sidebar, text="Translation Output:", anchor="w", text_color="#a0a0a0")
        self.lbl_lang.grid(row=3, column=0, padx=15, pady=(20, 5), sticky="w")

        self.lang_var = ctk.StringVar(value="English")
        self.lang_menu = ctk.CTkOptionMenu(self.sidebar, variable=self.lang_var,
                                           values=["English", "Hindi", "Telugu", "Kannada"], fg_color="#16213e",
                                           button_color="#0f3460")
        self.lang_menu.grid(row=4, column=0, padx=15, pady=5, sticky="ew")

        self.btn_speak = ctk.CTkButton(self.sidebar, text="🔊 Speak Answer",
                                       command=lambda: toggle_speech(self.btn_speak), height=35, fg_color="#4b5320",
                                       hover_color="#5f6b2e")
        self.btn_speak.grid(row=5, column=0, padx=15, pady=15, sticky="ew")

        self.btn_new_chat = ctk.CTkButton(self.sidebar, text="➕ New Chat", command=self.start_new_chat, height=35,
                                          fg_color="#2E8B57", hover_color="#3CB371")
        self.btn_new_chat.grid(row=6, column=0, padx=15, pady=(10, 5), sticky="ew")

        self.history_frame = ctk.CTkScrollableFrame(self.sidebar, fg_color="transparent")
        self.history_frame.grid(row=8, column=0, padx=5, pady=5, sticky="nsew")

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Ready", text_color="#00e676", font=("Consolas", 10))
        self.lbl_status.grid(row=9, column=0, padx=15, pady=15, sticky="s")

        # --- MAIN CHAT AREA ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame, font=("Consolas", 11), wrap=tk.WORD,
            bg="#16213e", fg="white", bd=0, highlightthickness=0, padx=15, pady=15
        )
        self.chat_display.pack(expand=True, fill="both")
        self.chat_display.tag_config("user", foreground="#4da6ff", font=("Consolas", 11, "bold"))
        self.chat_display.tag_config("ai", foreground="#00e676", font=("Consolas", 11, "bold"))
        self.chat_display.configure(state="disabled")

        self.input_frame = ctk.CTkFrame(self.main_frame, height=50, corner_radius=20, fg_color="#1a1a2e",
                                        border_width=1, border_color="#444")
        self.input_frame.pack(fill="x", pady=(10, 0))

        self.entry_query = ctk.CTkEntry(self.input_frame, placeholder_text="Ask a question...", height=40,
                                        font=("Segoe UI", 12), border_width=0, fg_color="transparent")
        self.entry_query.pack(side="left", fill="x", expand=True, padx=10, pady=5)
        self.entry_query.bind("<KeyRelease>", self.toggle_send_mic_button)
        self.entry_query.bind("<Return>", lambda event: self.start_generation())

        self.btn_action = ctk.CTkButton(self.input_frame, text="🎤", width=40, height=35, command=self.handle_mic_click,
                                        corner_radius=15, fg_color="#333", hover_color="#444")
        self.btn_action.pack(side="right", padx=(5, 10))

        # --- VISUALIZATION DASHBOARD ---
        self.vis_frame = ctk.CTkFrame(self, width=300, corner_radius=15, fg_color="#1a1a2e", border_width=1,
                                      border_color="#333")
        self.vis_frame.grid(row=0, column=2, sticky="nsew", padx=(5, 10), pady=10)
        self.vis_frame.grid_rowconfigure(1, weight=1)
        self.vis_frame.grid_columnconfigure(0, weight=1)

        self.lbl_vis_title = ctk.CTkLabel(self.vis_frame, text="📊 Live Data Visuals", font=("Segoe UI", 14, "bold"),
                                          text_color="#E59937")
        self.lbl_vis_title.grid(row=0, column=0, pady=(15, 10))

        self.vis_tabs = ctk.CTkTabview(self.vis_frame, width=280, height=500, fg_color="transparent")
        self.vis_tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.tab_cloud = self.vis_tabs.add("☁️ Topics")
        self.tab_bar = self.vis_tabs.add("📊 Keywords")
        self.tab_graph = self.vis_tabs.add("🕸️ Graph")

        if not os.path.exists(MODEL_PATH):
            self.log("Warning: Mistral Model not found.", "System")

        self.refresh_history_ui()

    # --- VISUALIZATION INTEGRATION ---
    def update_visuals(self, query, text_data):
        # CALLING THE EXTERNAL MODULE HERE
        fig_cloud = rag_visualizer.create_wordcloud_fig(text_data)
        self.display_figure_in_tab(self.tab_cloud, fig_cloud)

        fig_bar = rag_visualizer.create_barchart_fig(text_data)
        self.display_figure_in_tab(self.tab_bar, fig_bar)

        fig_graph = rag_visualizer.create_knowledge_graph_fig(query, text_data)
        self.display_figure_in_tab(self.tab_graph, fig_graph)

    def display_figure_in_tab(self, tab, fig):
        for widget in tab.winfo_children():
            widget.destroy()

        if fig is None:
            lbl = ctk.CTkLabel(tab, text="No Data Available")
            lbl.pack(pady=50)
            return

        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- UI LOGIC ---
    def start_new_chat(self):
        history_manager.start_new_session()
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        self.chat_display.configure(state="disabled")
        self.refresh_history_ui()

    def toggle_send_mic_button(self, event=None):
        global is_busy
        if is_busy: return
        text = self.entry_query.get().strip()
        if len(text) > 0:
            self.btn_action.configure(text="➤", fg_color="#00d4ff", hover_color="#00a3cc",
                                      command=self.start_generation)
        else:
            self.btn_action.configure(text="🎤", fg_color="#333", hover_color="#444", command=self.handle_mic_click)

    def handle_mic_click(self):
        global is_listening, is_busy
        if is_busy: return

        if is_listening:
            is_listening = False
            self.btn_action.configure(text="🎤", fg_color="#333")
            self.lbl_status.configure(text="Voice Stopped")
        else:
            is_listening = True
            self.btn_action.configure(text="🛑", fg_color="#ff4444", hover_color="#cc0000")
            self.lbl_status.configure(text="Listening... Speak Now")
            threading.Thread(target=self.voice_listener, daemon=True).start()

    def voice_listener(self):
        global is_listening
        if not os.path.exists(VOSK_PATH):
            self.after(0, lambda: messagebox.showerror("Error", "Vosk Model not found."))
            return

        model = Model(VOSK_PATH)
        rec = KaldiRecognizer(model, 16000)

        def callback(indata, frames, time, status):
            audio_queue.put(bytes(indata))

        try:
            with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1, callback=callback):
                while is_listening:
                    if not audio_queue.empty():
                        data = audio_queue.get()
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            text = result.get("text", "")
                            if text:
                                self.after(0, lambda t=text: self.entry_query.insert("end", t + " "))
                                self.after(0, self.toggle_send_mic_button)
        except:
            pass
        finally:
            is_listening = False
            self.after(0, self.finalize_voice_input)

    def finalize_voice_input(self):
        text = self.entry_query.get().strip()
        if text:
            self.start_generation()
        else:
            self.btn_action.configure(text="🎤", fg_color="#333", command=self.handle_mic_click)
            self.lbl_status.configure(text="Ready")

    def refresh_history_ui(self):
        for widget in self.history_frame.winfo_children():
            widget.destroy()

        sessions = history_manager.get_sessions()
        for session_id, title in sessions:
            btn = ctk.CTkButton(
                self.history_frame, text=title, fg_color="transparent", border_width=1, border_color="#333",
                anchor="w", height=30, command=partial(self.load_session_to_chat, session_id)
            )
            btn.pack(fill="x", pady=2)

    def load_session_to_chat(self, session_id):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", "end")
        messages = history_manager.get_session_messages(session_id)

        for msg in messages:
            role_tag = "user" if msg['role'] == "user" else "ai"
            sender_name = "👤 You: " if msg['role'] == "user" else "🤖 AI: "
            self.chat_display.insert("end", f"\n{sender_name}", role_tag)
            self.chat_display.insert("end", f"{msg['text']}\n")

        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def log(self, message, sender="System"):
        if sender == "System":
            display_text = message[:40] + "..." if len(message) > 40 else message
            self.lbl_status.configure(text=display_text)
            return

        self.chat_display.configure(state="normal")
        if sender == "You":
            self.chat_display.insert("end", f"\n\n👤 You: {message}\n", "user")
            history_manager.save_message("user", message)
            self.after(0, self.refresh_history_ui)
        elif sender == "AI":
            self.chat_display.insert("end", f"🤖 AI: {message}\n", "ai")
            history_manager.save_message("ai", message)

        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def stream_token(self, token):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", token, "ai")
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder = folder
            self.log(f"Selected: {os.path.basename(folder)}", "System")

    def start_processing(self):
        if not hasattr(self, 'selected_folder') or not self.selected_folder:
            messagebox.showwarning("Error", "Please select a folder first.")
            return

        self.btn_process.configure(state="disabled", text="Processing...")

        def worker():
            success_ingest = run_ingestion(self.selected_folder, lambda msg: self.after(0, self.log, msg, "System"))
            if success_ingest:
                self.after(0, self.log, "Ingestion complete. Connecting to LLM...", "System")
                load_llm(lambda msg: self.after(0, self.log, msg, "System"))
            else:
                self.after(0, self.log, "Ingestion failed.", "System")
            self.after(0, lambda: self.btn_process.configure(state="normal", text="⚙️ Process Docs"))

        threading.Thread(target=worker, daemon=True).start()

    def start_generation(self):
        global last_response_text, is_busy
        query = self.entry_query.get().strip()
        if not query: return

        is_busy = True
        self.entry_query.delete(0, "end")
        self.btn_action.configure(text="⏳", state="disabled", fg_color="#555")
        self.btn_speak.configure(state="disabled")

        self.log(query, "You")

        if llm is None:
            load_llm(lambda msg: self.after(0, self.log, msg, "System"))

        greetings = ["hi", "hello", "hlo", "hey", "hii", "hello ai"]
        if query.lower().strip() in greetings:
            self.log("Hello! I am your Offline AI.", "AI")
            self.finish_generation()
            return

        selected_lang = self.lang_var.get()

        trigger_visuals = any(
            x in query.lower() for x in ["visualize", "graph", "chart", "plot", "show data", "diagram", "map"])
        wants_ascii_flowchart = any(x in query.lower() for x in ["flowchart", "ascii"])

        def worker():
            global last_response_text
            try:
                self.after(0, self.lbl_status.configure, {"text": "Searching..."})
                context_text = "No local documents found."

                if chunks and embedder:
                    q_vec = embedder.encode([query])
                    scores = cosine_similarity(q_vec, embeddings)[0]
                    best_idx = int(np.argmax(scores))
                    if scores[best_idx] > 0.3:
                        doc = chunks[best_idx]
                        context_text = doc['text'][:800]
                        if trigger_visuals:
                            self.after(0, self.update_visuals, query, doc['text'])

                current_time_str = datetime.datetime.now().strftime("%A, %B %d, %Y %I:%M %p")

                if wants_ascii_flowchart:
                    prompt = f"""[INST] Draw a vertical ASCII flowchart for: "{query}".
                    Use [ ] for boxes and v for arrows. No markdown.
                    Context: {context_text} [/INST]"""
                else:
                    prompt = f"""[INST] Current Time: {current_time_str}.
                    Context: {context_text}
                    Question: {query}
                    Answer naturally. [/INST]"""

                self.after(0, self.lbl_status.configure, {"text": "Generating..."})

                if llm:
                    if selected_lang == "English":
                        self.after(0, lambda: self.chat_display.configure(state="normal"))
                        self.after(0, lambda: self.chat_display.insert("end", "🤖 AI: ", "ai"))

                        stream = llm(prompt, max_tokens=512, stop=["</s>"], stream=True)
                        full_answer = ""
                        for chunk in stream:
                            text_chunk = chunk['choices'][0]['text']
                            full_answer += text_chunk
                            self.after(0, self.stream_token, text_chunk)

                        self.after(0, lambda: self.chat_display.insert("end", "\n"))
                        answer = full_answer
                        history_manager.save_message("ai", answer)
                    else:
                        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=False)
                        answer = output['choices'][0]['text'].strip()
                        try:
                            lang_code = {"Hindi": "hi", "Telugu": "te", "Kannada": "kn"}.get(selected_lang, "en")
                            answer = GoogleTranslator(source='auto', target=lang_code).translate(answer)
                        except:
                            pass
                        self.after(0, self.log, answer, "AI")
                else:
                    answer = "LLM not connected."
                    self.after(0, self.log, answer, "AI")

                last_response_text = answer

            except Exception as e:
                self.after(0, self.lbl_status.configure, {"text": f"Error: {str(e)[:40]}..."})
            finally:
                self.after(0, self.finish_generation)

        threading.Thread(target=worker, daemon=True).start()

    def finish_generation(self):
        global is_busy
        is_busy = False
        self.btn_action.configure(state="normal")
        self.btn_speak.configure(state="normal")
        self.lbl_status.configure(text="Ready")
        self.toggle_send_mic_button()


if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()