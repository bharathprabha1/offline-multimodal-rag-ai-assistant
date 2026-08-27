# OFFLINE MULTIMODAL RAG AI ASSISTANT
# FINAL rag.py
# Modular core bridge with safe RAG grounding and 1536-token protection.

import os
import sys
import re
import json
import queue
import threading
from pathlib import Path
from tkinter import messagebox, filedialog

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LLAMA_CPP_LOG_LEVEL", "error")

try:
    import customtkinter as ctk
except ImportError:
    print("ERROR: customtkinter is not installed.")
    print("Run: python -m pip install customtkinter")
    raise

import numpy as np
import faiss

from core import config, ingestion, retrieval, answer_engine, document_processor

try:
    import rag_visualizer
    VISUALIZER_AVAILABLE = True
except Exception:
    rag_visualizer = None
    VISUALIZER_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
VOSK_PATH = os.path.join(BASE_DIR, "vosk-model-small-en-us-0.15")
HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

CONTEXT_SIZE = int(getattr(config, "CONTEXT_SIZE", 1536))
GPU_LAYERS = int(getattr(config, "GPU_LAYERS", 16))
CPU_THREADS = int(getattr(config, "CPU_THREADS", 8))
# Mistral is configured for 1536 context on this machine.
# Keep generation conservative so prompt + output never overflows.
MAX_OUTPUT_TOKENS = min(int(getattr(config, "MAX_OUTPUT_TOKENS", 96)), 96)
TOP_K = int(getattr(config, "TOP_K", 4))
SIMILARITY_THRESHOLD = float(getattr(config, "SIMILARITY_THRESHOLD", 0.25))

RAG_MAX_CONTEXT_CHUNKS = min(int(getattr(config, "RAG_MAX_CONTEXT_CHUNKS", 3)), 3)
RAG_MAX_CHUNK_CHARS = min(int(getattr(config, "RAG_MAX_CHUNK_CHARS", 1800)), 1600)
RAG_MAX_CONTEXT_CHARS = min(int(getattr(config, "RAG_MAX_CONTEXT_CHARS", 4200)), 3400)

NO_CONTEXT_ANSWER = "I could not find this information in the provided documents."

embedder = None
llm = None
chunks = []
embeddings = None
faiss_index = None
last_response_text = ""
last_retrieved_sources = []
last_rag_context = ""
current_engine = None
is_speaking = False
is_listening = False
audio_queue = queue.Queue()
is_busy = False
is_processing = False
is_generating = False
generation_thread = None
processing_thread = None
generation_stop_event = threading.Event()
processing_stop_event = threading.Event()
state_lock = threading.RLock()

CORE_STATE = {
    "chunks": [],
    "embeddings": None,
    "faiss_index": None,
    "embedder": None,
}

def sync_core_rag_state():
    global chunks, embeddings, faiss_index, embedder
    with state_lock:
        chunks = getattr(ingestion, "chunks", []) or []
        embeddings = getattr(ingestion, "embeddings", None)
        faiss_index = getattr(ingestion, "faiss_index", None)
        embedder = getattr(ingestion, "embedder", None)
        CORE_STATE.update({
            "chunks": chunks,
            "embeddings": embeddings,
            "faiss_index": faiss_index,
            "embedder": embedder,
        })
    return CORE_STATE

def push_core_rag_state():
    with state_lock:
        ingestion.chunks = chunks
        ingestion.embeddings = embeddings
        ingestion.faiss_index = faiss_index
        ingestion.embedder = embedder
        CORE_STATE.update({
            "chunks": chunks,
            "embeddings": embeddings,
            "faiss_index": faiss_index,
            "embedder": embedder,
        })

def clean_text(text):
    if text is None:
        return ""
    text = str(text).replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _fn(name):
    return getattr(document_processor, name, None)

def extract_pdf(path):
    fn = _fn("extract_pdf")
    return fn(path) if fn else ""

def extract_docx(path):
    fn = _fn("extract_docx")
    return fn(path) if fn else ""

def extract_txt(path):
    fn = _fn("extract_txt")
    return fn(path) if fn else ""

def extract_csv(path):
    fn = _fn("extract_csv")
    return fn(path) if fn else ""

def extract_xlsx(path):
    fn = _fn("extract_xlsx")
    return fn(path) if fn else ""

def extract_pptx(path):
    fn = _fn("extract_pptx")
    return fn(path) if fn else ""

def extract_html(path):
    fn = _fn("extract_html")
    return fn(path) if fn else ""

def extract_json(path):
    fn = _fn("extract_json")
    return fn(path) if fn else ""

def extract_image(path):
    fn = _fn("extract_image")
    return fn(path) if fn else ""

def process_audio(path):
    fn = _fn("process_audio")
    return fn(path) if fn else ""

def create_chunks(text, source):
    fn = _fn("create_chunks")
    if not fn:
        text = clean_text(text)
        if not text:
            return []
        out, start, size, overlap = [], 0, 500, 80
        while start < len(text):
            end = min(start + size, len(text))
            part = text[start:end].strip()
            if part:
                out.append({"source": source, "text": part})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return out
    try:
        result = fn(text, source)
    except TypeError:
        try:
            result = fn(text=text, source=source, chunk_size=500, overlap=80)
        except TypeError:
            result = fn(text=text, source_name=source, chunk_size=500, overlap=80)
    out = []
    for item in result or []:
        if isinstance(item, dict):
            item = dict(item)
            item["source"] = item.get("source") or source
            item["text"] = clean_text(item.get("text", ""))
            if item["text"]:
                out.append(item)
        else:
            value = clean_text(item)
            if value:
                out.append({"source": source, "text": value})
    return out

def load_embedder(status_callback=None):
    global embedder
    if callable(getattr(embedder, "encode", None)):
        ingestion.embedder = embedder
        return embedder
    try:
        if status_callback:
            status_callback("Loading local embedding model...")
        result = ingestion.load_embedder(status_callback=status_callback)
        if callable(getattr(result, "encode", None)):
            embedder = result
        if not callable(getattr(embedder, "encode", None)):
            embedder = getattr(ingestion, "embedder", None)
        if not callable(getattr(embedder, "encode", None)):
            raise RuntimeError("Embedding model did not load correctly.")
        ingestion.embedder = embedder
        CORE_STATE["embedder"] = embedder
        if status_callback:
            status_callback("Embedding model ready.")
        return embedder
    except Exception as exc:
        if status_callback:
            status_callback(f"Embedding model error: {exc}")
        print("Embedding model error:", exc)
        return None

def build_faiss_index(vectors):
    global faiss_index
    if vectors is None:
        return None
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        return None
    matrix = np.ascontiguousarray(matrix)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    faiss_index = index
    return index

def load_cached_memory(target_folder, status_callback=None):
    global chunks, embeddings, faiss_index, embedder
    try:
        if not callable(getattr(embedder, "encode", None)):
            embedder = load_embedder(status_callback)
        CORE_STATE["embedder"] = embedder
        # IMPORTANT: always pass the state dictionary.
        ok = ingestion.load_cached_memory(
            target_folder=target_folder,
            status_callback=status_callback,
            state=CORE_STATE
        )
        if ok:
            chunks = CORE_STATE.get("chunks", []) or []
            embeddings = CORE_STATE.get("embeddings")
            faiss_index = CORE_STATE.get("faiss_index")
            embedder = CORE_STATE.get("embedder", embedder)
            push_core_rag_state()
            return True
        return False
    except Exception as exc:
        print("Cache load error:", exc)
        if status_callback:
            status_callback(f"Cache load error: {exc}")
        return False

SUPPORTED_EXTENSIONS = {
    ".pdf",".docx",".txt",".md",".csv",".xlsx",".pptx",".html",".htm",".json",
    ".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp",".wav"
}

def _extract_file(path, ext, status_callback=None):
    if ext == ".pdf": return extract_pdf(path)
    if ext == ".docx": return extract_docx(path)
    if ext in (".txt",".md"): return extract_txt(path)
    if ext == ".csv": return extract_csv(path)
    if ext == ".xlsx": return extract_xlsx(path)
    if ext == ".pptx": return extract_pptx(path)
    if ext in (".html",".htm"): return extract_html(path)
    if ext == ".json": return extract_json(path)
    if ext in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}:
        if status_callback:
            status_callback(f"OCR processing: {os.path.basename(path)}")
        return extract_image(path)
    if ext == ".wav": return process_audio(path)
    return ""

def run_ingestion(target_folder, status_callback=None, stop_event=None):
    global is_processing, chunks, embeddings, faiss_index, embedder
    if status_callback is None:
        status_callback = lambda _msg: None
    if stop_event is None:
        stop_event = processing_stop_event
    if not target_folder or not os.path.isdir(target_folder):
        status_callback("Invalid document folder.")
        return False

    processing_stop_event.clear()
    is_processing = True
    try:
        if not callable(getattr(embedder, "encode", None)):
            embedder = load_embedder(status_callback)
        if not callable(getattr(embedder, "encode", None)):
            return False

        files = sorted(
            [f for f in os.listdir(target_folder)
             if os.path.isfile(os.path.join(target_folder, f))],
            key=str.lower
        )
        if not files:
            status_callback("Folder is empty.")
            return False

        all_chunks = []
        for n, filename in enumerate(files, 1):
            if stop_event.is_set():
                status_callback("Document processing stopped.")
                return False
            path = os.path.join(target_folder, filename)
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            status_callback(f"[{n}/{len(files)}] Reading: {filename}")
            try:
                text = clean_text(_extract_file(path, ext, status_callback))
                if not text:
                    continue
                all_chunks.extend(create_chunks(text, filename))
            except Exception as exc:
                status_callback(f"Error reading {filename}: {str(exc)[:140]}")

        if not all_chunks:
            status_callback("No supported document content found.")
            return False

        status_callback(f"Creating embeddings for {len(all_chunks)} segments...")
        texts = [clean_text(x.get("text","")) for x in all_chunks]
        texts = [x for x in texts if x]
        if not texts:
            return False

        if stop_event.is_set():
            status_callback("Document processing stopped before embedding.")
            return False

        vectors = np.asarray(
            embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32
        )
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            status_callback("Embedding generation returned no vectors.")
            return False

        index = build_faiss_index(vectors)
        if index is None:
            status_callback("Failed to build FAISS index.")
            return False

        with state_lock:
            chunks = all_chunks
            embeddings = vectors
            faiss_index = index
            ingestion.chunks = chunks
            ingestion.embeddings = embeddings
            ingestion.faiss_index = faiss_index
            ingestion.embedder = embedder
            CORE_STATE.update({
                "chunks": chunks, "embeddings": embeddings,
                "faiss_index": faiss_index, "embedder": embedder
            })

        cache_dir = os.path.join(target_folder, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(os.path.join(cache_dir,"chunks.json"),"w",encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            np.save(os.path.join(cache_dir,"embeddings.npy"), vectors)
            faiss.write_index(index, os.path.join(cache_dir,"faiss.index"))
            status_callback("Local cache saved.")
        except Exception as exc:
            status_callback(f"Cache save warning: {str(exc)[:120]}")

        status_callback(
            f"Ingestion complete: {len(chunks)} chunks, "
            f"{vectors.shape[0]} embeddings, {index.ntotal} FAISS vectors."
        )
        return True
    except Exception as exc:
        print("Processing error:", exc)
        status_callback(f"Processing error: {exc}")
        return False
    finally:
        is_processing = False

def retrieve_context(query, top_k=TOP_K, similarity_threshold=SIMILARITY_THRESHOLD):
    sync_core_rag_state()
    if not query or not chunks or faiss_index is None or embedder is None:
        return [], ""
    try:
        return retrieval.retrieve_context(
            query, top_k=top_k, similarity_threshold=similarity_threshold
        )
    except TypeError:
        return retrieval.retrieve_context(query, top_k=top_k)
    except Exception as exc:
        print("Retrieval error:", exc)
        return [], ""

def get_retrieved_sources(selected):
    try:
        return retrieval.get_retrieved_sources(selected)
    except Exception:
        result = []
        for i, _ in selected or []:
            try:
                s = normalize_source_name(chunks[i].get("source",""))
                if s and s not in result:
                    result.append(s)
            except Exception:
                pass
        return result

def normalize_source_name(value):
    try:
        return retrieval.normalize_source_name(value)
    except Exception:
        value = str(value or "").replace("\\", "/").strip()
        return value.split("/")[-1]

def compact_context(selected):
    parts, used = [], 0
    for index, score in selected[:RAG_MAX_CONTEXT_CHUNKS]:
        if index < 0 or index >= len(chunks):
            continue
        item = chunks[index]
        source = normalize_source_name(item.get("source",""))
        text = clean_text(item.get("text",""))[:RAG_MAX_CHUNK_CHARS]
        block = f"[Source: {source} | Relevance: {float(score):.3f}]\n{text}"
        remaining = RAG_MAX_CONTEXT_CHARS - used
        if remaining <= 100:
            break
        block = block[:remaining]
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)

def generate_answer(query, context=""):
    return answer_engine.generate_answer(query, context)

def is_greeting(q):
    return clean_text(q).lower() in {"hi","hello","hey","hii","hlo","hello ai","hi ai"}

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Offline Multimodal RAG AI Assistant")
        self.geometry("1200x760")
        self.minsize(950,650)
        self.selected_folder = ""
        self.build_ui()
        self.append_chat("SYSTEM",
            "Offline Multimodal RAG AI Assistant ready. "
            "Select and process a local document folder.")
        self.update_stats()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0,column=0,columnspan=2,sticky="ew")
        ctk.CTkLabel(header,text="OFFLINE MULTIMODAL RAG",
                     font=ctk.CTkFont(size=22,weight="bold")).pack(side="left",padx=20,pady=14)
        self.status_label = ctk.CTkLabel(header,text="Ready")
        self.status_label.pack(side="right",padx=20)

        side = ctk.CTkFrame(self,width=270)
        side.grid(row=1,column=0,padx=(12,6),pady=12,sticky="nsew")
        side.grid_propagate(False)

        ctk.CTkLabel(side,text="LOCAL KNOWLEDGE",
                     font=ctk.CTkFont(size=15,weight="bold")).pack(padx=16,pady=(18,8),anchor="w")
        self.folder_label = ctk.CTkLabel(side,text="No folder selected",
                                          wraplength=235,justify="left")
        self.folder_label.pack(padx=16,pady=8,anchor="w")

        self.select_button = ctk.CTkButton(side,text="Select Document Folder",
                                           command=self.select_folder)
        self.select_button.pack(padx=16,pady=6,fill="x")
        self.process_button = ctk.CTkButton(side,text="Process Documents",
                                            command=self.start_processing)
        self.process_button.pack(padx=16,pady=6,fill="x")
        self.stop_process_button = ctk.CTkButton(side,text="Stop Processing",
                                                 command=self.stop_processing,state="disabled")
        self.stop_process_button.pack(padx=16,pady=6,fill="x")
        self.speak_button = ctk.CTkButton(side,text="Speak",
                                          command=lambda: toggle_speech(self.speak_button))
        self.speak_button.pack(padx=16,pady=(18,6),fill="x")
        ctk.CTkButton(side,text="New Chat",command=self.new_chat).pack(
            padx=16,pady=6,fill="x")
        self.stats_label = ctk.CTkLabel(side,text="",justify="left")
        self.stats_label.pack(padx=16,pady=20,anchor="w")

        main = ctk.CTkFrame(self)
        main.grid(row=1,column=1,padx=(6,12),pady=12,sticky="nsew")
        main.grid_columnconfigure(0,weight=1)
        main.grid_rowconfigure(0,weight=1)

        self.chat_box = ctk.CTkTextbox(main,wrap="word",
                                       font=ctk.CTkFont(size=14))
        self.chat_box.grid(row=0,column=0,padx=12,pady=12,sticky="nsew")
        self.chat_box.configure(state="disabled")

        lang = ctk.CTkFrame(main,fg_color="transparent")
        lang.grid(row=1,column=0,padx=12,pady=(0,6),sticky="ew")
        ctk.CTkLabel(lang,text="Answer language:").pack(side="left",padx=(0,8))
        self.language_var = ctk.StringVar(value="English")
        ctk.CTkOptionMenu(lang,variable=self.language_var,
                          values=["English","Telugu","Hindi","Tamil","Kannada","Malayalam"],
                          width=140).pack(side="left")

        inp = ctk.CTkFrame(main)
        inp.grid(row=2,column=0,padx=12,pady=(0,12),sticky="ew")
        inp.grid_columnconfigure(0,weight=1)
        self.query_entry = ctk.CTkEntry(inp,height=44,
            placeholder_text="Ask about your indexed documents...")
        self.query_entry.grid(row=0,column=0,padx=8,pady=8,sticky="ew")
        self.query_entry.bind("<Return>",self.handle_enter)
        self.action_button = ctk.CTkButton(inp,text="Send",width=100,height=44,
                                           command=self.handle_action)
        self.action_button.grid(row=0,column=1,padx=(0,8),pady=8)

    def set_status(self,text):
        try: self.after(0,lambda:self.status_label.configure(text=str(text)))
        except Exception: pass

    def update_stats(self):
        try:
            self.stats_label.configure(text=
                f"Chunks: {len(chunks)}\n"
                f"FAISS: {faiss_index.ntotal if faiss_index else 0}\n"
                f"Embedder: {'ready' if callable(getattr(embedder,'encode',None)) else 'not loaded'}")
        except Exception: pass

    def append_chat(self,role,text):
        self.chat_box.configure(state="normal")
        label = "You" if role=="USER" else "Assistant" if role=="ASSISTANT" else role
        self.chat_box.insert("end",f"\n{label}:\n{text}\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Local Document Folder")
        if not folder: return
        self.selected_folder = folder
        self.folder_label.configure(text=folder)
        self.append_chat("SYSTEM",f"Selected folder: {folder}")
        self.set_status("Checking local cache...")

        def worker():
            ok = load_cached_memory(folder,
                lambda m:self.after(0,self.append_chat,"SYSTEM",m))
            self.after(0,self.update_stats)
            self.after(0,self.set_status,
                       "Cached knowledge loaded." if ok else "Folder selected. Process documents.")

        threading.Thread(target=worker,daemon=True).start()

    def start_processing(self):
        global processing_thread
        if not self.selected_folder:
            messagebox.showwarning("No folder","Please select a document folder first.")
            return
        if is_processing: return
        processing_stop_event.clear()
        self.process_button.configure(state="disabled",text="Processing...")
        self.stop_process_button.configure(state="normal")
        self.select_button.configure(state="disabled")
        self.set_status("Processing documents...")

        def worker():
            ok = run_ingestion(self.selected_folder,
                lambda m:self.after(0,self.append_chat,"SYSTEM",m),
                processing_stop_event)
            self.after(0,self.update_stats)
            self.after(0,self.append_chat,"SYSTEM",
                       "Knowledge base ready." if ok else "Document processing did not complete.")
            self.after(0,self.finish_processing)

        processing_thread = threading.Thread(target=worker,daemon=True)
        processing_thread.start()

    def finish_processing(self):
        self.process_button.configure(state="normal",text="Process Documents")
        self.stop_process_button.configure(state="disabled")
        self.select_button.configure(state="normal")
        self.update_stats()
        if not is_generating: self.set_status("Ready")

    def stop_processing(self):
        processing_stop_event.set()
        self.set_status("Stopping document processing...")

    def handle_enter(self,event=None):
        self.handle_action()
        return "break"

    def handle_action(self):
        if is_generating: self.stop_generation()
        else: self.start_generation()

    def start_generation(self):
        global is_generating, generation_thread
        global last_response_text, last_retrieved_sources, last_rag_context

        query = self.query_entry.get().strip()
        if not query: return
        if is_processing:
            messagebox.showwarning("Processing","Please wait until document processing finishes.")
            return

        sync_core_rag_state()
        if not chunks or faiss_index is None:
            messagebox.showwarning("No Documents",
                "Please select and process a document folder first.")
            return

        is_generating = True
        generation_stop_event.clear()
        self.query_entry.delete(0,"end")
        self.append_chat("USER",query)
        self.action_button.configure(text="Stop")
        self.query_entry.configure(state="disabled")

        language = self.language_var.get()

        def worker():
            global is_generating, last_response_text
            global last_retrieved_sources, last_rag_context
            try:
                if is_greeting(query):
                    answer = "Hello! I am your offline local AI assistant."
                    last_response_text = answer
                    self.after(0,self.append_chat,"ASSISTANT",answer)
                    return

                self.set_status("Retrieving relevant local documents...")
                selected, _raw = retrieve_context(query)

                if generation_stop_event.is_set(): return

                last_retrieved_sources = get_retrieved_sources(selected)
                context = compact_context(selected)
                last_rag_context = context

                # HARD GUARD: no retrieved evidence means no model call.
                # This prevents Mistral pretrained knowledge from answering
                # a missing document fact.
                if not selected or not context.strip():
                    answer = NO_CONTEXT_ANSWER
                    last_response_text = answer
                    self.after(0,self.append_chat,"ASSISTANT",answer)
                    return

                prompt = (
                    "Answer the question using ONLY the local document context.\n"
                    "Do not use internet information or invent facts.\n"
                    "Do not invent filenames.\n"
                    f"Answer in {language}.\n"
                    f"If the context does not support the answer, say exactly: {NO_CONTEXT_ANSWER}\n\n"
                    f"LOCAL DOCUMENT CONTEXT:\n{context}\n\n"
                    f"QUESTION:\n{query}\n\nANSWER:"
                )

                self.set_status("Generating answer with local Mistral...")
                answer = generate_answer(
                    query,
                    context
                )

                if generation_stop_event.is_set(): return

                answer = clean_text(answer)
                if not answer:
                    answer = "I could not generate an answer from the provided documents."

                last_response_text = answer

                if last_retrieved_sources:
                    source_text = ", ".join(last_retrieved_sources)
                    if source_text.lower() not in answer.lower():
                        answer += f"\n\nSource: {source_text}"

                self.after(0,self.append_chat,"ASSISTANT",answer)
                self.set_status("Ready")

            except Exception as exc:
                print("Generation error:",exc)
                self.after(0,self.append_chat,"SYSTEM",f"Generation error: {exc}")
            finally:
                is_generating = False
                self.after(0,self.finish_generation)

        generation_thread = threading.Thread(target=worker,daemon=True)
        generation_thread.start()

    def stop_generation(self):
        generation_stop_event.set()
        self.set_status("Stopping generation...")

    def finish_generation(self):
        self.query_entry.configure(state="normal")
        self.action_button.configure(state="normal",text="Send")
        self.query_entry.focus_set()
        generation_stop_event.clear()
        if not is_processing: self.set_status("Ready")

    def new_chat(self):
        global last_response_text,last_retrieved_sources,last_rag_context
        generation_stop_event.set()
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0","end")
        self.chat_box.configure(state="disabled")
        last_response_text = ""
        last_retrieved_sources = []
        last_rag_context = ""
        self.append_chat("SYSTEM","New chat started.")
        self.set_status("Ready")

    def on_close(self):
        generation_stop_event.set()
        processing_stop_event.set()
        try:
            if current_engine: current_engine.stop()
        except Exception: pass
        self.destroy()

def toggle_speech(button_widget):
    global is_speaking,current_engine
    if is_speaking:
        try:
            if current_engine: current_engine.stop()
        except Exception: pass
        is_speaking = False
        button_widget.configure(text="Speak")
        return
    if not last_response_text.strip(): return

    is_speaking = True
    button_widget.configure(text="Stop Voice")

    def worker():
        global is_speaking,current_engine
        try:
            import pyttsx3
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except Exception:
                pythoncom = None
            current_engine = pyttsx3.init()
            current_engine.setProperty("rate",150)
            current_engine.say(last_response_text)
            current_engine.runAndWait()
            if pythoncom: pythoncom.CoUninitialize()
        except Exception as exc:
            print("Speech error:",exc)
        finally:
            is_speaking = False
            try: button_widget.configure(text="Speak")
            except Exception: pass

    threading.Thread(target=worker,daemon=True).start()

def diagnostic():
    sync_core_rag_state()
    print("RAG IMPORT: OK")
    print("Chunks:",len(chunks))
    print("Embedder:",callable(getattr(embedder,"encode",None)))
    print("FAISS:",faiss_index.ntotal if faiss_index else None)
    print("Model exists:",os.path.exists(MODEL_PATH))

def main():
    print("="*70)
    print("        OFFLINE MULTIMODAL RAG AI ASSISTANT")
    print("="*70)
    print("Runtime: OFFLINE")
    print("RAG: FAISS")
    print("Embedding: all-MiniLM-L6-v2")
    print("LLM: Local Mistral 7B")
    print("Model:",MODEL_PATH)
    print("GPU layers:",GPU_LAYERS)
    print("Context:",CONTEXT_SIZE)
    print("CPU threads:",CPU_THREADS)
    print("Max output tokens:",MAX_OUTPUT_TOKENS)
    print("="*70)
    app = RAGApp()
    app.mainloop()

if __name__ == "__main__":
    main()





