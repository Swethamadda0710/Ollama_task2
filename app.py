"""Tkinter desktop application for the College Rules PDF chatbot."""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from chat_memory import ChatMemory
from rag_pipeline import CollegeRAG, ChatResult


class CollegeChatbotApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI College Rules & Regulations Chatbot")
        self.root.geometry("900x680")
        self.root.minsize(700, 500)
        self.root.configure(bg="#F4EFE6")
        self.memory = ChatMemory()
        self.rag = CollegeRAG()
        self.busy = False
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#F4EFE6")
        style.configure("Toolbar.TFrame", background="#172A3A")
        style.configure("Status.TLabel", background="#F4EFE6", foreground="#43505C")
        style.configure(
            "Accent.TButton",
            background="#E76F51",
            foreground="#FFFFFF",
            padding=(14, 8),
            borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", "#C9573C")])
        style.configure(
            "Quiet.TButton",
            background="#D8E2DC",
            foreground="#172A3A",
            padding=(14, 8),
            borderwidth=0,
        )
        style.map("Quiet.TButton", background=[("active", "#B8CDBF")])
        style.configure(
            "Chat.TEntry",
            fieldbackground="#FFFFFF",
            foreground="#172A3A",
            padding=8,
        )

        toolbar = ttk.Frame(self.root, padding=10, style="Toolbar.TFrame")
        toolbar.pack(fill=tk.X)
        self.upload_button = ttk.Button(
            toolbar,
            text="Upload College PDF",
            command=self.upload_pdf,
            style="Accent.TButton",
        )
        self.upload_button.pack(side=tk.LEFT)
        ttk.Button(
            toolbar, text="Clear Chat", command=self.clear_chat, style="Quiet.TButton"
        ).pack(side=tk.RIGHT)

        self.status = tk.StringVar(value="Upload a college PDF to begin.")
        ttk.Label(
            self.root, textvariable=self.status, padding=(10, 8), style="Status.TLabel"
        ).pack(fill=tk.X)

        chat_frame = ttk.Frame(self.root, padding=10, style="App.TFrame")
        chat_frame.pack(fill=tk.X, expand=False)
        self.chat = tk.Text(
            chat_frame,
            height=14,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 11),
            background="#FFFDF8",
            foreground="#172A3A",
            insertbackground="#172A3A",
            relief=tk.FLAT,
            padx=14,
            pady=14,
        )
        self.chat.tag_configure("user", foreground="#C9573C", font=("Segoe UI", 11, "bold"))
        self.chat.tag_configure("ai", foreground="#247A78", font=("Segoe UI", 11, "bold"))
        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat.yview)
        self.chat.configure(yscrollcommand=scrollbar.set)
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        bottom = ttk.Frame(self.root, padding=10, style="App.TFrame")
        bottom.pack(fill=tk.X)
        ttk.Label(
            bottom, text="Your question", style="Status.TLabel"
        ).pack(anchor=tk.W, pady=(0, 5))
        self.entry = tk.Text(
            bottom,
            height=5,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            background="#FFFFFF",
            foreground="#172A3A",
            insertbackground="#172A3A",
            relief=tk.FLAT,
            padx=10,
            pady=8,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.entry.bind("<Control-Return>", lambda _event: self.send_question())
        self.send_button = ttk.Button(
            bottom, text="Send", command=self.send_question, style="Accent.TButton"
        )
        self.send_button.pack(side=tk.LEFT, padx=(8, 0))

    def _append(self, speaker: str, text: str) -> None:
        self.chat.configure(state=tk.NORMAL)
        tag = "user" if speaker == "You" else "ai"
        self.chat.insert(tk.END, f"{speaker}:\n", tag)
        self.chat.insert(tk.END, f"{text}\n\n")
        self.chat.configure(state=tk.DISABLED)
        self.chat.see(tk.END)

    def upload_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Select college PDF", filetypes=[("PDF files", "*.pdf")]
        )
        if not path or self.busy:
            return
        self._set_busy(True, "Indexing PDF with Ollama embeddings...")
        threading.Thread(target=self._index_pdf, args=(path,), daemon=True).start()

    def _index_pdf(self, path: str) -> None:
        try:
            chunks = self.rag.load_pdf(path)
            self.root.after(0, lambda: self.status.set(
                f"Ready: {Path(path).name} ({chunks} searchable chunks)"
            ))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("PDF error", str(exc)))
            self.root.after(0, lambda: self.status.set("PDF indexing failed."))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def send_question(self) -> None:
        question = self.entry.get("1.0", tk.END).strip()
        if not question or self.busy:
            return
        if self.rag.vector_store is None:
            messagebox.showinfo("Upload required", "Please upload a college PDF first.")
            return
        self.entry.delete("1.0", tk.END)
        self._append("You", question)
        self.memory.add_user_message(question)
        self._set_busy(True, "Searching the PDF...")
        threading.Thread(target=self._answer, args=(question,), daemon=True).start()

    def _answer(self, question: str) -> None:
        try:
            result: ChatResult = self.rag.ask(question, self.memory)
            text = f"[{result.source}]\n{result.answer}"
            if result.comparison:
                text += f"\n\nWikipedia vs Ollama check:\n{result.comparison}"
            self.root.after(0, lambda: self._show_answer(text))
        except Exception as exc:
            error_text = str(exc)
            self.root.after(0, lambda: messagebox.showerror("Answer error", error_text))
        finally:
            self.root.after(0, lambda: self._set_busy(False))

    def _show_answer(self, text: str) -> None:
        self._append("AI", text)
        self.memory.add_ai_message(text)
        self.status.set("Ready for the next question.")

    def clear_chat(self) -> None:
        self.memory.clear()
        self.chat.configure(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.configure(state=tk.DISABLED)
        self.status.set("Chat cleared. The indexed PDF is still available.")

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.upload_button.configure(state=state)
        self.send_button.configure(state=state)
        self.entry.configure(state=state)
        if status:
            self.status.set(status)


def main() -> None:
    root = tk.Tk()
    CollegeChatbotApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
