"""
Session & Chat History Manager
Handles conversation persistence, system prompt presets, and chat export functionality.
"""

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

SYSTEM_PRESETS = {
    "Helpful Assistant": "You are a helpful, respectful, and honest AI assistant. Always answer accurately and concisely.",
    "Senior Software Engineer": "You are a world-class senior software architect. Provide clean, modular, bug-free, and well-commented code. Explain key design patterns and edge cases.",
    "Creative Writer & Storyteller": "You are a master storyteller and creative writer. Use evocative descriptions, compelling pacing, and rich character dialogues.",
    "JSON Structured Extractor": "You are a data extraction and transformation assistant. Always respond in valid, well-structured JSON without conversational commentary.",
    "Executive Summarizer": "You are an executive assistant. Summarize complex information into crisp, high-impact bullet points with clear actionable takeaways.",
    "Math & Scientific Reasoner": "You are an expert mathematician and scientific researcher. Break down problems step-by-step using first principles and rigorous logic.",
}


class ChatSession:
    def __init__(self, session_id: Optional[str] = None, title: str = "New Conversation"):
        self.id = session_id or str(uuid.uuid4())
        self.title = title
        self.created_at = time.time()
        self.updated_at = time.time()
        self.system_prompt = SYSTEM_PRESETS["Helpful Assistant"]
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str, metrics: Optional[Dict[str, Any]] = None):
        self.messages.append({
            "id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metrics": metrics or {}
        })
        self.updated_at = time.time()
        if len(self.messages) == 1 and role == "user" and self.title == "New Conversation":
            # Auto-title from the first prompt
            clean_title = content.strip().split("\n")[0][:36]
            self.title = clean_title if len(content) <= 36 else clean_title + "..."

    def clear(self):
        self.messages.clear()
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "system_prompt": self.system_prompt,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        session = cls(session_id=data.get("id"), title=data.get("title", "Conversation"))
        session.created_at = data.get("created_at", time.time())
        session.updated_at = data.get("updated_at", time.time())
        session.system_prompt = data.get("system_prompt", SYSTEM_PRESETS["Helpful Assistant"])
        session.messages = data.get("messages", [])
        return session

    def export_markdown(self) -> str:
        lines = [f"# {self.title}\n", f"*System Prompt:* {self.system_prompt}\n", "---\n"]
        for msg in self.messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            metrics = msg.get("metrics", {})
            metrics_str = ""
            if metrics and role.lower() == "assistant":
                tok_s = metrics.get("tok_per_sec", 0.0)
                toks = metrics.get("tokens", 0)
                if tok_s > 0 or toks > 0:
                    metrics_str = f" *(Generated {toks} tokens @ {tok_s:.1f} tok/s)*"

            lines.append(f"### {role}{metrics_str}\n\n{content}\n\n---\n")
        return "".join(lines)


class SessionManager:
    def __init__(self, storage_dir: str = "data/sessions"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.sessions: Dict[str, ChatSession] = {}
        self.active_session_id: Optional[str] = None
        self.load_all()

    def load_all(self):
        self.sessions.clear()
        if not os.path.exists(self.storage_dir):
            return

        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                fpath = os.path.join(self.storage_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        session = ChatSession.from_dict(data)
                        self.sessions[session.id] = session
                except Exception as e:
                    print(f"Error loading session {fname}: {e}")

        if not self.sessions:
            new_s = self.create_session()
            self.active_session_id = new_s.id
        else:
            # Sort by updated_at descending
            sorted_sessions = sorted(self.sessions.values(), key=lambda s: s.updated_at, reverse=True)
            self.active_session_id = sorted_sessions[0].id

    def create_session(self, title: str = "New Conversation") -> ChatSession:
        session = ChatSession(title=title)
        self.sessions[session.id] = session
        self.active_session_id = session.id
        self.save_session(session)
        return session

    def get_active_session(self) -> ChatSession:
        if not self.active_session_id or self.active_session_id not in self.sessions:
            return self.create_session()
        return self.sessions[self.active_session_id]

    def save_session(self, session: ChatSession):
        fpath = os.path.join(self.storage_dir, f"{session.id}.json")
        try:
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving session {session.id}: {e}")

    def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
            fpath = os.path.join(self.storage_dir, f"{session_id}.json")
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception as e:
                    print(f"Error deleting file {fpath}: {e}")

        if self.active_session_id == session_id:
            if self.sessions:
                sorted_sessions = sorted(self.sessions.values(), key=lambda s: s.updated_at, reverse=True)
                self.active_session_id = sorted_sessions[0].id
            else:
                self.active_session_id = None

    def list_sessions(self) -> List[ChatSession]:
        return sorted(self.sessions.values(), key=lambda s: s.updated_at, reverse=True)
