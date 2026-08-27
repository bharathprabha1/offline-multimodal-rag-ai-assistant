import json
import os
import datetime

from .config import HISTORY_FILE


class HistoryManager:

    def __init__(self):
        self.filepath = HISTORY_FILE
        self.history_data = self.load_history()
        self.start_new_session()

    def start_new_session(self):

        self.current_session_id = (
            datetime.datetime.now()
            .strftime("%Y-%m-%d %H:%M:%S")
        )

        if self.current_session_id not in self.history_data:

            self.history_data[self.current_session_id] = {
                "title": (
                    f"New Chat "
                    f"{datetime.datetime.now().strftime('%H:%M')}"
                ),
                "messages": []
            }

            self.save_history()

    def load_history(self):

        if not os.path.exists(self.filepath):
            return {}

        try:

            with open(
                self.filepath,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

                if isinstance(data, dict):
                    return data

        except Exception:
            pass

        return {}

    def save_history(self):

        try:

            with open(
                self.filepath,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    self.history_data,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

        except Exception as e:
            print("History save error:", e)

    def save_message(self, role, text):

        if self.current_session_id not in self.history_data:

            self.history_data[self.current_session_id] = {
                "title": "New Chat",
                "messages": []
            }

        session = self.history_data[self.current_session_id]

        if (
            role == "user"
            and len(session["messages"]) == 0
        ):

            session["title"] = (
                text[:30] + "..."
                if len(text) > 30
                else text
            )

        session["messages"].append(
            {
                "role": role,
                "text": text,
                "timestamp": str(
                    datetime.datetime.now()
                )
            }
        )

        self.save_history()

    def get_sessions(self):

        sessions = [
            (k, v.get("title", "Chat"))
            for k, v in self.history_data.items()
        ]

        return sorted(
            sessions,
            key=lambda x: x[0],
            reverse=True
        )

    def get_session_messages(self, session_id):

        return (
            self.history_data
            .get(session_id, {})
            .get("messages", [])
        )


history_manager = HistoryManager()
