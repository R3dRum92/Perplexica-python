from typing import List
from langchain.schema import BaseMessage


def format_chat_history_as_string(history: List[BaseMessage]) -> str:
    return "\n".join(
        [f"{type(message).__name__}: {message.content}" for message in history]
    )
