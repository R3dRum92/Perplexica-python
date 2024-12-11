from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Enum,
    JSON,
    ForeignKey,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()


class RoleEnum(enum.Enum):
    assistant = "assistant"
    user = "user"


class Chat(Base):
    __tablename__ = "chats"

    id = Column(String, primary_key=True, index=True)
    title = Column(Text, nullable=False)
    createdAt = Column(Text, nullable=False)
    focusMode = Column(Text, nullable=False)
    files = Column(JSON, nullable=False, default=list)

    messages = relationship(
        "Message", back_populates="chat", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    chatId = Column(String, ForeignKey("chats.id"), nullable=False, index=True)
    messageId = Column(String, nullable=False, index=True)
    role = Column(Enum(RoleEnum), nullable=True)
    metadata = Column(JSON, nullable=True)

    chat = relationship("Chat", back_populates="messages")
