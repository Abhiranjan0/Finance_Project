from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.sql import func
from .sqlite import Base
from sqlalchemy import Boolean

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)
    email = Column(String, unique=True)
    password_hash = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())



class Conversation(Base):
    __tablename__ = "conversation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String)
    role = Column(String)       # "user" or "assistant"
    message = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
