from .sqlite import SessionLocal
from .models import Conversation

def save_message(session_id, role, message):
    db = SessionLocal()
    entry = Conversation(session_id=session_id, role=role, message=message)
    db.add(entry)
    db.commit()
    db.close()

def get_conversation(session_id):
    db = SessionLocal()
    msgs = db.query(Conversation).filter_by(session_id=session_id).all()
    db.close()
    return msgs
