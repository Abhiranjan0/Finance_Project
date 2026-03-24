from .sqlite import SessionLocal
from .models import User

from .sqlite import SessionLocal
from .models import User
import uuid

def ensure_user(session_id: str):
    """
    TEMPORARY LOGIC:
    Since the User model no longer includes session_id,
    we create a placeholder user for this session if one does not exist.
    
    Later, when login is implemented, this will be replaced.
    """

    db = SessionLocal()

    # Look for a user whose user_id matches the session
    user = db.query(User).filter_by(user_id=session_id).first()

    # If not found → create a new temporary user
    if not user:
        user = User(
            user_id=session_id,
            email=None,
            password_hash=None,
        )
        db.add(user)
        db.commit()

    db.close()
