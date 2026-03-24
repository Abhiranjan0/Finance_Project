from fastapi import APIRouter, HTTPException
from finance_advisor.backend.db.sqlite import SessionLocal
from finance_advisor.backend.db.models import User
from finance_advisor.backend.utils.security import hash_password, verify_password
from finance_advisor.backend.models.auth import AuthRequest
import uuid

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
def register(payload: AuthRequest):

    email = payload.email
    password = payload.password

    db = SessionLocal()
    exists = db.query(User).filter_by(email=email).first()
    if exists:
        raise HTTPException(400, "Email already registered")

    user = User(
        user_id=str(uuid.uuid4()),
        email=email,
        password_hash=hash_password(password)
    )

    db.add(user)
    db.commit()

    user_id = user.user_id   # EXTRACT BEFORE CLOSING SESSION

    db.close()

    return {"message": "Registration successful", "user_id": user_id}



@router.post("/login")
def login(payload: AuthRequest):

    email = payload.email
    password = payload.password

    db = SessionLocal()
    user = db.query(User).filter_by(email=email).first()

    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    user_id = user.user_id
    db.close()
    return {"user_id": user_id}
