from pydantic import BaseModel

class AuthRequest(BaseModel):
    email: str
    password: str
