import json
from streamlit_lottie import st_lottie

def load_lottie(path: str):
    with open(path, "r") as f:
        return json.load(f)

def render_lottie(path: str, height: int = 200, key: str = None):
    animation_json = load_lottie(path)
    st_lottie(animation_json, height=height, key=key)
