import os


PROJECT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_DIR = os.path.abspath(os.getenv("RESEARCHGPT_DATA_DIR", PROJECT_DATA_DIR))
os.makedirs(DATA_DIR, exist_ok=True)
