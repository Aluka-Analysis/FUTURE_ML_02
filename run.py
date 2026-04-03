import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.main import app
import uvicorn

if __name__ == "__main__":
    # Remove reload=True or use import string format
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)