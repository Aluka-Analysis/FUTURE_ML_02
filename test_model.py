import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models"

print("=== TESTING MODEL LOADS ===\n")

# Test each file
files = ["model_queue.pkl", "model_priority.pkl", "vectorizer.pkl"]

for file in files:
    file_path = MODEL_PATH / file
    print(f"Testing: {file}")
    print(f"  Path: {file_path}")
    print(f"  Exists: {file_path.exists()}")
    print(f"  Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        data = joblib.load(file_path)
        print(f"  ✅ Loaded successfully")
        print(f"  Type: {type(data)}")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    print()

print("=== DONE ===")