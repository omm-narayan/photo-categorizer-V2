import pickle, shutil
from pathlib import Path
import numpy as np
from deepface import DeepFace

ENCODINGS_FILE = Path("encodings/encodings.pkl")
GROUP_PHOTOS_DIR = Path("group_photos")
OUTPUT_DIR = Path("output")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def categorize():
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_encodings, known_names = data["encodings"], data["names"]
    unique_names = sorted(set(known_names))
    for name in unique_names:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)
    photos = [f for f in GROUP_PHOTOS_DIR.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    print(f"Processing {len(photos)} photo(s)...")
    for photo_path in photos:
        try:
            faces = DeepFace.represent(str(photo_path), model_name="Facenet", enforce_detection=False)
            matched = set()
            for face in faces:
                for enc, name in zip(known_encodings, known_names):
                    if cosine_distance(face["embedding"], enc) < 0.4:
                        matched.add(name)
            print(f"{photo_path.name} -> {matched if matched else 'no match'}")
            if matched:
                for person in matched:
                    shutil.copy2(photo_path, OUTPUT_DIR / person / photo_path.name)
            else:
                (OUTPUT_DIR / "unmatched").mkdir(exist_ok=True)
                shutil.copy2(photo_path, OUTPUT_DIR / "unmatched" / photo_path.name)
        except Exception as e:
            print(f"Error on {photo_path.name}: {e}")

if __name__ == "__main__":
    categorize()