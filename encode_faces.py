import pickle
from pathlib import Path
from deepface import DeepFace
import numpy as np

KNOWN_FACES_DIR = Path("known_faces")
ENCODINGS_FILE = Path("encodings/encodings.pkl")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def encode_known_faces():
    known_encodings, known_names = [], []
    persons = [p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir()]
    if not persons:
        print("No person folders found in known_faces/")
        return
    print(f"Found {len(persons)} person(s)")
    for person_dir in persons:
        name = person_dir.name
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        for img_path in images:
            try:
                emb = DeepFace.represent(str(img_path), model_name="Facenet", enforce_detection=True)
                known_encodings.append(emb[0]["embedding"])
                known_names.append(name)
                print(f"Encoded {img_path.name} -> {name}")
            except Exception as e:
                print(f"Skipped {img_path.name}: {e}")
    ENCODINGS_FILE.parent.mkdir(exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)
    print(f"Done! Saved {len(known_encodings)} encoding(s)")

if __name__ == "__main__":
    encode_known_faces()