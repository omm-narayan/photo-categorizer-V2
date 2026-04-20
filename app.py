import pickle, shutil, os, base64
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder="static")
CORS(app)

KNOWN_FACES_DIR  = Path("known_faces")
GROUP_PHOTOS_DIR = Path("group_photos")
ENCODINGS_FILE   = Path("encodings/encodings.pkl")
OUTPUT_DIR       = Path("output")
SUPPORTED_EXT    = {".jpg", ".jpeg", ".png", ".webp"}

def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

# ── Output images ─────────────────────────────────────────────────────────────
@app.route("/output/<person>/<filename>")
def output_image(person, filename):
    return send_from_directory(OUTPUT_DIR / person, filename)

# ── Upload a known-face image ─────────────────────────────────────────────────
@app.route("/api/upload-known", methods=["POST"])
def upload_known():
    name  = request.form.get("name", "").strip()
    files = request.files.getlist("images")
    if not name:
        return jsonify({"error": "Person name is required"}), 400
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    dest = KNOWN_FACES_DIR / name
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if Path(f.filename).suffix.lower() in SUPPORTED_EXT:
            save_path = dest / f.filename
            f.save(save_path)
            saved.append(f.filename)

    return jsonify({"message": f"Saved {len(saved)} image(s) for '{name}'", "saved": saved})

# ── Encode all known faces ────────────────────────────────────────────────────
@app.route("/api/encode", methods=["POST"])
def encode_faces():
    try:
        from deepface import DeepFace
    except ImportError:
        return jsonify({"error": "deepface is not installed. Run: pip install deepface"}), 500

    known_encodings, known_names = [], []
    persons = [p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir()] if KNOWN_FACES_DIR.exists() else []

    if not persons:
        return jsonify({"error": "No person folders found in known_faces/"}), 400

    errors = []
    for person_dir in persons:
        name = person_dir.name
        images = [f for f in person_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXT]
        for img_path in images:
            try:
                emb = DeepFace.represent(str(img_path), model_name="Facenet", enforce_detection=True)
                known_encodings.append(emb[0]["embedding"])
                known_names.append(name)
            except Exception as e:
                errors.append(f"Skipped {img_path.name}: {e}")

    ENCODINGS_FILE.parent.mkdir(exist_ok=True)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    return jsonify({
        "message": f"Encoded {len(known_encodings)} face(s) across {len(persons)} person(s)",
        "people": sorted(set(known_names)),
        "errors": errors
    })

# ── Upload group photos & categorize ─────────────────────────────────────────
@app.route("/api/categorize", methods=["POST"])
def categorize():
    try:
        from deepface import DeepFace
    except ImportError:
        return jsonify({"error": "deepface is not installed. Run: pip install deepface"}), 500

    if not ENCODINGS_FILE.exists():
        return jsonify({"error": "No encodings found. Please encode known faces first."}), 400

    files = request.files.getlist("photos")
    if not files:
        return jsonify({"error": "No photos uploaded"}), 400

    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    known_encodings, known_names = data["encodings"], data["names"]
    unique_names = sorted(set(known_names))

    GROUP_PHOTOS_DIR.mkdir(exist_ok=True)
    for name in unique_names:
        (OUTPUT_DIR / name).mkdir(parents=True, exist_ok=True)

    # Save uploaded photos
    saved_paths = []
    for file in files:
        if Path(file.filename).suffix.lower() in SUPPORTED_EXT:
            dest = GROUP_PHOTOS_DIR / file.filename
            file.save(dest)
            saved_paths.append(dest)

    results = []
    for photo_path in saved_paths:
        try:
            faces = DeepFace.represent(str(photo_path), model_name="Facenet", enforce_detection=False)
            matched = set()
            for face in faces:
                for enc, name in zip(known_encodings, known_names):
                    if cosine_distance(face["embedding"], enc) < 0.4:
                        matched.add(name)

            if matched:
                for person in matched:
                    shutil.copy2(photo_path, OUTPUT_DIR / person / photo_path.name)
            else:
                (OUTPUT_DIR / "unmatched").mkdir(exist_ok=True)
                shutil.copy2(photo_path, OUTPUT_DIR / "unmatched" / photo_path.name)
                matched.add("unmatched")

            results.append({"photo": photo_path.name, "matched": list(matched)})
        except Exception as e:
            results.append({"photo": photo_path.name, "error": str(e)})

    return jsonify({"results": results})

# ── List people & their output photos ────────────────────────────────────────
@app.route("/api/gallery")
def gallery():
    if not OUTPUT_DIR.exists():
        return jsonify({"people": []})

    people = []
    for person_dir in sorted(OUTPUT_DIR.iterdir()):
        if person_dir.is_dir():
            photos = [
                f"/output/{person_dir.name}/{f.name}"
                for f in person_dir.iterdir()
                if f.suffix.lower() in SUPPORTED_EXT
            ]
            if photos:
                people.append({"name": person_dir.name, "photos": photos})
    return jsonify({"people": people})

# ── List encoded people ────────────────────────────────────────────────────────
@app.route("/api/people")
def get_people():
    if not ENCODINGS_FILE.exists():
        return jsonify({"people": []})
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    return jsonify({"people": sorted(set(data["names"]))})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
