# 📷 Photo Categorizer

Automatically sort group photos by the people who appear in them — powered by deep learning face recognition.

> Built with Python + [DeepFace](https://github.com/serengil/deepface) (FaceNet model)

---

## How It Works

1. You add reference photos of each person to `known_faces/<Name>/`
2. `encode_faces.py` converts each face into a unique 128-number embedding vector and saves it
3. You drop group photos into `group_photos/`
4. `categorize.py` detects every face in each photo, compares it against saved embeddings, and copies the photo into `output/<Name>/` for each person found
5. Photos with no recognized face go to `output/unmatched/`

---

## Project Structure

```
photo-categorizer/
├── known_faces/          ← One folder per person with their reference photos
│   ├── Alice/
│   │   ├── alice1.jpg
│   │   └── alice2.jpg
│   └── Bob/
│       └── bob1.jpg
├── group_photos/         ← Drop group photos here
├── output/               ← Sorted results appear here
│   ├── Alice/
│   ├── Bob/
│   └── unmatched/        ← Photos with no recognized face
├── encodings/            ← Auto-generated face encodings (don't edit)
├── encode_faces.py       ← Step 1: Build face encodings
├── categorize.py         ← Step 2: Sort group photos
└── requirements.txt
```

---

## Setup

### 1. Prerequisites

- Python 3.8+
- pip

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install deepface tf-keras opencv-python numpy Pillow tqdm
```

> ⚠️ First run will download the FaceNet model weights (~90MB). This is automatic.

---

## Usage

### Step 1 — Add reference photos

Create one subfolder per person inside `known_faces/` and add 2–5 clear photos of their face.

```
known_faces/
    Alice/
        alice_front.jpg
        alice_side.jpg
    Bob/
        bob1.jpg
```

### Step 2 — Encode known faces

```bash
python encode_faces.py
```

Re-run this whenever you add new people or new reference photos.

### Step 3 — Add group photos

Copy all group photos into `group_photos/`.

### Step 4 — Categorize!

```bash
python categorize.py
```

Results are copied into `output/<Name>/` for every person detected. Unrecognized photos go to `output/unmatched/`.

---

## Tips

- Use **3–5 reference photos per person** for better accuracy
- Reference photos should be **clear, well-lit, and front-facing**
- Avoid sunglasses, hats, or heavy shadows in reference photos
- If someone is being **missed**, add more reference photos of them
- If you're getting **false matches**, lower the threshold in `categorize.py` from `0.4` to `0.35`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| No person folders found | Create named subfolders inside `known_faces/` |
| Face not detected in reference photo | Use a clearer, more front-facing photo |
| All photos go to `unmatched/` | Re-run `encode_faces.py` and check it encodes successfully |
| False matches | Lower cosine threshold from `0.4` → `0.3` in `categorize.py` |
| Person not detected in group photo | Add more reference photos or raise threshold to `0.45` |

---

## Tech Stack

- **[DeepFace](https://github.com/serengil/deepface)** — face detection and embedding generation
- **FaceNet** — Google's face recognition model (128-dim embeddings)
- **Cosine distance** — similarity metric for comparing face vectors
- **OpenCV** — image loading and processing
- **TensorFlow / Keras** — backend for deep learning models
