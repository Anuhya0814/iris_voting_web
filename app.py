from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import sqlite3

app = Flask(__name__)

# ---------- FOLDER SETUP ----------
os.makedirs("voters/faces", exist_ok=True)
os.makedirs("voters/eyes", exist_ok=True)

# ---------- HAARCASCADE SETUP ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_xml = os.path.join(BASE_DIR, "haarcascades", "haarcascade_frontalface_default.xml")
eye_xml = os.path.join(BASE_DIR, "haarcascades", "haarcascade_eye.xml")

# Debugging print to confirm paths
print("Face Cascade Path:", face_xml)
print("Eye Cascade Path:", eye_xml)

# Validate cascade files
if not os.path.exists(face_xml) or not os.path.exists(eye_xml):
    raise FileNotFoundError(f"Haarcascade files not found! Check {face_xml} and {eye_xml}")

face_cascade = cv2.CascadeClassifier(face_xml)
eye_cascade = cv2.CascadeClassifier(eye_xml)

if face_cascade.empty() or eye_cascade.empty():
    raise IOError("Failed to load Haarcascade classifiers. Check your XML files.")

# ---------- DATABASE FUNCTIONS ----------
def create_db():
    """Create database tables if not exist"""
    conn = sqlite3.connect('voting.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id TEXT UNIQUE,
                 face_path TEXT,
                 eye_path TEXT,
                 has_voted INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE IF NOT EXISTS votes (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 voter_id TEXT,
                 candidate TEXT)''')
    conn.commit()
    conn.close()

def add_user(user_id, face_path, eye_path):
    conn = sqlite3.connect("voting.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (user_id, face_path, eye_path) VALUES (?, ?, ?)",
                  (user_id, face_path, eye_path))
        conn.commit()
    except sqlite3.IntegrityError:
        return False
    conn.close()
    return True

def get_user(user_id):
    conn = sqlite3.connect("voting.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    user = c.fetchone()
    conn.close()
    return user

def mark_voted(user_id):
    conn = sqlite3.connect("voting.db")
    c = conn.cursor()
    c.execute("UPDATE users SET has_voted=1 WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

def has_already_voted(user_id):
    user = get_user(user_id)
    return user[4] == 1 if user else False

def save_vote(user_id, candidate):
    conn = sqlite3.connect("voting.db")
    c = conn.cursor()
    c.execute("INSERT INTO votes (voter_id, candidate) VALUES (?, ?)", (user_id, candidate))
    conn.commit()
    conn.close()

# --------- FETCH VOTE COUNTS FOR RESULTS PAGE ---------
def get_vote_counts():
    conn = sqlite3.connect("voting.db")
    c = conn.cursor()
    c.execute("SELECT candidate, COUNT(*) FROM votes GROUP BY candidate")
    results = c.fetchall()
    conn.close()
    return results

# ---------- IMAGE CAPTURE ----------
def capture_image(user_id, img_type):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if img_type == "face":
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                file_path = f"voters/faces/{user_id}.jpg"
                cv2.imwrite(file_path, roi)
                cap.release()
                cv2.destroyAllWindows()
                return file_path

        elif img_type == "eye":
            eyes = eye_cascade.detectMultiScale(gray)
            for (x, y, w, h) in eyes:
                roi = frame[y:y+h, x:x+w]
                file_path = f"voters/eyes/{user_id}.jpg"
                cv2.imwrite(file_path, roi)
                cap.release()
                cv2.destroyAllWindows()
                return file_path

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

# ---------- IMAGE MATCHING ----------
def compare_images(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    if img1 is None or img2 is None:
        return False
    img1 = cv2.resize(img1, (200, 200))
    img2 = cv2.resize(img2, (200, 200))
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score > 0.8  # threshold

# ---------- ROUTES ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_id = request.form['user_id']
        face_path = capture_image(user_id, "face")
        eye_path = capture_image(user_id, "eye")
        if add_user(user_id, face_path, eye_path):
            return f"User {user_id} registered successfully!"
        else:
            return "User already exists!"
    return render_template('register.html')

@app.route('/vote', methods=['GET', 'POST'])
def vote():
    if request.method == 'POST':
        user_id = request.form['user_id']
        candidate = request.form['candidate']

        if has_already_voted(user_id):
            return "You have already voted!"

        new_face = capture_image(user_id, "face")
        new_eye = capture_image(user_id, "eye")
        user = get_user(user_id)

        if not user:
            return "User not found!"

        registered_face = user[2]
        registered_eye = user[3]

        if compare_images(new_face, registered_face) and compare_images(new_eye, registered_eye):
            save_vote(user_id, candidate)
            mark_voted(user_id)
            return "Vote registered successfully!"
        else:
            return "Verification failed! Vote not registered."
    return render_template('vote.html')

# ---------- RESULTS PAGE ----------
@app.route('/results')
def results():
    vote_counts = get_vote_counts()
    return render_template('results.html', vote_counts=vote_counts)

# ---------- MAIN ----------
if __name__ == '__main__':
    create_db()
    app.run(debug=True)
