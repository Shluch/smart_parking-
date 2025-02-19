from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import cv2
import numpy as np
import pickle
import os
import time
import csv
import base64

import firebase_admin
from firebase_admin import credentials, db

# Initialize the Firebase app
cred = credentials.Certificate('parking-77f66-firebase-adminsdk-fbsvc-9bdab7b973.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://parking-77f66-default-rtdb.firebaseio.com/'
})

app = Flask(__name__, static_folder="assets")
app.secret_key = "your_secret_key_here"  # Change this to your secret key


# -----------------------------
# Configuration and Helper Functions
# -----------------------------
DB_FILE = os.path.abspath("face_db.pkl")
EVENT_CSV_FILE = os.path.abspath("parking_log.csv")
REGISTERED_USERS_CSV = os.path.abspath("registered_users.csv")
MAX_SLOTS = 100
THRESHOLD = 50  # Recognition confidence threshold

# Predefined login credentials
USER_CREDENTIALS = {'username': '123', 'password': '123'}

def fetch_slot_statuses():
    try:
        # Reference to the 'slots' node in your Firebase Realtime Database
        slots_ref = db.reference('/slots')
        # Fetch the data at the reference (returns a dictionary)
        slots_data = slots_ref.get()
        return slots_data
    except Exception as e:
        print(f"Error fetching slot statuses: {e}")
        return None

def load_database(db_file=DB_FILE):
    if os.path.exists(db_file):
        with open(db_file, "rb") as f:
            return pickle.load(f)
    else:
        return []

def save_database(db_obj, db_file=DB_FILE):
    with open(db_file, "wb") as f:
        pickle.dump(db_obj, f)

def get_next_id(db_obj):
    return max(entry['id'] for entry in db_obj) + 1 if db_obj else 1

def get_next_slot(db_obj, max_slots=MAX_SLOTS):
    # Determine slots in use by entries with no exit_time and a defined slot
    used_slots = {entry.get("slot", 0) for entry in db_obj if entry.get("exit_time") is None and entry.get("slot") is not None}
    slot = 1
    while slot in used_slots and slot <= max_slots:
        slot += 1
    return None if slot > max_slots else slot

def train_recognizer(db_obj):
    active_entries = [entry for entry in db_obj if 'face' in entry]
    if not active_entries:
        return None
    images = [entry['face'] for entry in active_entries]
    labels = [entry['id'] for entry in active_entries]
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer

def log_event(event_details, csv_file=EVENT_CSV_FILE):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "phone", "adhaar", "vehicle", "slot", "event_type", "event_time"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(event_details)

def log_registered_user(user_details, csv_file=REGISTERED_USERS_CSV):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, mode="a", newline="") as f:
        fieldnames = ["id", "name", "phone", "adhaar", "vehicle", "registration_time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "id": user_details.get("id"),
            "name": user_details.get("name"),
            "phone": user_details.get("phone"),
            "adhaar": user_details.get("adhaar"),
            "vehicle": user_details.get("vehicle"),
            "registration_time": user_details.get("entry_time") if user_details.get("entry_time") else time.strftime("%Y-%m-%d %H:%M:%S")
        })

# -----------------------------
# Global Database and Recognizer
# -----------------------------
face_db = load_database()
recognizer = train_recognizer(face_db)

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')
    if username == USER_CREDENTIALS['username'] and password == USER_CREDENTIALS['password']:
        return redirect(url_for('home'))
    return "Invalid credentials, try again."

@app.route('/home')
def home():
    # Read event CSV data
    slot_statuses = fetch_slot_statuses()
    event_data = []
    if os.path.exists(EVENT_CSV_FILE):
        with open(EVENT_CSV_FILE, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                event_data.append(row)

    # Read registered users CSV data
    registered_data = []
    if os.path.exists(REGISTERED_USERS_CSV):
        with open(REGISTERED_USERS_CSV, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                registered_data.append(row)

    return render_template("home.html", event_data=event_data, registered_data=registered_data, slot_statuses=slot_statuses)

@app.route('/park')
def park():
    return render_template('park.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html', face_data="")
    else:
        name = request.form.get("name")
        phone = request.form.get("phone")
        adhaar = request.form.get("adhaar")
        vehicle = request.form.get("vehicle")
        face_data = request.form.get("face_data")
        
        if not (name and phone and adhaar and vehicle and face_data):
            flash("All fields and face data are required!")
            return redirect(url_for("register"))
        
        try:
            img_bytes = base64.b64decode(face_data)
            face_np = np.frombuffer(img_bytes, dtype=np.uint8)
            face_roi = cv2.imdecode(face_np, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            flash(f"Error decoding face data: {e}")
            return redirect(url_for("register"))
        
        new_user = {
            "id": get_next_id(face_db),
            "name": name,
            "phone": phone,
            "adhaar": adhaar,
            "vehicle": vehicle,
            "face": face_roi
        }
        
        face_db.append(new_user)
        save_database(face_db)
        global recognizer
        recognizer = train_recognizer(face_db)
        log_registered_user(new_user)
        flash("Registration successful!")
        return redirect(url_for("home"))


@app.route('/log_entry', methods=['POST'])
def log_entry():
    global recognizer  # Use global recognizer
    face_data = request.form.get("face_data")
    selected_slot = request.form.get("slot")  # Slot chosen by the user (from park.html)

    if not face_data:
        flash("Face data is required!")
        return redirect(url_for("park"))

    try:
        img_bytes = base64.b64decode(face_data)
        face_np = np.frombuffer(img_bytes, dtype=np.uint8)
        face_roi = cv2.imdecode(face_np, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        flash(f"Error decoding face data: {e}")
        return redirect(url_for("park"))

    if recognizer is None:
        flash("Face recognizer is not trained!")
        return redirect(url_for("park"))

    # Recognize the face
    label, confidence = recognizer.predict(face_roi)
    if confidence > THRESHOLD:
        flash("Face not recognized. Please register first!")
        return redirect(url_for("park"))

    # Retrieve user from the local face database
    user = next((entry for entry in face_db if entry["id"] == label), None)
    if user is None:
        flash("User not found in the database!")
        return redirect(url_for("park"))

    # Determine slot:
    if selected_slot:
        slot = int(selected_slot)
    else:
        slot = get_next_slot(face_db)
        if slot is None:
            flash("No available slot!")
            return redirect(url_for("park"))

    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # Helper function: get the last event for this user from the CSV log.
    def get_last_event_for_user(user_id):
        if os.path.exists(EVENT_CSV_FILE):
            with open(EVENT_CSV_FILE, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                last_event = None
                for row in reader:
                    if str(row["id"]) == str(user_id):
                        last_event = row
                return last_event
        else:
            return None

    last_event = get_last_event_for_user(user["id"])
    if last_event and last_event["event_type"] == "entry":
        new_event_type = "exit"
    else:
        new_event_type = "entry"

    # Log the event in the CSV file.
    log_event({
        "id": user["id"],
        "name": user["name"],
        "phone": user["phone"],
        "adhaar": user["adhaar"],
        "vehicle": user["vehicle"],
        "slot": slot,
        "event_type": new_event_type,
        "event_time": current_time
    })

    # Update the slot status in Firebase
    try:
        slots_ref = db.reference('/slots')
        if new_event_type == "entry":
            # Mark the slot as booked.
            slots_ref.update({f'slot{slot}': "1"})
            # slots_ref.child(f'slot{slot}').update({"status": "booked"})
        else:
            # Mark the slot as available (0).
            slots_ref.update({f'slot{slot}': "0"})
            # slots_ref.child(f'slot{slot}').update({"status": "0"})
    except Exception as e:
        flash(f"Failed to update slot status in Firebase: {e}")

    flash(f"{new_event_type.capitalize()} logged for {user['name']} at slot {slot}!")
    return redirect(url_for("home"))


if __name__ == '__main__':
    app.run(debug=True)
