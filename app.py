import base64
import os
import smtplib
import ssl
from io import BytesIO
from json import JSONEncoder
from typing import Dict, Union, List
import comtypes
import cv2
import geocoder
import numpy as np
import pyttsx3
import requests
import speech_recognition as sr
import tensorflow as tf
import datetime as dt
from email.mime.text import MIMEText
from datetime import datetime, time as dt_time, timedelta
from email.mime.multipart import MIMEMultipart
from deep_translator import GoogleTranslator
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from gtts import gTTS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import logging
import time

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Initialize database
db = SQLAlchemy(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    alarms = db.relationship('Alarm', backref='user', lazy=True)
    contacts = db.relationship('EmergencyContact', backref='user', lazy=True)
    health_data = db.relationship('HealthData', backref='user', lazy=True)

class Alarm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    alarm_type = db.Column(db.String(20), nullable=False)  # 'Wake Up', 'Tablets', 'Sleep'
    time = db.Column(db.String(5), nullable=False)  # HH:MM format

class HealthData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    heart_rate = db.Column(db.Integer)
    blood_pressure = db.Column(db.String(10))

class EmergencyContact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    phone = db.Column(db.String(20))
    relationship = db.Column(db.String(50))
    
    def __repr__(self):
        return f"EmergencyContact('{self.name}', '{self.email}')"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Custom JSON Encoder
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, dt.time):
            return obj.isoformat()
        return super().default(obj)

app.json_encoder = CustomJSONEncoder

# Initialize speech engine
comtypes.CoInitialize()
engine = pyttsx3.init()

# Constants
AUDIO_PATHS = {
    'wake': "static/sounds/wake_up.mp3",
    'tablets': "static/sounds/tablets.mp3",
    'sleep': "static/sounds/sleep.mp3",
    'emergency': "static/sounds/emergency.mp3"
}

TOGETHER_API_KEY = "4e5ed785b761e8a31e04bcd6529761f554c27030601eb1a163bb1a0dd23487fd"

LANGUAGES = {
    'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta', 'Telugu': 'te',
    'Bengali': 'bn', 'Kannada': 'kn', 'Gujarati': 'gu', 'Marathi': 'mr',
    'Punjabi': 'pa', 'Malayalam': 'ml'
}

CLASS_NAMES = {0: 'Chair', 1: 'Tablets', 2: 'Bottle'}
FACE_CLASS_NAMES = {0: 'Bhavat', 1: 'Vignesh'}

# Initialize models
model = tf.keras.models.load_model('models/object_classifier.h5')
face_model = tf.keras.models.load_model('models/face_detection.h5')

# Warm up models
dummy_input = np.zeros((1, 224, 224, 3))
_ = model.predict(dummy_input)
_ = face_model.predict(dummy_input)

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html', user=current_user)

@app.route('/')
@login_required
def home():
    init_session_defaults()
    return render_template('home.html')

# Health Monitoring Routes
@app.route('/health_monitoring', methods=['GET', 'POST'])
@login_required
def health_monitoring():
    # Ensure alarms exist for this user
    wake_alarm = Alarm.query.filter_by(user_id=current_user.id, alarm_type='wake').first()
    tablets_alarm = Alarm.query.filter_by(user_id=current_user.id, alarm_type='tablets').first()
    sleep_alarm = Alarm.query.filter_by(user_id=current_user.id, alarm_type='sleep').first()

    # Create alarms if they don't exist
    if not wake_alarm:
        wake_alarm = Alarm(user_id=current_user.id, alarm_type='wake', time='07:00')
        db.session.add(wake_alarm)
    if not tablets_alarm:
        tablets_alarm = Alarm(user_id=current_user.id, alarm_type='tablets', time='08:00')
        db.session.add(tablets_alarm)
    if not sleep_alarm:
        sleep_alarm = Alarm(user_id=current_user.id, alarm_type='sleep', time='22:00')
        db.session.add(sleep_alarm)

    if request.method == 'POST':
        # Update alarm times
        wake_alarm.time = request.form.get('wake_up_time', wake_alarm.time)
        tablets_alarm.time = request.form.get('tablets_time', tablets_alarm.time)
        sleep_alarm.time = request.form.get('sleep_time', sleep_alarm.time)
        
        # Save health data if provided
        heart_rate = request.form.get('heart_rate')
        blood_pressure = request.form.get('blood_pressure')
        
        if heart_rate or blood_pressure:
            health_data = HealthData(
                user_id=current_user.id,
                heart_rate=heart_rate,
                blood_pressure=blood_pressure
            )
            db.session.add(health_data)
        
        db.session.commit()
        flash('Settings saved successfully!', 'success')
        return redirect(url_for('health_monitoring'))
    
    # Get health data
    health_data = HealthData.query.filter_by(user_id=current_user.id).order_by(HealthData.date.desc()).all()
    
    return render_template('health_monitoring.html',
                        wake_up_time=wake_alarm.time,
                        tablets_time=tablets_alarm.time,
                        sleep_time=sleep_alarm.time,
                        health_data=health_data,
                        current_time=datetime.now().strftime("%H:%M:%S %p"))

# Voice Assistant Routes
@app.route('/voice_assistant', methods=['GET', 'POST'])
@login_required
def voice_assistant():
    if request.method == 'POST':
        if request.form.get('listen'):
            command = listen()
            session['user_query'] = command
            return jsonify({'command': command})
        
        elif request.form.get('send_message'):
            user_query = request.form.get('user_query')
            target_lang_code = request.form.get('target_language', 'en')
            
            if user_query:
                ai_response = voice_companion(user_query)
                audio_b64, translated = play_audio(ai_response, target_lang_code)
                
                session['messages'] = session.get('messages', [])
                session['messages'].extend([
                    {"role": "USER", "content": user_query, "timestamp": datetime.now().strftime("%H:%M")},
                    {"role": "assistant", "content": ai_response, "translated": translated, "timestamp": datetime.now().strftime("%H:%M")}
                ])
                
                return jsonify({
                    'ai_response': ai_response,
                    'audio_b64': audio_b64,
                    'translated': translated
                })
    
    return render_template('voice_assistant.html',
                         languages=LANGUAGES,
                         messages=session.get('messages', []))


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
@login_required
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
    return jsonify({'status': 'success'})

@app.route('/object_detection', methods=['POST'])
@login_required
def object_detection():
    if model is None:
        return jsonify({'error': 'Object detection model not loaded'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        class_name, confidence = predict_image(img_rgb)
        result = {
            'status': 'success',
            'filename': filename,
            'prediction': class_name,
            'confidence': f"{confidence*100:.2f}%",
            'heatmap_available': False
        }
        
        # Try to generate heatmap if possible
        base_layer, conv_layer = get_last_conv_layer()
        if base_layer and conv_layer:
            try:
                grad_model = tf.keras.models.Model(
                    inputs=model.inputs,
                    outputs=[conv_layer.output, model.layers[-1].output]
                )
                
                img_array = np.expand_dims(img_rgb / 255.0, axis=0)
                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(img_array)
                    loss = predictions[:, np.argmax(predictions[0])]
                
                grads = tape.gradient(loss, conv_outputs)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
                heatmap = np.maximum(heatmap, 0)
                heatmap /= np.max(heatmap)
                heatmap = cv2.resize(heatmap[0], (img.shape[1], img.shape[0]))
                
                heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
                
                result_filename = 'result_' + filename
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                cv2.imwrite(result_path, superimposed_img)
                
                result['result_filename'] = result_filename
                result['heatmap_available'] = True
            except Exception as e:
                print(f"Heatmap generation failed: {e}")
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/face_detection', methods=['POST'])
@login_required
def face_detection():
    if face_model is None:
        return jsonify({'error': 'Face detection model not loaded'})
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        class_name, confidence = predict_face(img_rgb)
        result = {
            'status': 'success',
            'filename': filename,
            'prediction': class_name,
            'confidence': f"{confidence*100:.2f}%"
        }
        
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'})

@app.route('/object_detection', methods=['GET'])
@login_required
def object_detection_page():
    return render_template('object_detection.html')
@app.route('/face_detection', methods=['GET'])
@login_required
def face_detection_page():
    return render_template('face_detection.html')


# Emergency SOS Routes
@app.route('/emergency_sos')
@login_required
def emergency_sos():
    return render_template('emergency_sos.html')

@app.route('/send_emergency', methods=['POST'])
@login_required
def send_emergency():
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
    if not contacts:
        return jsonify({'status': 'error', 'message': 'No emergency contacts found'})
    
    sender = LocationEmailSender()
    results = []
    
    for contact in contacts:
        success = sender.send_email(
            sender_email='julururahul13@gmail.com',
            sender_password='knaz seay bhph ifrf',
            recipient=contact.email
        )
        results.append({
            'status': 'success' if success else 'error',
            'recipient': contact.email
        })
    
    if all(r['status'] == 'success' for r in results):
        return jsonify({'status': 'success'})
    elif any(r['status'] == 'success' for r in results):
        return jsonify({
            'status': 'partial_success',
            'results': results,
            'message': 'Some alerts failed to send'
        })
    else:
        return jsonify({'status': 'error', 'results': results})

@app.route('/add_contact', methods=['POST'])
@login_required
def add_contact():
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    relationship = request.form.get('relationship')
    
    if not name or not email:
        return jsonify({'status': 'error', 'message': 'Name and email are required'})
    
    new_contact = EmergencyContact(
        user_id=current_user.id,
        name=name,
        email=email,
        phone=phone,
        relationship=relationship
    )
    
    try:
        db.session.add(new_contact)
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': 'Contact added successfully',
            'contact': {
                'id': new_contact.id,
                'name': new_contact.name,
                'email': new_contact.email,
                'phone': new_contact.phone,
                'relationship': new_contact.relationship
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/delete_contact/<int:contact_id>', methods=['DELETE'])
@login_required
def delete_contact(contact_id):
    contact = EmergencyContact.query.get_or_404(contact_id)
    
    if contact.user_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Unauthorized'}), 403
    
    try:
        db.session.delete(contact)
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Contact deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_contacts')
@login_required
def get_contacts():
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
    contacts_data = [{
        'id': contact.id,
        'name': contact.name,
        'email': contact.email,
        'phone': contact.phone,
        'relationship': contact.relationship
    } for contact in contacts]
    
    return jsonify({'status': 'success', 'contacts': contacts_data})

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Then modify your check_alarm route:
@app.route('/check_alarm')
@login_required
def check_alarm():
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    current_datetime = now.replace(second=0, microsecond=0)
    
    # Ensure session has a record of last triggered alarms
    if 'last_triggered_alarms' not in session:
        session['last_triggered_alarms'] = {}

    alarms = Alarm.query.filter_by(user_id=current_user.id).all()
    response_data = {'alarm_triggered': False}
    
    for alarm in alarms:
        alarm_time = datetime.strptime(alarm.time, "%H:%M").replace(
            year=now.year, month=now.month, day=now.day
        )
        
        # Check if current time matches the alarm time and it hasn't been triggered yet
        if current_datetime == alarm_time and session['last_triggered_alarms'].get(alarm.alarm_type) != current_time:
            session['last_triggered_alarms'][alarm.alarm_type] = current_time
            session.modified = True  # Ensure session changes are saved
            response_data = {
                'alarm_triggered': True,
                'type': alarm.alarm_type,
                'message': f"It's time for your {alarm.alarm_type}!",
                'audio_file': f'/static/sounds/{alarm.alarm_type}.mp3'
            }
            break  # Only trigger one alarm at a time
    
    return jsonify(response_data)


@app.route('/get_current_time')
def get_current_time():
    return jsonify({'current_time': datetime.now().strftime("%H:%M:%S %p")})


# Helper Functions
def init_session_defaults():
    defaults = {
        'wake_up_time': '07:00',
        'tablets_time': '08:00',
        'sleep_time': '22:00',
        'messages': [],
        'health_data': [],
        'last_alarm': None
    }
    for key, value in defaults.items():
        if key not in session:
            session[key] = value

def parse_time(time_str):
    """Convert 'HH:MM' string to time object"""
    hh, mm = map(int, time_str.split(':'))
    return dt_time(hh, mm)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            return recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand."
        except sr.RequestError:
            return "Speech service is unavailable."
        except sr.WaitTimeoutError:
            return "No speech detected."

def voice_companion(inputs):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": inputs}
        ],
        "temperature": 0.7
    }
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers=headers,
        json=data
    )
    return (response.json()["choices"][0]["message"]["content"] 
            if response.status_code == 200 
            else f"Error: {response.status_code} - {response.text}")

def play_audio(text, lang_code):
    try:
        translated = GoogleTranslator(source='en', target=lang_code).translate(text)
        tts = gTTS(translated, lang=lang_code)
        buffer = BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode()
        return audio_b64, translated
    except Exception as e:
        print(f"Audio error: {e}")
        return None, None

def get_last_conv_layer():
    """Find the last convolutional layer in MobileNetV2"""
    for layer in model.layers:
        if 'mobilenetv2' in layer.name.lower():
            for sub_layer in layer.layers[::-1]:
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return layer, sub_layer
    return None, None

def predict_image(img_array):
    """Make prediction on an image array"""
    if model is None:
        return "Model not loaded", 0.0
    
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])
    return CLASS_NAMES[class_idx], confidence

def predict_image(img_array):
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])
    return CLASS_NAMES.get(class_idx, "Unknown"), confidence

def predict_face(img_array):
    """Make prediction on an image array for face detection"""
    if face_model is None:
        return "Face model not loaded", 0.0
    
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_to_array(img_array)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    preds = face_model.predict(img_array, verbose=0)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])
    return FACE_CLASS_NAMES[class_idx], confidence

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Perform detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            class_name, confidence = predict_image(frame_rgb)
            
            # Add prediction to frame
            cv2.putText(frame,
                    f"{class_name}: {confidence:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

class LocationEmailSender:
    @staticmethod
    def get_location():
        try:
            g = geocoder.ip('me')
            if g.ok:
                return {
                    'address': g.address,
                    'city': g.city,
                    'state': g.state,
                    'country': g.country,
                    'latitude': g.lat,
                    'longitude': g.lng
                }
        except:
            pass
       
        try:
            response = requests.get('https://ipinfo.io/json', timeout=5)
            if response.status_code == 200:
                data = response.json()
                loc = data.get('loc', '').split(',')
                return {
                    'ip': data.get('ip'),
                    'city': data.get('city'),
                    'region': data.get('region'),
                    'country': data.get('country'),
                    'latitude': float(loc[0]) if loc else None,
                    'longitude': float(loc[1]) if len(loc) > 1 else None
                }
        except:
            pass
       
        return {'error': 'Could not determine location'}

    @staticmethod
    def format_message(location_data: dict) -> tuple:
        """Format location data into text and HTML versions."""
        if 'error' in location_data:
            return ("Could not determine device location.",
                    "<p>Could not determine device location.</p>")
       
        # Text version
        text = "📍 Current Device Location:\n\n"
        if 'address' in location_data:
            text += f"🏠 Address: {location_data['address']}\n"
        if 'city' in location_data:
            text += f"🏙️ City: {location_data['city']}\n"
        if 'region' in location_data:
            text += f"🗺️ Region: {location_data['region']}\n"
        if 'country' in location_data:
            text += f"🌎 Country: {location_data['country']}\n"
        if 'latitude' and 'longitude' in location_data:
            text += f"📌 Coordinates: {location_data['latitude']}, {location_data['longitude']}\n"
            text += f"🗺️ Google Maps: https://maps.google.com/?q={location_data['latitude']},{location_data['longitude']}\n"
       
        # HTML version
        html = f"""
        <h2>📍 Device Location Report</h2>
        <p><strong>🏠 Address:</strong> {location_data.get('address', 'N/A')}</p>
        <p><strong>🏙️ City:</strong> {location_data.get('city', 'N/A')}</p>
        <p><strong>🗺️ Region:</strong> {location_data.get('region', location_data.get('state', 'N/A'))}</p>
        <p><strong>🌎 Country:</strong> {location_data.get('country', 'N/A')}</p>
        """
        if 'latitude' in location_data and 'longitude' in location_data:
            html += f"""
            <p><strong>📌 Coordinates:</strong> {location_data['latitude']}, {location_data['longitude']}</p>
            <p><a href="https://maps.google.com/?q={location_data['latitude']},{location_data['longitude']}">View on Google Maps</a></p>
            """
       
        return (text, html)

    def send_email(self, sender_email, sender_password, recipient: List[str]):
        location = self.get_location()
        text_body, html_body = self.format_message(location)
       
        msg = MIMEMultipart('alternative')
        msg['From'] = sender_email
        msg['To'] = recipient
        msg['Subject'] = "⚠️ EMERGENCY ALERT from AI Companion ⚠️"
       
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
       
        try:
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, recipient, msg.as_string())
            return True
        except Exception as e:
            print(f"Email sending failed: {e}")
            return False

# Initialize database tables and create admin user
with app.app_context():
    db.create_all()
    # Check if admin user exists, if not create one
    if not User.query.filter_by(username='admin').first():
        hashed_password = generate_password_hash('brv123')
        admin_user = User(username='admin', password=hashed_password)
        db.session.add(admin_user)
        db.session.commit()
        print("Created admin user with username: admin and password: password123")

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)