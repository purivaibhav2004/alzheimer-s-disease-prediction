
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from PIL import Image
import os
import torch
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import hashlib
import sqlite3
from werkzeug.utils import secure_filename
import io
import base64

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = os.path.join(r'C:\Users\vaibh\Downloads\Alz_predcition_main_final', 'alzheimer_efficientnet_model.pth')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Load ML Model
def load_model():
    try:
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=4)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}")
        return None

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image

def predict_alzheimer(image):
    if model is None:
        return None, "Model not loaded"
    
    try:
        with torch.no_grad():
            preprocessed = preprocess_image(image)
            output = model(preprocessed)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            _, predicted = torch.max(output, 1)
            
            labels = [
                'Middle Stage Alzheimer\'s Disease',
                'High Alzheimer\'s Disease', 
                'Normal',
                '2.	Middle Stage Alzheimer\'s Disease (MCI)'
            ]
            
            prediction = labels[predicted.item()]
            confidence = probabilities[predicted.item()].item() * 100
            
            return prediction, confidence
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('prediction'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = hash_password(request.form['password'])
        
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        try:
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                         (username, email, password))
            conn.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/prediction')
def prediction():
    if 'user_id' not in session:
        flash('Please login to access the prediction feature.', 'warning')
        return redirect(url_for('login'))
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return jsonify({'error': 'Please login first'}), 401
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Process image
            image = Image.open(file.stream).convert('RGB')
            
            # Convert image to base64 for display
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='JPEG')
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Make prediction
            prediction, confidence = predict_alzheimer(image)
            
            if prediction is None:
                return jsonify({'error': f'Prediction failed: {confidence}'}), 500
            
            return jsonify({
                'prediction': prediction,
                'confidence': round(confidence, 2),
                'image': f'data:image/jpeg;base64,{img_str}'
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle contact form submission
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        # In a real app, you'd save this to database or send email
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)