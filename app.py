import os
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from textblob import TextBlob
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

# --- 1. Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)

# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'feedback.db'))
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Security Config
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'super-secret-key-change-this')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
db = SQLAlchemy(app)
jwt = JWTManager(app)

# --- HUGGING FACE API CONFIG (Cloud Friendly) ---
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
# Get key from environment, or warn if missing
HF_API_KEY = os.environ.get('HF_API_KEY')

if not HF_API_KEY:
    print("WARNING: HF_API_KEY not found in environment variables. ML categorization will fail.")

def classify_with_api(text, labels):
    if not HF_API_KEY:
        print("No API Key provided.")
        return "Uncategorized", 0.0

    payload = {
        "inputs": text,
        "parameters": {"candidate_labels": labels}
    }
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        result = response.json()
        
        # The API returns data like: {'sequence': '...', 'labels': ['Quality', ...], 'scores': [0.9, ...]}
        if 'labels' in result and len(result['labels']) > 0:
            return result['labels'][0], result['scores'][0]
        else:
            print("Model API Error:", result)
            return "Uncategorized", 0.0
    except Exception as e:
        print(f"API Request Failed: {e}")
        return "Uncategorized", 0.0

# --- 2. Database Models ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {"id": self.id, "username": self.username}

class Store(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    area = db.Column(db.String(200), nullable=True) 
    feedbacks = db.relationship('Feedback', backref='store', lazy=True)
    
    def to_dict(self):
        return {"id": self.id, "name": self.name, "area": self.area}

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    platform = db.Column(db.String(100), nullable=False)
    text = db.Column(db.String(1000), nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    category = db.Column(db.String(100), nullable=True)
    sentiment = db.Column(db.String(50), nullable=True)
    sentiment_score = db.Column(db.Float, nullable=True)
    store_id = db.Column(db.Integer, db.ForeignKey('store.id'), nullable=True)
    status = db.Column(db.String(50), nullable=False, default='New')

    def to_dict(self):
        store_name = self.store.name if self.store else "Unknown Store"
        return {
            "id": self.id, "platform": self.platform, "text": self.text,
            "timestamp": self.timestamp.isoformat(), "category": self.category,
            "sentiment": self.sentiment, "sentiment_score": self.sentiment_score,
            "store_id": self.store_id, "store_name": store_name, "status": self.status
        }

# --- 3. API Endpoints ---

@app.route('/v1/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'msg': 'Missing username or password'}), 400
    
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'msg': 'User already exists'}), 400
    
    user = User(username=data['username'])
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()
    return jsonify({'msg': 'User created', 'user': user.to_dict()}), 201

@app.route('/v1/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    
    if user and user.check_password(data.get('password')):
        access_token = create_access_token(identity=user.id)
        return jsonify({'msg': 'Login successful', 'access_token': access_token, 'user': user.to_dict()}), 200
        
    return jsonify({'msg': 'Invalid credentials'}), 401

@app.route('/v1/feedback', methods=['POST'])
def add_feedback():
    data = request.get_json()
    if not data or 'text' not in data: return jsonify({"error": "Missing text"}), 400

    # Sentiment Analysis (Local TextBlob is fast & light)
    blob = TextBlob(data['text'])
    sentiment = "Neutral"
    if blob.sentiment.polarity > 0.2: sentiment = "Positive"
    elif blob.sentiment.polarity < -0.1: sentiment = "Negative"

    # Categorization (Via Hugging Face API to save RAM)
    candidate_labels = ["Quality of food", "Customer service", "Speed", "Ambience"]
    category, confidence = classify_with_api(data['text'], candidate_labels)

    new_feedback = Feedback(
        platform=data.get('platform', 'Unknown'),
        text=data['text'],
        category=category,
        sentiment=sentiment,
        sentiment_score=blob.sentiment.polarity,
        store_id=data.get('store_id'),
        status='New'
    )
    db.session.add(new_feedback)
    db.session.commit()
    
    socketio.emit('new_feedback', new_feedback.to_dict())
    return jsonify({"msg": "Feedback added", "feedback": new_feedback.to_dict()}), 201

@app.route('/v1/feedback', methods=['GET'])
def list_feedback():
    start = request.args.get('start')
    end = request.args.get('end')
    store_id = request.args.get('store_id')
    status = request.args.get('status')
    area = request.args.get('area')
    
    query = Feedback.query
    if area: query = query.join(Store).filter(Store.area == area)

    if start:
        try:
            s_date = datetime.strptime(start, '%Y-%m-%d').date()
            query = query.filter(Feedback.timestamp >= s_date)
        except: pass
    
    if end:
        try:
            e_date = datetime.strptime(end, '%Y-%m-%d').date() + timedelta(days=1)
            query = query.filter(Feedback.timestamp < e_date)
        except: pass

    if store_id: query = query.filter(Feedback.store_id == store_id)
    if status: query = query.filter(Feedback.status == status)

    per_page = int(request.args.get('per_page', 100))
    all_feedback = query.order_by(Feedback.timestamp.desc()).limit(per_page).all()
    
    return jsonify({"feedback": [f.to_dict() for f in all_feedback]})

@app.route('/v1/feedback/<int:id>/resolve', methods=['POST'])
def resolve(id):
    fb = Feedback.query.get(id)
    if not fb: return jsonify({"error": "Not found"}), 404
    fb.status = 'Resolved'
    db.session.commit()
    socketio.emit('feedback_resolved', fb.to_dict())
    return jsonify({"msg": "Resolved"})

@app.route('/v1/metrics', methods=['GET'])
def metrics():
    total = Feedback.query.count()
    cat_query = Feedback.query.group_by(Feedback.category).with_entities(
        Feedback.category, func.count(Feedback.id), func.avg(Feedback.sentiment_score)).all()
    
    cats = {}
    for c, count, avg in cat_query:
        cats[c] = {"count": count, "average_sentiment_score": avg}
        
    return jsonify({"total_feedback": total, "feedback_by_category": cats})

@app.route('/v1/metrics/trend', methods=['GET'])
def trend():
    daily = Feedback.query.with_entities(
        func.strftime('%Y-%m-%d', Feedback.timestamp).label('date'),
        func.avg(Feedback.sentiment_score).label('avg_sentiment')
    ).group_by('date').order_by('date').all()
    
    return jsonify([{"date": r.date, "average_sentiment": r.avg_sentiment} for r in daily])

@app.route('/v1/stores', methods=['GET', 'POST'])
def stores():
    if request.method == 'POST':
        data = request.get_json()
        new_store = Store(name=data['name'], area=data.get('area'))
        db.session.add(new_store)
        db.session.commit()
        return jsonify({"message": "Store added", "id": new_store.id}), 201
    
    area = request.args.get('area')
    query = Store.query
    if area: query = query.filter(Store.area == area)
    return jsonify([s.to_dict() for s in query.all()])

@app.route('/v1/areas', methods=['GET'])
def areas():
    query = db.session.query(Store.area).distinct().filter(Store.area != None)
    return jsonify([r[0] for r in query])

# --- AUTOMATIC DB INIT ---
with app.app_context():
    db.create_all()

@app.route('/')
def hello():
    return "Hello! Your feedback server is running."

# --- MAGIC ROUTE TO FORCE DATABASE CREATION ---
@app.route('/force-init-db', methods=['GET'])
def force_init_db():
    try:
        with app.app_context():
            db.create_all()
        return "Database tables created successfully! You can now Sign Up."
    except Exception as e:
        return f"Error creating database: {str(e)}"
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)