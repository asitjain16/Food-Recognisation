from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import os
from food_model import FoodRecognitionModel
from calorie_data import CALORIE_DATABASE

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize the food recognition model
print("Initializing food recognition model...")
try:
    food_model = FoodRecognitionModel()
    print(" Model initialized successfully!")
except Exception as e:
    print(f" Error initializing model: {e}")
    food_model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if not food_model:
            return jsonify({
                'error': 'Food recognition model is not available. Please check server logs.',
                'is_food': False
            }), 500
        
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'is_food': False
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'is_food': False
            }), 400
        
        if file and allowed_file(file.filename):
            # Process the image
            try:
                image = Image.open(file.stream)
                # Verify image is valid
                image.verify()
                # Reopen after verify (verify() closes the image)
                image = Image.open(file.stream)
            except Exception as e:
                return jsonify({
                    'error': f'Invalid image file: {str(e)}',
                    'is_food': False
                }), 400
            
            # Get prediction from model
            prediction = food_model.predict(image)
            
            # Check if model returned an error (non-food image)
            if 'error' in prediction or not prediction.get('is_food', False):
                error_msg = prediction.get('error', 'Could not identify food in the image')
                return jsonify({
                    'error': error_msg,
                    'is_food': False,
                    'confidence': prediction.get('confidence', 0)
                }), 400
            
            # Get calorie information (try to match with database)
            calorie_info = get_calorie_info_smart(prediction['food_name'])
            
            # Combine results
            result = {
                'food_name': prediction['food_name'],
                'confidence': prediction['confidence'],
                'calories_per_100g': calorie_info['calories'],
                'description': calorie_info['description'],
                'nutritional_info': calorie_info['nutrition'],
                'is_food': True
            }
            
            return jsonify(result)
        
        return jsonify({
            'error': 'Invalid file type. Please upload JPG, PNG, or GIF images.',
            'is_food': False
        }), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Server error: {str(e)}',
            'is_food': False
        }), 500

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_calorie_info_smart(food_name):
    """Smart lookup for calorie information with fuzzy matching"""
    food_name_lower = food_name.lower().replace(' ', '_')
    
    # Direct match
    if food_name_lower in CALORIE_DATABASE:
        return CALORIE_DATABASE[food_name_lower]
    
    # Try to find partial matches
    for db_food in CALORIE_DATABASE.keys():
        # Check if any word matches
        food_words = food_name_lower.split('_')
        db_words = db_food.split('_')
        
        if any(word in db_food for word in food_words if len(word) > 3):
            return CALORIE_DATABASE[db_food]
        if any(word in food_name_lower for word in db_words if len(word) > 3):
            return CALORIE_DATABASE[db_food]
    
    # Category-based matching
    category_mapping = {
        'pizza': 'pizza',
        'burger': 'hamburger', 
        'hamburger': 'hamburger',
        'cake': 'chocolate_cake',
        'chocolate': 'chocolate_cake',
        'chocolate_cake': 'chocolate_cake',
        'salad': 'caesar_salad',
        'caesar': 'caesar_salad',
        'pasta': 'spaghetti_bolognese',
        'spaghetti': 'spaghetti_bolognese',
        'noodles': 'ramen',
        'ramen': 'ramen',
        'fries': 'french_fries',
        'french_fries': 'french_fries',
        'ice_cream': 'ice_cream',
        'ice cream': 'ice_cream',
        'steak': 'steak',
        'chicken': 'chicken_curry',
        'curry': 'chicken_curry',
        'chicken_curry': 'chicken_curry',
        'sushi': 'sushi',
        'tacos': 'tacos',
        'taco': 'tacos',
        'pancakes': 'pancakes',
        'pancake': 'pancakes',
        'waffles': 'waffles',
        'waffle': 'waffles',
        'apple_pie': 'apple pie',
        'cheesecake': 'cheesecake',
        'lasagna': 'lasagna',
        'pad_thai': 'pad thai'
    }
    
    for keyword, db_food in category_mapping.items():
        if keyword in food_name_lower:
            if db_food in CALORIE_DATABASE:
                return CALORIE_DATABASE[db_food]
    
    # Default fallback with estimated values
    return {
        'calories': 250,
        'description': f'Estimated nutritional information for {food_name}. Values are approximate.',
        'nutrition': {
            'protein': '15g',
            'carbs': '30g',
            'fat': '10g',
            'fiber': '3g'
        }
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

