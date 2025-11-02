# 🍔 Food Recognition App

A modern web application that uses deep learning to recognize food items from images and provide detailed nutritional information including calorie estimates.

## ✨ Features

- **Real-time Food Recognition**: Upload food images through an intuitive web interface
- **AI-Powered Analysis**: Uses TensorFlow for accurate food classification
- **Calorie Estimation**: Get detailed calorie information for recognized foods
- **Nutritional Breakdown**: View protein, carbs, fat, and fiber content
- **Modern UI**: Responsive design with Bootstrap and Font Awesome icons
- **Drag & Drop**: Easy image upload with drag-and-drop functionality

## 🛠️ Tech Stack

- **Backend**: Python 3.11+ with Flask
- **AI/ML**: TensorFlow for deep learning
- **Frontend**: Bootstrap 5, HTML5, JavaScript
- **Image Processing**: Pillow (PIL)
- **Dataset**: Food-101 dataset via Kaggle Hub

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone or download this project**

2. **Run the setup script**:
   ```bash
   python setup.py
   ```
   This will:
   - Install all required dependencies
   - Download the Food-101 dataset
   - Create necessary directories

3. **Start the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

### Manual Installation (Alternative)

If you prefer manual setup:

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir static uploads models

# Run the app
python app.py
```

## 📱 How to Use

1. **Upload an Image**: 
   - Click the upload area or drag and drop a food image
   - Supported formats: JPG, PNG, GIF
   - Maximum file size: 16MB

2. **Get Results**:
   - The app will analyze your image using AI
   - View the recognized food name with confidence level
   - See calorie information per 100g
   - Check detailed nutritional breakdown

3. **Try More Images**:
   - Click "Try Another Image" to analyze more foods

## 🏗️ Project Structure

```
food-recognition-app/
├── app.py                 # Main Flask application
├── food_model.py          # TensorFlow model and prediction logic
├── calorie_data.py        # Nutritional database
├── requirements.txt       # Python dependencies
├── setup.py              # Automated setup script
├── templates/
│   └── index.html        # Main web interface
└── README.md             # This file
```

## 🧠 How It Works

1. **Image Upload**: Users upload food images through the web interface
2. **Preprocessing**: Images are resized and normalized for the model
3. **AI Recognition**: TensorFlow CNN model classifies the food item
4. **Data Lookup**: Nutritional information is retrieved from the calorie database
5. **Results Display**: Food name, confidence, calories, and nutrition facts are shown

## 🎯 Model Details

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 224x224 RGB images
- **Classes**: 101 food categories from Food-101 dataset
- **Framework**: TensorFlow/Keras

## 📊 Supported Foods

The app recognizes 101 different food categories including:
- Main dishes (pizza, hamburger, steak, etc.)
- Desserts (cheesecake, chocolate cake, ice cream, etc.)
- International cuisine (sushi, pad thai, ramen, etc.)
- Appetizers and sides (french fries, caesar salad, etc.)

## 🔧 Configuration

### Environment Variables

You can customize the app behavior with these environment variables:

- `FLASK_ENV`: Set to `development` for debug mode
- `FLASK_PORT`: Change the port (default: 5000)
- `MAX_CONTENT_LENGTH`: Maximum upload size in bytes

### Model Customization

To use your own trained model:

1. Replace the model creation in `food_model.py`
2. Update the class names list
3. Modify the preprocessing if needed

## 🚨 Troubleshooting

### Common Issues

**Dataset Download Fails**:
- Check your internet connection
- Ensure you have a Kaggle account (may be required)
- The app will use fallback food classes if download fails

**Memory Issues**:
- Reduce image size before upload
- Close other applications to free up RAM

**Slow Predictions**:
- Consider using a GPU-enabled TensorFlow installation
- Optimize the model architecture for your hardware

### Error Messages

- `No file uploaded`: Make sure to select an image file
- `Invalid file type`: Only JPG, PNG, and GIF are supported
- `File too large`: Reduce image size (max 16MB)

## 🤝 Contributing

Feel free to contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Food-101 dataset by Bossard, Guillaumin & Van Gool
- TensorFlow team for the deep learning framework
- Bootstrap team for the UI framework
- Kaggle for dataset hosting

---

**Happy Food Recognition! 🍕🥗🍰**