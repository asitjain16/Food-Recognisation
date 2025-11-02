# 🍔 Food Recognition App

A modern, AI-powered web application that recognizes food items from images and provides detailed nutritional information including calorie estimates. Built with PyTorch and integrated with the Food-101 dataset.

## ✨ Features

- **Real-time Food Recognition**: Upload food images through an intuitive web interface
- **AI-Powered Analysis**: Uses PyTorch with EfficientNet-B0 for accurate food classification
- **Smart ImageNet Mapping**: Hybrid approach using ImageNet predictions for better accuracy even without training
- **Calorie Estimation**: Get detailed calorie information for 101+ food categories
- **Nutritional Breakdown**: View protein, carbs, fat, and fiber content
- **Non-Food Detection**: Automatically detects and rejects non-food images
- **Modern UI**: Responsive design with Bootstrap 5 and Font Awesome icons
- **Drag & Drop**: Easy image upload with drag-and-drop functionality
- **Training Support**: Complete training pipeline for Food-101 dataset

## 🛠️ Tech Stack

- **Backend**: Python 3.9+ with Flask
- **AI/ML**: PyTorch with EfficientNet-B0 architecture
- **Model Library**: Timm (PyTorch Image Models)
- **Frontend**: Bootstrap 5, HTML5, JavaScript
- **Image Processing**: Pillow (PIL), torchvision
- **Dataset**: Food-101 dataset via Kaggle Hub
- **Model Training**: PyTorch with DataLoader, transforms, and augmentation

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) GPU support for faster training and inference

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/asitjain16/Food-Recognisation.git
   cd Food-Recognisation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**:
   ```bash
   python app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:5001
   ```
   Note: Port 5001 is used to avoid conflicts with macOS AirPlay Receiver on port 5000.

## 📱 How to Use

### Basic Usage

1. **Upload an Image**: 
   - Click the upload area or drag and drop a food image
   - Supported formats: JPG, PNG, GIF, WEBP
   - Maximum file size: 16MB

2. **Get Results**:
   - The app will analyze your image using AI
   - View the recognized food name with confidence level
   - See calorie information per 100g
   - Check detailed nutritional breakdown

3. **Try More Images**:
   - Click "Try Another Image" to analyze more foods

### Training the Model (Optional but Recommended)

For improved accuracy, train the model on the Food-101 dataset:

1. **Download the dataset** (if not already downloaded):
   ```bash
   python load_kaggle_dataset.py
   ```
   Note: Requires Kaggle API credentials. See [TRAINING.md](TRAINING.md) for details.

2. **Train the model**:
   ```bash
   python train_model.py
   ```
   This will train the model for 10 epochs (takes ~2-4 hours on CPU, ~30-60 min on GPU).

3. **Restart the app** - The trained model will be automatically loaded:
   ```bash
   python app.py
   ```

## 🏗️ Project Structure

```
Food-Recognisation/
├── app.py                      # Main Flask application
├── food_model.py               # PyTorch model with ImageNet mapping
├── train_model.py              # Training script for Food-101 dataset
├── calorie_data.py             # Nutritional database
├── load_kaggle_dataset.py      # Dataset downloader
├── requirements.txt            # Python dependencies
├── setup.py                    # Setup script
├── templates/
│   └── index.html              # Web interface
├── models/                     # Trained models (created after training)
│   ├── food101_best.pth        # Best model checkpoint
│   └── food101_final.pth       # Final model checkpoint
├── README.md                   # This file
├── TRAINING.md                 # Training guide
├── IMPROVEMENTS.md             # Improvement documentation
└── .gitignore                 # Git ignore rules
```

## 🧠 How It Works

### Architecture

1. **Image Upload**: Users upload food images through the web interface
2. **Preprocessing**: Images are resized to 224x224 and normalized
3. **Dual Model Approach**:
   - **ImageNet Model**: Uses pre-trained EfficientNet-B0 on ImageNet for accurate food detection
   - **Food-101 Model**: Uses EfficientNet-B0 with Food-101 classifier (101 classes)
4. **Smart Mapping**: Maps ImageNet predictions to Food-101 classes using:
   - Direct class mappings
   - Keyword-based matching
   - Partial string matching
   - Confidence-weighted scoring
5. **Food Detection**: Validates that the image contains food (rejects non-food images)
6. **Data Lookup**: Nutritional information retrieved from calorie database
7. **Results Display**: Food name, confidence, calories, and nutrition facts

### Model Details

- **Architecture**: EfficientNet-B0 (PyTorch)
- **Backbone**: ImageNet pre-trained weights
- **Classifier**: Custom Food-101 classifier (101 classes)
- **Input Size**: 224x224 RGB images
- **Classes**: 101 food categories from Food-101 dataset
- **Framework**: PyTorch with Timm library

## 🎯 Accuracy & Performance

### Without Training (ImageNet Mapping)
- Uses ImageNet pre-trained model predictions
- Maps ImageNet classes to Food-101 classes
- Accuracy: Moderate (~40-60% for common foods)
- Works immediately without training

### With Training (Food-101 Dataset)
- Trained on Food-101 dataset
- Validation Accuracy: ~60-80% (depending on training epochs)
- Much better accuracy for all 101 food classes
- Recommended for production use

## 📊 Supported Foods

The app recognizes **101 different food categories** including:

- **Fast Food**: Pizza, Hamburger, Hot Dog, French Fries, Tacos
- **Desserts**: Cheesecake, Chocolate Cake, Ice Cream, Donuts, Waffles, Pancakes
- **International**: Sushi, Ramen, Pad Thai, Tacos, Paella, Pho
- **Meats**: Steak, Baby Back Ribs, Chicken Curry, Pork Chop
- **Salads**: Caesar Salad, Greek Salad, Caprese Salad
- **Pastas**: Spaghetti Bolognese, Lasagna, Ravioli, Macaroni and Cheese
- **And many more!**

See `food_model.py` for the complete list of 101 food classes.

## 🔧 Configuration

### Environment Variables

You can customize the app behavior:

- `FLASK_ENV`: Set to `development` for debug mode (default: development)
- `FLASK_PORT`: Change the port (default: 5001)
- `MAX_CONTENT_LENGTH`: Maximum upload size in bytes (default: 16MB)

### Model Path

The app automatically looks for trained models in `./models/food101_best.pth`. To use a custom model:

```python
from food_model import FoodRecognitionModel

model = FoodRecognitionModel(model_path='./path/to/your/model.pth')
```

## 📚 Documentation

- **[TRAINING.md](TRAINING.md)**: Complete guide for training the model
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)**: Detailed list of improvements and features
- **[GIT_SETUP.md](GIT_SETUP.md)**: Git and GitHub setup instructions

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use**:
- Port 5000 is often used by macOS AirPlay Receiver
- The app uses port 5001 by default
- Change port in `app.py` if needed: `app.run(port=YOUR_PORT)`

**Model Loading Fails**:
- Ensure PyTorch and dependencies are installed: `pip install -r requirements.txt`
- Check that models directory exists: `mkdir -p models`

**Low Accuracy**:
- Train the model on Food-101 dataset for better accuracy
- See [TRAINING.md](TRAINING.md) for training instructions

**Dataset Download Fails**:
- Requires Kaggle API credentials
- See `load_kaggle_dataset.py` for setup instructions
- App will work with ImageNet mapping even without dataset

**Memory Issues**:
- Reduce batch size in `train_model.py` (default: 32)
- Use smaller image sizes
- Consider using a GPU for training

### Error Messages

- `No file uploaded`: Make sure to select an image file
- `Invalid file type`: Only JPG, PNG, GIF, and WEBP are supported
- `File too large`: Reduce image size (max 16MB)
- `This image does not appear to contain food`: Upload a food image (non-food images are rejected)

## 🔬 Advanced Features

### ImageNet Mapping System

The app uses a sophisticated mapping system to translate ImageNet predictions to Food-101 classes:

1. **Direct Mappings**: 40+ direct ImageNet → Food-101 mappings
2. **Keyword Matching**: Flexible keyword-based matching for variations
3. **Partial Matching**: Finds matches even with partial class name overlap
4. **Confidence Weighting**: ImageNet predictions get 90% weight, Food-101 gets 10%

### Non-Food Detection

The app intelligently detects non-food images:
- Checks ImageNet top predictions for food-related indicators
- Rejects images with strong non-food signals (people, animals, objects)
- Provides clear error messages for rejected images

### Training Features

- Automatic train/validation split (80/20)
- Data augmentation (random crops, flips, rotations, color jitter)
- Learning rate scheduling (ReduceLROnPlateau)
- Best model checkpointing
- Training history logging

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Food-101 Dataset**: Bossard, Guillaumin & Van Gool - ETH Zurich
- **PyTorch Team**: For the deep learning framework
- **Timm Library**: For EfficientNet and model utilities
- **Kaggle**: For dataset hosting and Kaggle Hub
- **Bootstrap Team**: For the UI framework
- **Flask Team**: For the web framework

## 📞 Support

For issues, questions, or contributions:
- Open an issue on [GitHub](https://github.com/asitjain16/Food-Recognisation/issues)
- Check the documentation files (TRAINING.md, IMPROVEMENTS.md)

---

**Happy Food Recognition! 🍕🥗🍰**

Built with ❤️ using PyTorch and Flask
