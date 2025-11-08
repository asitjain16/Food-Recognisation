# Food Recognition App

A modern, AI-powered web application that recognizes food items from images and provides detailed nutritional information including calorie estimates. Built with PyTorch and integrated with the Food-101 dataset.

## Features

- **Real-time Food Recognition**: Upload food images through an intuitive web interface
- **AI-Powered Analysis**: Uses PyTorch with ResNet50 deep learning architecture for accurate food classification
- **Smart ImageNet Mapping**: Hybrid approach using ImageNet predictions for better accuracy even without training
- **Calorie Estimation**: Get detailed calorie information for 101+ food categories
- **Nutritional Breakdown**: View protein, carbs, fat, and fiber content
- **Non-Food Detection**: Automatically detects and rejects non-food images
- **Modern UI**: Responsive design with Bootstrap 5 and Font Awesome icons
- **Drag & Drop**: Easy image upload with drag-and-drop functionality
- **Training Support**: Complete training pipeline for Food-101 dataset

## Tech Stack

- **Backend**: Python 3.9+ with Flask
- **AI/ML**: PyTorch with ResNet50 architecture (deep CNN for image recognition)
- **Model Library**: Timm (PyTorch Image Models)
- **Frontend**: Bootstrap 5, HTML5, JavaScript
- **Image Processing**: Pillow (PIL), torchvision
- **Dataset**: Food-101 dataset via Kaggle Hub
- **Model Training**: PyTorch with DataLoader, transforms, and augmentation

## Dependencies and Requirements

### System Requirements

- Python 3.9 or higher
- pip package manager
- Minimum 8GB RAM (16GB recommended for training)
- (Optional) NVIDIA GPU with CUDA support for faster training and inference
- Internet connection for downloading the dataset

### Python Dependencies

The project requires the following Python packages (specified in `requirements.txt`):

- **Flask (2.3.3)**: Web framework for the application server
- **torch (>=1.9.0)**: PyTorch deep learning framework
- **torchvision (>=0.10.0)**: Computer vision utilities for PyTorch
- **Pillow (>=8.4.0)**: Image processing library
- **numpy (>=1.19.5)**: Numerical computing library
- **requests (>=2.26.0)**: HTTP library for API requests
- **opencv-python-headless (>=4.5.4)**: Computer vision library (headless version)
- **scipy (>=1.7.3)**: Scientific computing library
- **matplotlib (>=3.4.3)**: Plotting and visualization library
- **tqdm (>=4.62.3)**: Progress bar library
- **kagglehub[pandas-datasets] (>=0.2.0)**: Kaggle dataset downloader
- **timm (>=0.9.0)**: PyTorch Image Models library for pre-trained models

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/asitjain16/Food-Recognisation.git
   cd Food-Recognisation
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; import flask; print('Installation successful!')"
   ```

## Downloading the Food-101 Dataset

To train the model and achieve better accuracy, you need to download the Food-101 dataset. The dataset contains 101 food categories with 1000 images per category, totaling over 100,000 food images.

### Method 1: Using the Provided Script (Recommended)

The easiest way to download the dataset is using the included script. First, you need to set up Kaggle API credentials:

1. **Create a Kaggle account** if you don't have one at https://www.kaggle.com

2. **Generate API credentials**:
   - Go to https://www.kaggle.com/settings
   - Scroll down to the "API" section
   - Click "Create New API Token"
   - This will download a file named `kaggle.json`

3. **Set up the credentials**:
   - Create the `.kaggle` directory in your home folder if it doesn't exist:
     ```bash
     mkdir -p ~/.kaggle
     ```
   - Move the downloaded `kaggle.json` file to `~/.kaggle/kaggle.json`
   - Set the correct permissions:
     ```bash
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. **Install kagglehub** (if not already installed):
   ```bash
   pip install kagglehub[pandas-datasets]
   ```

5. **Download the dataset**:
   ```bash
   python load_kaggle_dataset.py
   ```

The script will download the Food-101 dataset (also known as food41 on Kaggle) to your local cache. The dataset is approximately 5GB in size, so the download may take some time depending on your internet connection. The dataset will be saved to a location like `~/.cache/kagglehub/datasets/kmader/food41/versions/5/`.

### Method 2: Manual Download

If you prefer to download the dataset manually:

1. Visit the Kaggle dataset page: https://www.kaggle.com/datasets/kmader/food41
2. Click the "Download" button (you'll need to accept the dataset rules if prompted)
3. Extract the downloaded zip file
4. Place the extracted folder in a location accessible to the training script

### Verifying the Dataset

After downloading, verify that the dataset structure is correct. The dataset should have an `images` folder containing 101 subdirectories, each named after a food category (e.g., `apple_pie`, `pizza`, `hamburger`, etc.). Each subdirectory should contain 1000 images for that food category.

You can check the dataset location by running:
```bash
python load_kaggle_dataset.py
```

The script will display the path where the dataset was downloaded.

### Using the Dataset for Training

Once the dataset is downloaded, the training script (`train_model.py`) will automatically detect it. The script looks for the dataset in common cache locations. If your dataset is in a different location, you can modify the `find_dataset_path()` function in `train_model.py` to point to your dataset location.

## Inference Instructions

Inference is the process of using the trained model to recognize food items in new images. This section covers how to run inference using the web application.

### Starting the Application

1. **Ensure dependencies are installed**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Flask server**:
   ```bash
   python app.py
   ```

3. **Verify the model loads**:
   You should see output like:
   ```
   Initializing food recognition model...
   Loading food recognition model...
   Food recognition model loaded successfully!
   Using CPU  # or "Using GPU: [GPU Name]" if GPU available
   Model initialized successfully!
   * Running on http://0.0.0.0:5001
   ```

4. **Open the web interface**:
   Navigate to `http://localhost:5001` in your web browser

### Using the Web Interface

1. **Upload an Image**: 
   - Click the upload area or drag and drop a food image
   - Supported formats: JPG, PNG, GIF, WEBP
   - Maximum file size: 16MB
   - The image will be automatically processed

2. **View Results**:
   The application displays:
   - **Food Name**: Recognized food item (e.g., "Pizza", "Hamburger")
   - **Confidence Level**: Prediction confidence percentage
   - **Calories per 100g**: Calorie information
   - **Nutritional Breakdown**: Protein, carbs, fat, and fiber content
   - **Description**: Additional information about the food item

3. **Try More Images**:
   - Click "Try Another Image" to analyze more foods
   - Upload different food images to test the model

### Inference via Python API

You can also use the model programmatically:

```python
from food_model import FoodRecognitionModel
from PIL import Image

# Initialize the model
model = FoodRecognitionModel(model_path='./models/food101_best.pth')

# Load and predict on an image
image = Image.open('path/to/your/food_image.jpg')
result = model.predict(image)

# Check if food was detected
if result.get('is_food', False):
    print(f"Food: {result['food_name']}")
    print(f"Confidence: {result['confidence']}%")
    if 'all_predictions' in result:
        print("Top predictions:")
        for name, conf in result['all_predictions']:
            print(f"  - {name}: {conf}%")
else:
    print(f"Error: {result.get('error', 'Unknown error')}")
```

### Model Loading Behavior

The application automatically handles model loading:

- **Trained Model Available**: If `./models/food101_best.pth` exists, it loads the trained model
- **No Trained Model**: Falls back to ImageNet pre-trained weights with Food-101 classifier
- **Custom Model Path**: Specify a custom path when initializing `FoodRecognitionModel`

### Performance Considerations

- **First Inference**: May be slower as the model loads into memory
- **Subsequent Inferences**: Faster as the model is already loaded
- **GPU Acceleration**: If available, inference is significantly faster on GPU
- **Batch Processing**: For multiple images, process them sequentially or modify the code for batch inference

### Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- WEBP

### Image Requirements

- **Minimum Size**: 32x32 pixels (recommended: 224x224 or larger)
- **Aspect Ratio**: Any (images are automatically resized and cropped)
- **Color Space**: RGB (converted automatically if needed)
- **File Size**: Maximum 16MB (configurable in `app.py`)

## Training Instructions

Training the model on the Food-101 dataset significantly improves accuracy. Follow these steps to train your own model.

### Prerequisites for Training

1. **Download the Food-101 dataset** (see "Downloading the Food-101 Dataset" section above)
2. **Ensure sufficient disk space** (at least 10GB free for dataset and models)
3. **GPU recommended** for faster training (CPU training takes 2-4 hours)

### Step-by-Step Training Process

1. **Verify dataset is downloaded**:
   ```bash
   python load_kaggle_dataset.py
   ```
   This will display the dataset path if it's already downloaded, or download it if missing.

2. **Review training configuration** (optional):
   Open `train_model.py` to adjust hyperparameters:
   - `BATCH_SIZE`: Number of images per batch (default: 32)
   - `LEARNING_RATE`: Initial learning rate (default: 0.001)
   - `NUM_EPOCHS`: Number of training epochs (default: 10)
   - `NUM_WORKERS`: DataLoader workers (default: 4)

3. **Start training**:
   ```bash
   python train_model.py
   ```

4. **Monitor training progress**:
   The script displays:
   - Training loss and accuracy per epoch
   - Validation loss and accuracy per epoch
   - Best model checkpoint saves automatically
   - Progress bars for each epoch

5. **Training output**:
   - Best model saved to: `./models/food101_best.pth`
   - Final model saved to: `./models/food101_final.pth`
   - Training history saved to: `./models/training_history.json`

### Training Configuration Details

- **Data Split**: 80% training, 20% validation
- **Data Augmentation**: Random resized crops, horizontal flips, rotations (15 degrees), color jitter
- **Optimizer**: AdamW with weight decay (0.01)
- **Learning Rate Schedule**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Loss Function**: CrossEntropyLoss
- **Model Checkpointing**: Saves best model based on validation accuracy

### Training Tips

- **GPU Training**: If you have a GPU, PyTorch will automatically use it. Verify with:
  ```python
  import torch
  print(torch.cuda.is_available())
  ```

- **Memory Issues**: If you encounter out-of-memory errors:
  - Reduce `BATCH_SIZE` in `train_model.py` (try 16 or 8)
  - Reduce `NUM_WORKERS` to 2 or 1

- **Resume Training**: To continue from a checkpoint, modify `train_model.py` to load existing weights

- **Monitor Overfitting**: Watch for large gap between training and validation accuracy

### After Training

Once training completes, the trained model will be automatically used when you run the application:
```bash
python app.py
```

The app looks for `./models/food101_best.pth` and loads it automatically if found.

## Project Structure

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

## File Descriptions

### Core Application Files

**app.py**
- Main Flask web application server
- Handles HTTP requests and routes
- Processes image uploads from the web interface
- Integrates the food recognition model with the calorie database
- Returns JSON responses with food predictions and nutritional information
- Runs on port 5001 by default

**food_model.py**
- Contains the `FoodRecognitionModel` class
- Implements ResNet50-based deep learning model for food classification
- Handles image preprocessing and normalization
- Manages dual model approach (ImageNet and Food-101 models)
- Implements smart mapping from ImageNet predictions to Food-101 classes
- Includes food vs non-food detection logic
- Supports both trained and untrained model modes

**calorie_data.py**
- Contains the nutritional database (CALORIE_DATABASE dictionary)
- Stores calorie information per 100g for various food items
- Includes nutritional breakdown (protein, carbs, fat, fiber)
- Provides food descriptions and metadata

**train_model.py**
- Training script for the Food-101 dataset
- Handles dataset loading and preprocessing
- Implements data augmentation (random crops, flips, rotations, color jitter)
- Manages train/validation split (80/20)
- Configures optimizer (AdamW) and learning rate scheduler
- Saves model checkpoints during training
- Logs training history and metrics

**load_kaggle_dataset.py**
- Script to download Food-101 dataset from Kaggle
- Handles Kaggle API authentication
- Downloads dataset to local cache
- Verifies dataset structure after download
- Provides troubleshooting information

**setup.py**
- Automated setup script for the project
- Installs required dependencies
- Creates necessary directories (static, uploads, models)
- Optionally downloads the dataset
- Provides setup verification

### Configuration and Documentation Files

**requirements.txt**
- Lists all Python package dependencies with version specifications
- Used by pip to install required packages

**templates/index.html**
- Frontend web interface
- HTML template with Bootstrap 5 styling
- JavaScript for image upload and drag-and-drop functionality
- Displays food recognition results and nutritional information

**TRAINING.md**
- Detailed guide for training the model
- Step-by-step training instructions
- Configuration options and hyperparameters

**IMPROVEMENTS.md**
- Documentation of improvements and features
- Bug fixes and enhancements log

**GIT_SETUP.md**
- Git and GitHub setup instructions
- Repository configuration guide

### Generated Files and Directories

**models/**
- Directory created after training
- Stores trained model checkpoints:
  - `food101_best.pth`: Best model based on validation accuracy
  - `food101_final.pth`: Final model after all epochs
  - `training_history.json`: Training metrics and history

## How It Works

### Architecture

1. **Image Upload**: Users upload food images through the web interface
2. **Preprocessing**: Images are resized to 224x224 and normalized
3. **Dual Model Approach**:
   - **ImageNet Model**: Uses pre-trained ResNet50 on ImageNet for accurate food detection
   - **Food-101 Model**: Uses ResNet50 with Food-101 classifier (101 classes)
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

### Deep Learning Model

We're using a ResNet50 architecture, which is a deep convolutional neural network (CNN) that's well-suited for image recognition tasks. ResNet50 (Residual Network with 50 layers) is a powerful architecture that uses residual connections to enable training of very deep networks, making it highly effective for complex image classification problems like food recognition. The model leverages transfer learning by using pre-trained ImageNet weights, which are then fine-tuned on the Food-101 dataset to achieve optimal performance for food classification.

## Accuracy & Performance

### Without Training (ImageNet Mapping)

- Uses ImageNet pre-trained model predictions
- Maps ImageNet classes to Food-101 classes
- Accuracy: Moderate (approximately 40-60% for common foods)
- Works immediately without training

### With Training (Food-101 Dataset)

- Trained on Food-101 dataset
- Validation Accuracy: approximately 60-80% (depending on training epochs)
- Much better accuracy for all 101 food classes
- Recommended for production use

## Supported Foods

The app recognizes 101 different food categories including:

- **Fast Food**: Pizza, Hamburger, Hot Dog, French Fries, Tacos
- **Desserts**: Cheesecake, Chocolate Cake, Ice Cream, Donuts, Waffles, Pancakes
- **International**: Sushi, Ramen, Pad Thai, Tacos, Paella, Pho
- **Meats**: Steak, Baby Back Ribs, Chicken Curry, Pork Chop
- **Salads**: Caesar Salad, Greek Salad, Caprese Salad
- **Pastas**: Spaghetti Bolognese, Lasagna, Ravioli, Macaroni and Cheese
- And many more

See `food_model.py` for the complete list of 101 food classes.

## Configuration

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

## Documentation

- **[TRAINING.md](TRAINING.md)**: Complete guide for training the model
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)**: Detailed list of improvements and features
- **[GIT_SETUP.md](GIT_SETUP.md)**: Git and GitHub setup instructions

## Troubleshooting

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
- Make sure you've set up `kaggle.json` in `~/.kaggle/` directory
- Verify your internet connection
- Check that kagglehub is installed: `pip install kagglehub[pandas-datasets]`
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

## Advanced Features

### ImageNet Mapping System

The app uses a sophisticated mapping system to translate ImageNet predictions to Food-101 classes:

1. **Direct Mappings**: 40+ direct ImageNet to Food-101 mappings
2. **Keyword Matching**: Flexible keyword-based matching for variations
3. **Partial Matching**: Finds matches even with partial class name overlap
4. **Confidence Weighting**: ImageNet predictions get 90% weight, Food-101 gets 10%


### Training Features

- Automatic train/validation split (80/20)
- Data augmentation (random crops, flips, rotations, color jitter)
- Learning rate scheduling (ReduceLROnPlateau)
- Best model checkpointing
- Training history logging

## Acknowledgments

- **Food-101 Dataset**: Bossard, Guillaumin & Van Gool - ETH Zurich
- **PyTorch Team**: For the deep learning framework
- **Timm Library**: For EfficientNet and model utilities
- **Kaggle**: For dataset hosting and Kaggle Hub
- **Bootstrap Team**: For the UI framework
- **Flask Team**: For the web framework
---
## Project By

**Asit Jain (M25DE1049)**  
**Avinash Singh (M25DE1024)**  
**Prashnat Kumar Mishra (M25DE1063)**
---
Built with PyTorch and Flask
