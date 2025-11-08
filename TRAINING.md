# Training the Food-101 Model

This guide explains how to train the Food Recognition model on the Food-101 dataset for improved accuracy.

## Prerequisites

1. **Dataset**: The Food-101 dataset should be downloaded. If not, run:
   ```bash
   python load_kaggle_dataset.py
   ```

2. **Dependencies**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

## Training Configuration

The training script (`train_model.py`) uses the following default settings:

- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 10
- **Optimizer**: AdamW with weight decay
- **Scheduler**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Train/Val Split**: 80/20

You can modify these in `train_model.py` if needed.

## Running Training

### Step 1: Start Training

Simply run the training script:

```bash
python train_model.py
```

The script will:
1. Automatically find the Food-101 dataset
2. Load and preprocess the images
3. Split data into training and validation sets (80/20)
4. Train the model for the specified number of epochs
5. Save the best model based on validation accuracy

### Step 2: Monitor Training

During training, you'll see:
- **Training progress** with loss and accuracy per batch
- **Validation results** after each epoch
- **Best model saves** when validation accuracy improves
- **Learning rate adjustments** when validation loss plateaus

Example output:
```
Epoch 1/10
------------------------------------------------------------
Training: 100%|████████| 2500/2500 [10:30<00:00, loss=2.3456, acc=45.23%]
Validation: 100%|████████| 625/625 [02:15<00:00, loss=1.8765, acc=58.45%]
 Saved best model (val_acc: 58.45%)
Train Loss: 2.3456, Train Acc: 45.23%
Val Loss: 1.8765, Val Acc: 58.45%
```

### Step 3: Model Files

After training, the following files will be created in `./models/`:

- **`food101_best.pth`**: Best model based on validation accuracy
- **`food101_final.pth`**: Final model after all epochs
- **`training_history.json`**: Training history (losses and accuracies)

## Using Trained Model

The application (`app.py`) will automatically load the trained model if it exists:

1. **Automatic Loading**: When you start the app, it checks for `./models/food101_best.pth`
2. **If found**: Loads the trained model with better accuracy
3. **If not found**: Uses ImageNet pre-trained weights (lower accuracy)

### Manual Model Path

You can also specify a custom model path when initializing:

```python
from food_model import FoodRecognitionModel

model = FoodRecognitionModel(model_path='./models/food101_best.pth')
```

## Expected Results

### Before Training (ImageNet weights only):
- Validation Accuracy: ~10-15% (random for 101 classes)
- Food Recognition: Very inaccurate
- Confidence: Low

### After Training (Food-101 trained):
- Validation Accuracy: ~60-80% (depending on training time)
- Food Recognition: Much more accurate
- Confidence: Higher and more reliable

### Training Time Estimates

- **10 epochs** (recommended): ~2-4 hours on CPU, ~30-60 minutes on GPU
- **20 epochs**: ~4-8 hours on CPU, ~1-2 hours on GPU
- **50 epochs**: ~10-20 hours on CPU, ~2-4 hours on GPU

## Tips for Better Training

1. **More Epochs**: Increase `NUM_EPOCHS` for better accuracy (but watch for overfitting)
2. **Larger Batch Size**: If you have more GPU memory, increase `BATCH_SIZE`
3. **Learning Rate**: Adjust `LEARNING_RATE` if training is unstable
4. **Data Augmentation**: Already included (random crops, flips, rotations, color jitter)
5. **Early Stopping**: Currently saves best model - you can add early stopping if needed

## Troubleshooting

### Issue: "Dataset not found"
**Solution**: Make sure you've downloaded the dataset:
```bash
python load_kaggle_dataset.py
```

### Issue: "Out of memory"
**Solutions**:
- Reduce `BATCH_SIZE` (e.g., from 32 to 16)
- Reduce `NUM_WORKERS` (e.g., from 4 to 2)

### Issue: "Training is too slow"
**Solutions**:
- Use GPU if available (training will be much faster)
- Reduce number of epochs
- Reduce batch size if causing memory issues

### Issue: "Low validation accuracy"
**Solutions**:
- Train for more epochs
- Check if dataset is loading correctly
- Ensure class names match between training and inference

## Advanced: Custom Training

You can modify `train_model.py` to:
- Change optimizer (SGD, Adam, etc.)
- Add custom loss functions
- Implement different data augmentation
- Add learning rate warmup
- Implement gradient clipping
- Add tensorboard logging

## Next Steps

After training:
1. Test the model with `app.py`
2. Compare accuracy with untrained model
3. Fine-tune if needed
4. Deploy the trained model

The trained model will automatically be used when you run `app.py`!

