"""
Training script for Food-101 Recognition Model
Trains the EfficientNet-B0 model on Food-101 dataset for improved accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import timm
import numpy as np
import os
from tqdm import tqdm
from food_model import FoodRecognitionModel
import json
from pathlib import Path

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_dataset_path():
    """Find the Food-101 dataset path"""
    # Check common kagglehub cache locations
    possible_paths = [
        '/Users/asitjain/.cache/kagglehub/datasets/kmader/food41/versions/5',
        os.path.expanduser('~/.cache/kagglehub/datasets/kmader/food41/versions/5'),
        './data/food41',
    ]
    
    for path in possible_paths:
        images_path = os.path.join(path, 'images')
        if os.path.exists(images_path):
            print(f"✅ Found dataset at: {path}")
            return path
    
    # Try to download if not found
    try:
        import kagglehub
        print("Dataset not found locally. Downloading...")
        dataset_path = kagglehub.dataset_download("kmader/food41")
        return dataset_path
    except Exception as e:
        print(f"❌ Could not find or download dataset: {e}")
        return None

def load_dataset(data_dir):
    """Load Food-101 dataset with train/validation split"""
    images_dir = os.path.join(data_dir, 'images')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(images_dir, transform=train_transform)
    
    # Get class names
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} food classes")
    
    # Split dataset into train and validation (80/20)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    
    # Create validation dataset with different transform
    val_dataset = datasets.ImageFolder(images_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, class_names

def create_model(num_classes, device):
    """Create and initialize the model"""
    print("Creating model...")
    
    # Load EfficientNet-B0 with ImageNet weights
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1000)
    
    # Get feature dimension
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
        else:
            in_features = model.classifier.in_features
    else:
        in_features = 1280
    
    # Replace classifier with Food-101 classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    
    model = model.to(device)
    return model

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def main():
    """Main training function"""
    print("=" * 60)
    print("Food-101 Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    # Find dataset
    dataset_path = find_dataset_path()
    if dataset_path is None:
        print("❌ Could not find dataset. Please download it first using load_kaggle_dataset.py")
        return
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        train_loader, val_loader, class_names = load_dataset(dataset_path)
        print(f"✅ Dataset loaded: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create model
    num_classes = len(class_names)
    model = create_model(num_classes, DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = './models/food101_best.pth'
    os.makedirs('./models', exist_ok=True)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names
            }, best_model_path)
            print(f"✅ Saved best model (val_acc: {val_acc:.2f}%)")
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Save final model
    final_model_path = './models/food101_final.pth'
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'class_names': class_names,
        'history': history
    }, final_model_path)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    print("=" * 60)
    
    # Save training history
    history_path = './models/training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

if __name__ == '__main__':
    main()

