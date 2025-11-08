"""
Script to load Food-101 dataset from Kaggle.
This script downloads and prepares the dataset for training/evaluation.
"""
import os
import sys

def load_kaggle_dataset():
    """Load Food-101 dataset from Kaggle"""
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        print("=" * 60)
        print("Food-101 Dataset Loader")
        print("=" * 60)
        print("\nNote: You need Kaggle API credentials to download datasets.")
        print("Setup instructions:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("\n" + "=" * 60 + "\n")
        
        print("Downloading Food-101 dataset from Kaggle...")
        print("Dataset: kmader/food41")
        print("This may take a while on first run...\n")
        
        # Download the dataset
        dataset_path = kagglehub.dataset_download("kmader/food41")
        
        print("\n" + "=" * 60)
        print(f" Dataset downloaded successfully!")
        print(f"Location: {dataset_path}")
        print("=" * 60)
        
        # List contents
        print("\nDataset contents:")
        if os.path.exists(dataset_path):
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path) / (1024 * 1024)  # MB
                    print(f"  📄 {item} ({size:.2f} MB)")
        
        return dataset_path
        
    except ImportError:
        print("Error: kagglehub is not installed.")
        print("Please run: pip install kagglehub[pandas-datasets]")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Kaggle API credentials are set up")
        print("2. Check your internet connection")
        print("3. Verify the dataset name: kmader/food41")
        return None

if __name__ == '__main__':
    load_kaggle_dataset()

