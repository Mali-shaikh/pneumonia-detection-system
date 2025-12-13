"""
Data Preprocessing Utilities for Pneumonia Detection
"""
import os
import shutil
from pathlib import Path
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetDownloader:
    """Download and organize Kaggle dataset"""
    
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = Path(dataset_path)
        
    def download_dataset(self):
        """Download dataset from Kaggle"""
        import kaggle
        
        print("📥 Downloading Chest X-Ray Pneumonia dataset from Kaggle...")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path=self.dataset_path,
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        
    def organize_dataset(self):
        """Organize dataset structure"""
        # The downloaded dataset has structure: chest_xray/train, chest_xray/test, chest_xray/val
        chest_xray_path = self.dataset_path / 'chest_xray'
        
        if chest_xray_path.exists():
            # Check if already organized
            if (self.dataset_path / 'train').exists():
                print("✅ Dataset already organized!")
                return
                
            # Move contents to parent directory
            for item in chest_xray_path.iterdir():
                dest = self.dataset_path / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            
            # Remove empty chest_xray directory
            try:
                chest_xray_path.rmdir()
            except:
                pass
            
        print("✅ Dataset organized successfully!")
        
    def get_dataset_info(self):
        """Print dataset statistics"""
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if split_path.exists():
                normal_count = len(list((split_path / 'NORMAL').glob('*.jpeg')))
                pneumonia_count = len(list((split_path / 'PNEUMONIA').glob('*.jpeg')))
                
                print(f"\n{split.upper()} SET:")
                print(f"  Normal: {normal_count}")
                print(f"  Pneumonia: {pneumonia_count}")
                print(f"  Total: {normal_count + pneumonia_count}")


class DataPreprocessor:
    """Preprocess images for model training"""
    
    def __init__(self, img_height=224, img_width=224, batch_size=32):
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        
    def create_data_generators(self, dataset_path='dataset'):
        """Create train, validation, and test data generators"""
        
        # Data augmentation for training set
        # Note: MobileNetV2 expects inputs in [-1, 1] range, but its preprocess_input method
        # handles the scaling. We pass [0, 255] images to the model.
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No rescaling for validation/test either
        val_test_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            f'{dataset_path}/train',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            f'{dataset_path}/val',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            f'{dataset_path}/test',
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        print("\n✅ Data generators created successfully!")
        print(f"Class indices: {train_generator.class_indices}")
        
        return train_generator, val_generator, test_generator
    
    def visualize_samples(self, generator, num_samples=9):
        """Visualize sample images from generator"""
        plt.figure(figsize=(12, 12))
        
        # Get a batch of images
        images, labels = next(generator)
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            
            # Get class name
            class_idx = np.argmax(labels[i])
            class_name = list(generator.class_indices.keys())[class_idx]
            
            plt.title(f'Class: {class_name}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        print("✅ Sample images saved to 'dataset_samples.png'")


if __name__ == "__main__":
    # Download and prepare dataset
    downloader = DatasetDownloader()
    downloader.download_dataset()
    downloader.organize_dataset()
    downloader.get_dataset_info()
    
    # Create data generators
    preprocessor = DataPreprocessor()
    train_gen, val_gen, test_gen = preprocessor.create_data_generators()
    
    # Visualize samples
    preprocessor.visualize_samples(train_gen)
