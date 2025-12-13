"""
Visualization Utilities for Model Training and Evaluation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import itertools


class TrainingVisualizer:
    """Visualize training history and results"""
    
    @staticmethod
    def plot_training_history(history, save_path='training_history.png'):
        """Plot training and validation accuracy/loss"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
        axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='s')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Training history saved to '{save_path}'")
        
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to '{save_path}'")
        
    @staticmethod
    def print_classification_report(y_true, y_pred, classes):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        report = classification_report(y_true, y_pred, target_names=classes)
        print(report)
        
    @staticmethod
    def plot_prediction_samples(model, generator, num_samples=9, save_path='prediction_samples.png'):
        """Visualize sample predictions"""
        # Get a batch of images
        images, labels = next(generator)
        predictions = model.predict(images[:num_samples])
        
        plt.figure(figsize=(15, 15))
        
        class_names = list(generator.class_indices.keys())
        
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            
            true_class_idx = np.argmax(labels[i])
            pred_class_idx = np.argmax(predictions[i])
            
            true_class = class_names[true_class_idx]
            pred_class = class_names[pred_class_idx]
            confidence = predictions[i][pred_class_idx] * 100
            
            color = 'green' if true_class == pred_class else 'red'
            
            plt.title(f'True: {true_class}\nPred: {pred_class} ({confidence:.1f}%)',
                     color=color, fontweight='bold')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Prediction samples saved to '{save_path}'")


if __name__ == "__main__":
    print("Visualization utilities loaded successfully!")
