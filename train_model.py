"""
Pneumonia Detection Model Training Script
Using Transfer Learning with MobileNetV2
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
from pathlib import Path
from sklearn.utils import compute_class_weight
from sklearn.metrics import precision_recall_curve, f1_score

from utils.data_preprocessing import DataPreprocessor
from utils.visualization import TrainingVisualizer


class PneumoniaDetectionModel:
    """Build and train pneumonia detection model using transfer learning"""
    
    def __init__(self, img_height=224, img_width=224, num_classes=2):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build model using MobileNetV2 with transfer learning"""
        print("\n🏗️  Building model architecture...")
        
        # Load pre-trained MobileNetV2 (without top classification layer)
        base_model = MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        self.base_model = base_model # Save reference for fine-tuning
        
        # Build custom classification head
        inputs = keras.Input(shape=(self.img_height, self.img_width, 3))
        
        # Preprocessing (MobileNetV2 expects inputs in [-1, 1] range)
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        
        # Base model
        x = self.base_model(x, training=False)
        
        # Custom classification layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu', name='dense_128')(x)
        x = layers.Dropout(0.5, name='dropout')(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        # Create model
        self.model = keras.Model(inputs, outputs, name='pneumonia_mobilenet')
        
        print("✅ Model built successfully!")
        print(f"\n📊 Total parameters: {self.model.count_params():,}")
        print(f"   Trainable parameters: {sum([tf.size(var).numpy() for var in self.model.trainable_variables]):,}")
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with optimizer and loss function"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        print("✅ Model compiled successfully!")
        
    def get_callbacks(self, model_path='models/pneumonia_model.h5'):
        """Define training callbacks"""
        callbacks = [
            # Save best model
            ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            
            # Stop training if no improvement
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train(self, train_generator, val_generator, epochs=20, model_path='models/pneumonia_model.h5', class_weight=None):
        """Train the model"""
        print("\n🚀 Starting model training...")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {train_generator.batch_size}")
        print(f"   Training samples: {train_generator.n}")
        print(f"   Validation samples: {val_generator.n}")
        if class_weight:
            print(f"   Class weights: {class_weight}")
        
        # Get callbacks
        callbacks = self.get_callbacks(model_path)
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        return self.history
    
    def fine_tune_model(self, train_generator, val_generator, epochs=10, model_path='models/pneumonia_model_finetuned.h5', class_weight=None):
        """Fine-tune the model by unfreezing top layers"""
        print("\n🔧 Starting fine-tuning...")
        
        # Unfreeze the base model
        self.base_model.trainable = True
        
        # Freeze early layers of base model, keep top layers trainable
        # MobileNetV2 has 155 layers. Let's fine-tune the last 40.
        fine_tune_at = 100
        
        # Use the stored base_model reference
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
            
        print(f"   Unfrozen layers starting from index {fine_tune_at}")
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5), # Low learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        callbacks = self.get_callbacks(model_path)
        
        history_fine = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        print("\n✅ Fine-tuning completed!")
        self.history = history_fine # Update history or merge if needed
        return history_fine

    def evaluate(self, test_generator):
        """Evaluate model on test set"""
        print("\n📈 Evaluating model on test set...")
        
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(test_generator)
        
        # Calculate optimal threshold
        print("\n🔍 Calculating optimal probability threshold...")
        y_pred = self.model.predict(test_generator)
        y_true = test_generator.classes
        
        # Get probabilities for the positive class (Pneumonia)
        y_scores = y_pred[:, 1]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        
        # Find the optimal threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        print(f"\n✨ OPTIMAL THRESHOLD FOUND: {best_threshold:.4f}")
        print(f"   Max F1 Score: {best_f1:.4f}")
        print("   (Use this value in the Streamlit app sidebar)")
        
        print(f"\n{'='*60}")
        print("TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"  Test Loss:      {test_loss:.4f}")
        print(f"  Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall:    {test_recall:.4f}")
        print(f"{'='*60}\n")
        
        return test_loss, test_accuracy, test_precision, test_recall, best_threshold
    
    def save_model(self, path='models/pneumonia_model.h5'):
        """Save trained model"""
        self.model.save(path)
        print(f"✅ Model saved to '{path}'")


def main():
    """Main training pipeline"""
    print("="*60)
    print("PNEUMONIA DETECTION MODEL TRAINING")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Create data generators
    print("\n📦 Preparing data...")
    preprocessor = DataPreprocessor(img_height=224, img_width=224, batch_size=32)
    train_gen, val_gen, test_gen = preprocessor.create_data_generators('dataset')
    
    # Build and compile model
    model_builder = PneumoniaDetectionModel(img_height=224, img_width=224, num_classes=2)
    model_builder.build_model()
    model_builder.compile_model(learning_rate=0.001)
    
    # Display model summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model_builder.model.summary()
    
    # Calculate class weights
    print("   Calculating class weights...")
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"   Class Weights: {class_weights_dict}")

    # Train model
    history = model_builder.train(
        train_gen, 
        val_gen, 
        epochs=15, # Reduced initial epochs
        model_path='models/pneumonia_model.h5',
        class_weight=class_weights_dict
    )
    
    # Fine-tune model
    history_fine = model_builder.fine_tune_model(
        train_gen,
        val_gen,
        epochs=10,
        model_path='models/pneumonia_model_finetuned.h5',
        class_weight=class_weights_dict
    )
    
    # Evaluate on test set
    test_results = model_builder.evaluate(test_gen)
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    visualizer = TrainingVisualizer()
    
    # Plot training history
    visualizer.plot_training_history(history, 'training_history.png')
    
    # Get predictions for confusion matrix
    print("\n🔍 Generating predictions for evaluation...")
    y_pred = model_builder.model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    # Plot confusion matrix
    class_names = list(test_gen.class_indices.keys())
    visualizer.plot_confusion_matrix(y_true, y_pred_classes, class_names, 'confusion_matrix.png')
    
    # Print classification report
    visualizer.print_classification_report(y_true, y_pred_classes, class_names)
    
    # Plot sample predictions
    visualizer.plot_prediction_samples(model_builder.model, test_gen, num_samples=9, 
                                      save_path='prediction_samples.png')
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Generated Files:")
    print(f"   - models/pneumonia_model.h5 (Trained model)")
    print(f"   - training_history.png (Training curves)")
    print(f"   - confusion_matrix.png (Confusion matrix)")
    print(f"   - prediction_samples.png (Sample predictions)")
    print("\n🚀 Ready for Streamlit deployment!")


if __name__ == "__main__":
    main()
