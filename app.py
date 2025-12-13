"""
🫁 Pneumonia Detection System - Streamlit Application
AI-powered chest X-ray analysis using Deep Learning
"""
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .pneumonia-result {
        background-color: #fee;
        border: 2px solid #f88;
    }
    .normal-result {
        background-color: #efe;
        border: 2px solid #8f8;
    }
    .confidence-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border: none;
        border-radius: 10px;
    }
    .disclaimer {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    model_path = 'models/pneumonia_model.h5'
    
    if not os.path.exists(model_path):
        st.error("❌ Model file not found! Please train the model first.")
        st.stop()
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Note: We do NOT normalize here because the model has a 
    # built-in preprocessing layer that handles [0, 255] inputs.
    
    return img_array


def make_prediction(model, image):
    """Make prediction on preprocessed image"""
    # Preprocess image
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    
    return prediction[0]


def plot_confidence_chart(probabilities, class_names):
    """Create a confidence bar chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['#ff6b6b' if class_names[i] == 'PNEUMONIA' else '#51cf66' 
              for i in range(len(class_names))]
    
    bars = ax.barh(class_names, probabilities * 100, color=colors, alpha=0.7)
    
    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.2f}%', 
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(axis='x', alpha=0.3)
    
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generate Grad-CAM heatmap using manual layer iteration"""
    # 1. Identify layers
    base_model = None
    classifier_layers = []
    found_base = False
    
    # Identify Base Model and Classifier Layers
    # We skip pre-processing layers (TFOpLambda) because they cause TypeError when called manually
    # We will handle preprocessing ourselves
    
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            base_model = layer
            found_base = True
            continue
        
        if found_base:
            classifier_layers.append(layer)
            
    if base_model is None:
        raise ValueError("Could not find MobileNetV2 base layer in model.")

    # 2. Run Forward Pass with GradientTape
    with tf.GradientTape() as tape:
        # A. Manual Preprocessing
        # MobileNetV2 expects [-1, 1], but our wrapper model expects [0, 255]
        # Since we are skipping the wrapper's preprocessing layers, we must do it here.
        x_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(img_array, tf.float32))
        
        # B. Run Base Model (MobileNetV2)
        features = base_model(x_preprocessed)
        tape.watch(features)
        
        # C. Run Classifier Layers
        curr_x = features
        for layer in classifier_layers:
            curr_x = layer(curr_x)
        
        preds = curr_x
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Compute Gradients
    grads = tape.gradient(class_channel, features)
    
    if grads is None:
        raise ValueError("Gradient calculation failed (grads is None).")

    # 4. Global Average Pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Create Heatmap
    features = features[0]
    heatmap = features @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 6. Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumonia Detection System</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Chest X-Ray Analysis using Deep Learning</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About the Model")
        st.write("""
        **Architecture:** MobileNetV2 (Transfer Learning)
        
        **Training Dataset:** Chest X-Ray Images (Pneumonia)
        - Source: Kaggle
        - Classes: NORMAL, PNEUMONIA
        - Images: 5,800+
        
        **Model Performance:**
        - Accuracy: 90%+
        - Input Size: 224x224 pixels
        - Framework: TensorFlow/Keras
        """)
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        confidence_threshold = st.slider(
            "Prediction Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.75, 
            step=0.05,
            help="Adjust how strict the model is. Higher values reduce false positives (Normal classified as Pneumonia)."
        )
        
        st.markdown("---")
        
        st.header("ℹ️ How to Use")
        st.write("""
        1. Upload a chest X-ray image
        2. Adjust threshold if needed
        3. Click 'Detect Pneumonia'
        4. View the prediction results
        """)
        
        st.markdown("---")
        
        st.header("👨‍💻 Developer Info")
        st.write("""
        **Project:** Pneumonia Detection System
        
        **Technology Stack:**
        - Python
        - TensorFlow
        - Streamlit
        - Transfer Learning (CNNs)
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Chest X-Ray")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a chest X-ray image in JPG or PNG format"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded X-Ray Image', use_column_width=True)
            
            # Clear button
            if st.button("🗑️ Clear Image"):
                st.rerun()
    
    with col2:
        st.header("🔬 Prediction Results")
        
        if uploaded_file is not None:
            # Prediction button
            if st.button("🚀 Detect Pneumonia", type="primary"):
                with st.spinner("🔍 Analyzing X-ray image..."):
                    # Load model
                    model = load_model()
                    
                    # Make prediction
                    probabilities = make_prediction(model, image)
                    
                    # Get class names (assuming NORMAL is index 0, PNEUMONIA is index 1)
                    class_names = ['NORMAL', 'PNEUMONIA']
                    
                    # Probability of Pneumonia
                    pneumonia_prob = probabilities[1]
                    
                    if pneumonia_prob > confidence_threshold:
                        predicted_class = "PNEUMONIA"
                        confidence = pneumonia_prob * 100
                    else:
                        predicted_class = "NORMAL"
                        confidence = (1 - pneumonia_prob) * 100
                    
                    # Display results
                    st.success("✅ Analysis Complete!")
                    
                    # Result box
                    result_class = "pneumonia-result" if predicted_class == "PNEUMONIA" else "normal-result"
                    
                    st.markdown(f"""
                        <div class="result-box {result_class}">
                            <h2>Diagnosis: {predicted_class}</h2>
                            <div class="confidence-text">{confidence:.2f}%</div>
                            <p style="font-size: 1.1rem;">Confidence Level</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Grad-CAM Visualization
                    with st.expander("🔍 Explain Prediction (Grad-CAM Heatmap)", expanded=True):
                        st.write("Visual explanation of what the model is looking at:")
                        
                        try:
                            # Preprocess image for Grad-CAM
                            img_array = preprocess_image(image)
                            
                            # Generate heatmap
                            heatmap = make_gradcam_heatmap(img_array, model)
                            
                            # Create overlay
                            # Resize heatmap to match original image size
                            heatmap = np.uint8(255 * heatmap)
                            heatmap = np.array(Image.fromarray(heatmap).resize(image.size, Image.Resampling.BILINEAR))
                            
                            # Convert original image to array (ensure RGB)
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            original_img = np.array(image)
                            
                            # Apply colormap
                            import cv2
                            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                            
                            # Superimpose
                            superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
                            
                            # Display side by side
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.image(image, caption="Original X-Ray", use_column_width=True)
                            with col_b:
                                st.image(superimposed_img, caption="AI Attention Heatmap", use_column_width=True)
                                
                            st.info("""
                            **How to interpret:**
                            - **Red/Yellow areas**: Regions contributing MOST to the 'Pneumonia' prediction.
                            - If the model highlights the **lungs**, it's looking at the right place.
                            - If it highlights **text/bones/edges**, it might be a false positive artifact.
                            """)
                            
                        except Exception as e:
                            import traceback
                            st.error(f"Could not generate heatmap: {e}")
                            st.code(traceback.format_exc())
                    
                    # Display confidence chart
                    st.pyplot(plot_confidence_chart(probabilities, class_names))
                    
                    # Additional information
                    if predicted_class == "PNEUMONIA":
                        st.warning("⚠️ **PNEUMONIA DETECTED** - Please consult with a healthcare professional immediately.")
                    else:
                        st.success("✅ **NORMAL** - No signs of pneumonia detected in the X-ray.")
                    
        else:
            st.info("👆 Please upload a chest X-ray image to begin analysis")
    
    # Disclaimer
    st.markdown("""
        <div class="disclaimer">
            <h4>⚠️ Medical Disclaimer</h4>
            <p>This AI model is designed for educational and research purposes only. 
            It should NOT be used as a substitute for professional medical diagnosis. 
            Always consult with qualified healthcare professionals for medical advice and treatment.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <p style="text-align: center; color: #666;">
            Built with ❤️ using Python, TensorFlow, and Streamlit | 
            Deep Learning Assignment Project
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
