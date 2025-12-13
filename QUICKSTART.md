# 🚀 Quick Start Guide - Pneumonia Detection System

## 📍 Current Status

✅ **Completed:**
- Project structure created
- Virtual environment set up
- All dependencies installed (TensorFlow, Streamlit, etc.)
- Dataset downloaded from Kaggle (5,863 X-ray images)
- Dataset organized and verified
- Application code ready (`app.py`, `train_model.py`)

📊 **Dataset Statistics:**
- **Training Set**: 5,216 images (1,341 Normal, 3,875 Pneumonia)
- **Validation Set**: 16 images (8 Normal, 8 Pneumonia)
- **Test Set**: 624 images (234 Normal, 390 Pneumonia)

---

## 🎯 Next Steps

### Step 1: Train the Model (Required)

The model needs to be trained before you can use the Streamlit app. This will take approximately **30-45 minutes with GPU** or **2-3 hours without GPU**.

```powershell
cd "c:\Users\Ali Shaikh\Documents\python\pneumonia-detection-system"
.\venv\Scripts\python.exe train_model.py
```

**What happens during training:**
1. Loads and preprocesses the dataset
2. Builds MobileNetV2 transfer learning model
3. Trains for up to 20 epochs (with early stopping)
4. Saves the best model to `models/pneumonia_model.h5`
5. Generates visualization plots:
   - `training_history.png` - Training/validation curves
   - `confusion_matrix.png` - Model performance
   - `prediction_samples.png` - Sample predictions

**Expected Output:**
- Test Accuracy: 90%+
- Training will stop automatically when performance plateaus

---

### Step 2: Run the Streamlit Application

Once the model is trained, launch the web application:

```powershell
cd "c:\Users\Ali Shaikh\Documents\python\pneumonia-detection-system"
.\venv\Scripts\streamlit.exe run app.py
```

The application will open in your browser at `http://localhost:8501`

**Features:**
- 🖼️ **Upload** chest X-ray images
- 🤖 **AI Analysis** with standard MobileNetV2
- 🌡️ **Confidence Slider**: Adjust sensitivity to reduce false positives
- 🔍 **X-ray Vision (Grad-CAM)**: See *exactly* what the AI is looking at
- 📊 **Visual Charts**: Professional confidence metrics

---

### Step 3: Test with "X-ray Vision"

1.  **Upload a test image**.
2.  **Adjust Threshold**: Use the sidebar slider (default 0.75). Higher = Stricter.
3.  **Explain Prediction**: Expand the "Explain Prediction" box.
    - **Red/Yellow on Lungs**: Good! The model is looking at the right place.
    - **Red on Cables/Text**: Bad! The model might be confused (try a different image).

### Step 4: Verify Results

1.  **Check Diagnosis**: Ensure it matches the folder (Normal vs Pneumonia).
2.  **Verify Confidence**: A high confidence (>90%) is good.
3.  **Check Heatmap**: Ensure it makes sense (lungs vs artifacts).

---

## 🌐 Deployment to Streamlit Cloud (Optional)

### Prerequisites:
- GitHub account
- Trained model file (`models/pneumonia_model.h5`)

### Deployment Steps:

#### 1. Initialize Git Repository
```powershell
cd "c:\Users\Ali Shaikh\Documents\python\pneumonia-detection-system"
git init
git add .
git commit -m "Initial commit: Pneumonia Detection System"
```

#### 2. Create GitHub Repository
1. Go to https://github.com/new
2. Name: `pneumonia-detection-system`
3. Don't initialize with README (we already have one)
4. Click "Create repository"

#### 3. Push to GitHub
```powershell
# Replace 'yourusername' with your GitHub username
git remote add origin https://github.com/yourusername/pneumonia-detection-system.git
git branch -M main
git push -u origin main
```

#### 4. Handle Large Model File (if needed)
If `pneumonia_model.h5` is larger than 100MB, use Git LFS:
```powershell
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add models/pneumonia_model.h5
git commit -m "Add model with Git LFS"
git push
```

#### 5. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `pneumonia-detection-system`
5. Main file path: `app.py`
6. Click "Deploy"

**Deployment time**: 3-5 minutes

Your app will be live at: `https://yourusername-pneumonia-detection-system.streamlit.app`

---

## 🔧 Troubleshooting

### Issue: "ValueError: numpy.dtype size changed"
**Solution**: This is a conflict between differrent libraries. We fixed it by using a specific numpy version:
```powershell
.\venv\Scripts\pip.exe install --force-reinstall "numpy==1.26.4"
```

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution**: You are missing the image processing library:
```powershell
.\venv\Scripts\pip.exe install opencv-python-headless
```

### Issue: "Model file not found"
**Solution**: You need to train the model first using Step 1.

### Issue: Out of memory during training
**Solution**: Reduce batch size in `train_model.py`:
```python
preprocessor = DataPreprocessor(img_height=224, img_width=224, batch_size=16)  # Changed from 32 to 16
```

### Issue: Streamlit not found
**Solution**: Make sure you're using the virtual environment:
```powershell
.\venv\Scripts\streamlit.exe run app.py
```

### Issue: Slow training
**Solution**: This is normal without GPU. To use GPU:
1. Install CUDA and cuDNN
2. Reinstall TensorFlow with GPU support:
   ```powershell
   .\venv\Scripts\pip.exe install tensorflow-gpu==2.15.0
   ```

---

## 📋 Assignment Submission Checklist

Once you complete the training and testing:

- [ ] Trained model file (✅ `models/pneumonia_model.h5`)
- [ ] Complete project folder (✅ All files created)
- [ ] app.py Streamlit file (✅ Created)
- [ ] GitHub repository (⏳ To be created)
- [ ] Deployment link (⏳ To be deployed)
- [ ] Training accuracy curves (⏳ Generated after training)
- [ ] Confusion matrix (⏳ Generated after training)
- [ ] Application screenshots (⏳ Capture after running app)
- [ ] README.md documentation (✅ Created)

---

## 💡 Tips for Success

1. **Training Time**: Start the training and let it run. It will save progress automatically.

2. **Testing**: Use images from the `dataset/test` folder to verify predictions.

3. **Screenshots**: Capture screenshots of:
   - Streamlit app interface
   - Successful predictions
   - Confidence charts
   - Training curves

4. **Documentation**: The README.md has all the information needed for your assignment report.

5. **GitHub**: Make sure to push all files including the trained model before deploying.

---

## 🆘 Need Help?

If you encounter any issues:

1. Check the error message carefully
2. Verify you're in the correct directory
3. Ensure the virtual environment is activated
4. Check that all files are in the right locations
5. Review the training output for errors

---

## 🎓 Learning Outcomes

By completing this project, you've learned:

✅ Transfer Learning with MobileNetV2
✅ Image Classification for Medical AI
✅ Data preprocessing and augmentation
✅ Model training with callbacks
✅ Streamlit application development
✅ Model deployment on the cloud
✅ Git and GitHub workflow
✅ Professional software documentation

---

**Ready to start? Run Step 1 to train your model!** 🚀
