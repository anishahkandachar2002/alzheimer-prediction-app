# 🧠 Alzheimer's Disease Prediction - Streamlit Deployment

## 📦 Deployment Files

This folder contains **ONLY** the files needed to deploy the Streamlit app to Streamlit Cloud.

### ✅ Files to Push to GitHub:

```
streamlit_deploy/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── alzheimer_rf_model.pkl          # Trained model (2.3 MB)
├── feature_importance.csv          # Feature importance data
├── ml_dataset_cleaned.csv          # Training data for LIME (108 KB)
├── README.md                       # This file
├── DEPLOYMENT.md                   # Deployment instructions
└── .gitignore                      # Git ignore rules
```

**Total size: ~2.5 MB** (well within GitHub limits)

---

## 🚀 How to Deploy to Streamlit Cloud

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `alzheimer-prediction-app`
3. Make it **Public** (required for free Streamlit Cloud)
4. Don't initialize with README (we already have one)

### Step 2: Push Files to GitHub

Open terminal in the `streamlit_deploy` folder and run:

```bash
git init
git add .
git commit -m "Initial commit: Alzheimer's prediction app with LIME explanations"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/alzheimer-prediction-app.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select:
   - **Repository**: `YOUR_USERNAME/alzheimer-prediction-app`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

### Step 4: Wait for Deployment

- Streamlit Cloud will install dependencies (takes 2-3 minutes)
- Your app will be live at: `https://YOUR_USERNAME-alzheimer-prediction-app.streamlit.app`

---

## 🔧 Troubleshooting

### Issue: "Module not found" error
**Solution**: Make sure `requirements.txt` is in the same folder as `app.py`

### Issue: "File not found: alzheimer_rf_model.pkl"
**Solution**: Ensure the `.pkl` file is committed to GitHub (check file size < 100MB)

### Issue: App crashes on prediction
**Solution**: Make sure `ml_dataset_cleaned.csv` is included (needed for LIME)

### Issue: Slow loading
**Solution**: This is normal on first load. Streamlit Cloud caches the model after first run.

---

## 📊 What the App Does

- **Predicts Alzheimer's diagnosis** (CN, MCI, or Dementia)
- **Explains predictions** using LIME (Local Interpretable Model-agnostic Explanations)
- **Shows feature contributions** for transparency
- **Provides educational information** about clinical tests

---

## 🔒 Important Notes

### Data Privacy
- This app uses **synthetic/anonymized** ADNI data
- No real patient data is stored or transmitted
- For research and educational purposes only

### Medical Disclaimer
⚠️ **This is NOT a diagnostic tool**
- Predictions are for educational purposes only
- Always consult qualified healthcare professionals
- Do not use for actual medical decisions

---

## 📝 License

This project is for educational and research purposes.

---

## 🤝 Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Verify all files are in the repository
3. Ensure requirements.txt has correct versions

---

**Made with ❤️ for transparent and explainable AI in healthcare**
