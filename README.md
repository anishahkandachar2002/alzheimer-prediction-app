# 🧠 Alzheimer's Disease Prediction - Explainable AI System

An interactive Streamlit application that provides transparent, explainable predictions for Alzheimer's disease diagnosis using LIME (Local Interpretable Model-agnostic Explanations).

## 🌟 Features

### 1. **Interactive Prediction Interface**
- User-friendly input forms for patient data
- Real-time predictions with confidence scores
- Support for all clinical and cognitive test scores

### 2. **Explainable AI with LIME**
- **Feature Importance Visualization**: Show how each feature contributes to the prediction
- **Local Explanations**: Understand the decision-making process for individual predictions
- **Feature Contribution Tables**: Detailed breakdown of each feature's impact
- **Transparency**: Understand exactly why the model made its prediction

### 3. **Model Insights**
- Feature importance visualization
- Model performance metrics
- Configuration details

### 4. **Educational Feature Guide**
- Comprehensive descriptions of all features
- Normal ranges for clinical tests
- Feature categorization (Demographics, Cognitive, Memory, etc.)

## 📋 Prerequisites

- Python 3.8 or higher
- Trained model file: `alzheimer_rf_model.pkl`
- Feature importance file: `feature_importance.csv`

## 🚀 Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify model files exist:**
   - `alzheimer_rf_model.pkl` (trained Random Forest model)
   - `feature_importance.csv` (feature importance data)

## ▶️ Running the Application

Run the Streamlit app with:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📊 How to Use

### Making a Prediction

1. **Navigate to the "Make Prediction" tab**

2. **Enter Patient Information:**
   - **Demographics**: Age, Education, Gender, APOE4 status
   - **Cognitive Tests**: MMSE, MOCA, ADAS11, ADAS13, CDRSB
   - **Memory & Function**: FAQ, RAVLT tests, Logical Memory, Trail Making

3. **Click "Predict Diagnosis"**

4. **Review Results:**
   - **Predicted Diagnosis**: CN (Cognitively Normal), MCI (Mild Cognitive Impairment), or Dementia
   - **Confidence Score**: Model's confidence in the prediction
   - **Probability Distribution**: Probabilities for all three classes

5. **Understand the Prediction with LIME:**
   - **Feature Importance Plot**: Shows the top features contributing to the prediction
   - **Feature Contributions Table**: Lists all features and their impact
   - **Local Explanation**: Visualization showing how features influence the specific prediction

### Understanding LIME Values

- **Positive values**: Push the prediction toward the predicted class
- **Negative values**: Push the prediction away from the predicted class
- **Magnitude**: Larger absolute values indicate stronger influence

### Example Interpretation

If predicting "Dementia":
- High CDRSB (Clinical Dementia Rating) → Pushes toward Dementia (positive)
- Low MMSE (cognitive score) → Pushes toward Dementia (positive)
- High RAVLT scores → Pushes away from Dementia (negative)

## 🎯 Input Features

### Demographics (4 features)
- **AGE**: Patient age (50-100 years)
- **PTEDUCAT**: Years of education (0-30)
- **PTGENDER_encoded**: Gender (0=Female, 1=Male)
- **APOE4**: Number of APOE4 alleles (0, 1, or 2)

### Cognitive Assessments (5 features)
- **MMSE**: Mini-Mental State Examination (0-30)
- **MOCA**: Montreal Cognitive Assessment (0-30)
- **ADAS11**: Alzheimer's Disease Assessment Scale 11 items (0-70)
- **ADAS13**: Alzheimer's Disease Assessment Scale 13 items (0-85)
- **CDRSB**: Clinical Dementia Rating Sum of Boxes (0-18)

### Memory & Function Tests (6 features)
- **FAQ**: Functional Activities Questionnaire (0-30)
- **RAVLT_immediate**: Rey Auditory Verbal Learning Test - Immediate (0-75)
- **RAVLT_learning**: Learning score (0-15)
- **RAVLT_forgetting**: Forgetting score (0-15)
- **LDELTOTAL**: Logical Memory Delayed Total (0-25)
- **TRABSCOR**: Trail Making Test (0-300 seconds)

## 📈 Model Information

- **Model Type**: Random Forest Classifier
- **Number of Trees**: 200
- **Test Accuracy**: ~90%+ (see sidebar for exact accuracy)
- **Training Data**: ADNI (Alzheimer's Disease Neuroimaging Initiative)
- **Classes**: 
  - CN (Cognitively Normal)
  - MCI (Mild Cognitive Impairment)
  - Dementia

## 🔍 Why LIME?

LIME (Local Interpretable Model-agnostic Explanations) provides:

1. **Local Interpretability**: Explains individual predictions by approximating the model locally
2. **Model-Agnostic**: Works with any machine learning model
3. **Intuitive**: Easy to understand feature contributions
4. **Transparency**: Makes the "black box" model transparent

This is crucial for medical applications where understanding the reasoning behind predictions is essential for clinical decision-making.

## ⚠️ Important Notes

1. **Not a Diagnostic Tool**: This application is for educational and research purposes only. It should NOT be used as a substitute for professional medical diagnosis.

2. **Clinical Validation Required**: All predictions should be reviewed by qualified healthcare professionals.

3. **Data Quality**: Ensure input data is accurate and complete for reliable predictions.

4. **Model Limitations**: The model is trained on specific data and may not generalize to all populations.

## 🛠️ Troubleshooting

### Model file not found
- Ensure `alzheimer_rf_model.pkl` is in the same directory as `app.py`
- If missing, run `save_model.py` to generate it

### LIME visualization issues
- Update matplotlib: `pip install --upgrade matplotlib`
- Clear Streamlit cache: Click "Clear cache" in the hamburger menu

### Slow performance
- LIME calculations can be computationally intensive
- Consider reducing the number of samples for the explainer

## 📚 References

- **LIME**: [github.com/marcotcr/lime](https://github.com/marcotcr/lime)
- **Streamlit**: [streamlit.io](https://streamlit.io)
- **ADNI**: [adni.loni.usc.edu](http://adni.loni.usc.edu)

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please ensure any changes maintain the explainability and transparency of the system.

---

**Made with ❤️ for transparent and explainable AI in healthcare**
