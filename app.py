"""
Alzheimer's Disease Prediction - Streamlit UI with LIME Explanations
This application provides transparent predictions with explainable AI using LIME
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import LIME for explanations
try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("LIME not available")

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    h3 {
        color: #34495e;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model and associated data"""
    try:
        with open('alzheimer_rf_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'alzheimer_rf_model.pkl' exists.")
        st.stop()

@st.cache_data
def load_feature_importance():
    """Load feature importance data"""
    try:
        return pd.read_csv('feature_importance.csv')
    except FileNotFoundError:
        return None

@st.cache_data
def load_training_data():
    """Load training data for LIME background"""
    try:
        df = pd.read_csv('ml_dataset_cleaned.csv')
        return df
    except FileNotFoundError:
        return None

def get_feature_descriptions():
    """Return descriptions for each feature"""
    return {
        'AGE': 'Patient age in years',
        'PTEDUCAT': 'Years of education',
        'PTGENDER_encoded': 'Gender (0: Female, 1: Male)',
        'APOE4': 'Number of APOE4 alleles (0, 1, or 2)',
        'MMSE': 'Mini-Mental State Examination score (0-30)',
        'ADAS11': 'Alzheimer\'s Disease Assessment Scale 11 items',
        'ADAS13': 'Alzheimer\'s Disease Assessment Scale 13 items',
        'CDRSB': 'Clinical Dementia Rating Sum of Boxes',
        'FAQ': 'Functional Activities Questionnaire',
        'MOCA': 'Montreal Cognitive Assessment score',
        'RAVLT_immediate': 'Rey Auditory Verbal Learning Test - Immediate',
        'RAVLT_learning': 'Rey Auditory Verbal Learning Test - Learning',
        'RAVLT_forgetting': 'Rey Auditory Verbal Learning Test - Forgetting',
        'LDELTOTAL': 'Logical Memory Delayed Total score',
        'TRABSCOR': 'Trail Making Test score'
    }

def main():
    # Header
    st.title("🧠 Alzheimer's Disease Prediction System")
    st.markdown("### Explainable AI for Medical Diagnosis with LIME")
    st.markdown("---")
    
    # Load model
    model_package = load_model()
    model = model_package['model']
    scaler = model_package['scaler']
    feature_names = model_package['feature_names']
    class_names = model_package['class_names']
    test_accuracy = model_package['test_accuracy']
    
    # Load feature importance
    feature_importance_df = load_feature_importance()
    
    # Load training data for LIME
    training_data = load_training_data()
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        st.metric("Model Type", "Random Forest")
        st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
        st.metric("Number of Features", len(feature_names))
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        This application uses a Random Forest model trained on ADNI data to predict 
        Alzheimer's disease diagnosis. 
        
        **LIME (Local Interpretable Model-agnostic Explanations)** is used to explain each 
        prediction, showing which features contributed most to the result.
        """)
        
        st.markdown("---")
        st.header("🎯 Diagnosis Classes")
        st.write("**CN**: Cognitively Normal")
        st.write("**MCI**: Mild Cognitive Impairment")
        st.write("**Dementia**: Alzheimer's Disease")
        
        if not LIME_AVAILABLE:
            st.warning("⚠️ LIME library not loaded. Install with: pip install lime")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📈 Model Insights", "📚 Feature Guide"])
    
    # TAB 1: Make Prediction
    with tab1:
        st.header("Patient Information Input")
        
        col1, col2, col3 = st.columns(3)
        
        feature_descriptions = get_feature_descriptions()
        input_data = {}
        
        # Organize inputs by category
        with col1:
            st.subheader("Demographics")
            input_data['AGE'] = st.number_input(
                "Age (years)", 
                min_value=50, max_value=100, value=70,
                help=feature_descriptions['AGE']
            )
            input_data['PTEDUCAT'] = st.number_input(
                "Education (years)", 
                min_value=0, max_value=30, value=16,
                help=feature_descriptions['PTEDUCAT']
            )
            input_data['PTGENDER_encoded'] = st.selectbox(
                "Gender", 
                options=[0, 1], 
                format_func=lambda x: "Female" if x == 0 else "Male",
                help=feature_descriptions['PTGENDER_encoded']
            )
            input_data['APOE4'] = st.selectbox(
                "APOE4 Alleles", 
                options=[0, 1, 2],
                help=feature_descriptions['APOE4']
            )
        
        with col2:
            st.subheader("Cognitive Tests")
            input_data['MMSE'] = st.slider(
                "MMSE Score", 
                min_value=0, max_value=30, value=24,
                help=feature_descriptions['MMSE']
            )
            input_data['MOCA'] = st.slider(
                "MOCA Score", 
                min_value=0, max_value=30, value=20,
                help=feature_descriptions['MOCA']
            )
            input_data['ADAS11'] = st.number_input(
                "ADAS11", 
                min_value=0.0, max_value=70.0, value=10.0, step=0.5,
                help=feature_descriptions['ADAS11']
            )
            input_data['ADAS13'] = st.number_input(
                "ADAS13", 
                min_value=0.0, max_value=85.0, value=15.0, step=0.5,
                help=feature_descriptions['ADAS13']
            )
            input_data['CDRSB'] = st.number_input(
                "CDR Sum of Boxes", 
                min_value=0.0, max_value=18.0, value=2.0, step=0.5,
                help=feature_descriptions['CDRSB']
            )
        
        with col3:
            st.subheader("Memory & Function Tests")
            input_data['FAQ'] = st.slider(
                "FAQ Score", 
                min_value=0, max_value=30, value=5,
                help=feature_descriptions['FAQ']
            )
            input_data['RAVLT_immediate'] = st.number_input(
                "RAVLT Immediate", 
                min_value=0.0, max_value=75.0, value=30.0, step=1.0,
                help=feature_descriptions['RAVLT_immediate']
            )
            input_data['RAVLT_learning'] = st.number_input(
                "RAVLT Learning", 
                min_value=0.0, max_value=15.0, value=5.0, step=0.5,
                help=feature_descriptions['RAVLT_learning']
            )
            input_data['RAVLT_forgetting'] = st.number_input(
                "RAVLT Forgetting", 
                min_value=0.0, max_value=15.0, value=3.0, step=0.5,
                help=feature_descriptions['RAVLT_forgetting']
            )
            input_data['LDELTOTAL'] = st.number_input(
                "Logical Memory Delayed", 
                min_value=0.0, max_value=25.0, value=10.0, step=0.5,
                help=feature_descriptions['LDELTOTAL']
            )
            input_data['TRABSCOR'] = st.number_input(
                "Trail Making Test", 
                min_value=0.0, max_value=300.0, value=80.0, step=5.0,
                help=feature_descriptions['TRABSCOR']
            )
        
        st.markdown("---")
        
        # Prediction button
        if st.button("🔍 Predict Diagnosis", type="primary", use_container_width=True):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct feature order
            input_df = input_df[feature_names]
            
            # Scale the input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Display prediction
            st.markdown("## 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            # Predicted class
            predicted_class = class_names[prediction]
            
            # Color coding
            color_map = {
                'CN (Cognitively Normal)': '#2ecc71',
                'MCI (Mild Cognitive Impairment)': '#f39c12',
                'Dementia': '#e74c3c'
            }
            
            with col1:
                st.markdown(f"""
                <div style='background-color: {color_map[predicted_class]}; padding: 20px; 
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>Predicted Diagnosis</h2>
                    <h1 style='color: white; margin: 10px 0;'>{predicted_class.split('(')[0].strip()}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: #3498db; padding: 20px; 
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>Confidence</h2>
                    <h1 style='color: white; margin: 10px 0;'>{prediction_proba[prediction]*100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='background-color: #9b59b6; padding: 20px; 
                border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>Model Accuracy</h2>
                    <h1 style='color: white; margin: 10px 0;'>{test_accuracy*100:.1f}%</h1>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("### 📊 Probability Distribution")
            
            prob_df = pd.DataFrame({
                'Diagnosis': [name.split('(')[0].strip() for name in class_names],
                'Probability': prediction_proba * 100
            })
            
            fig = px.bar(
                prob_df, 
                x='Diagnosis', 
                y='Probability',
                color='Diagnosis',
                color_discrete_map={
                    'CN': '#2ecc71',
                    'MCI': '#f39c12',
                    'Dementia': '#e74c3c'
                },
                text='Probability'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(
                showlegend=False,
                yaxis_title="Probability (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # LIME Explanation
            st.markdown("---")
            st.markdown("## 🔍 Explanation: Why This Prediction?")
            st.info("""
            **LIME (Local Interpretable Model-agnostic Explanations)** creates a simple, 
            interpretable model around this specific prediction to show which features 
            contributed most to the result.
            """)
            
            # Create LIME explainer
            if LIME_AVAILABLE and training_data is not None:
                with st.spinner("Generating LIME explanations..."):
                    try:
                        # Prepare training data for LIME
                        feature_cols = [col for col in training_data.columns if col not in ['PTID', 'DX', 'DX_encoded', 'PTGENDER']]
                        X_train = training_data[feature_cols]
                        
                        # Fill missing values
                        X_train = X_train.fillna(X_train.median())
                        
                        # Scale training data
                        X_train_scaled = scaler.transform(X_train)
                        
                        # Create LIME explainer
                        explainer = lime_tabular.LimeTabularExplainer(
                            X_train_scaled,
                            feature_names=feature_names,
                            class_names=[name.split('(')[0].strip() for name in class_names],
                            mode='classification',
                            random_state=42
                        )
                        
                        # Explain the prediction
                        explanation = explainer.explain_instance(
                            input_scaled[0],
                            model.predict_proba,
                            num_features=len(feature_names),
                            top_labels=3
                        )
                        
                        # Display explanation for predicted class
                        st.markdown(f"### Feature Contributions for Predicted Class: **{predicted_class.split('(')[0].strip()}**")
                        
                        # Get feature contributions for the predicted class
                        lime_values = explanation.as_list(label=prediction)
                        
                        # Parse LIME output
                        features_list = []
                        contributions_list = []
                        
                        for item in lime_values:
                            feature_desc = item[0]
                            contribution = item[1]
                            
                            # Extract feature name (LIME gives ranges like "feature <= value")
                            for fname in feature_names:
                                if fname in feature_desc:
                                    features_list.append(fname)
                                    contributions_list.append(contribution)
                                    break
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        y_pos = np.arange(len(features_list))
                        colors = ['#ff0051' if x > 0 else '#008bfb' for x in contributions_list]
                        
                        ax.barh(y_pos, contributions_list, color=colors, alpha=0.8)
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(features_list)
                        ax.set_xlabel('LIME Contribution (impact on prediction)', fontsize=12)
                        ax.set_title(f'Feature Contributions to "{predicted_class.split("(")[0].strip()}" Prediction', 
                                   fontsize=14, fontweight='bold')
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        
                        # Add value labels
                        for i, val in enumerate(contributions_list):
                            label = f"{val:.3f}"
                            x_pos = val + (0.01 if val > 0 else -0.01)
                            ha = 'left' if val > 0 else 'right'
                            ax.text(x_pos, i, label, va='center', ha=ha, fontsize=9)
                        
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()
                        
                        # Feature contribution table
                        st.markdown("### 📋 Detailed Feature Contributions")
                        
                        contribution_df = pd.DataFrame({
                            'Feature': features_list,
                            'LIME Contribution': contributions_list,
                            'Impact': ['Supports Prediction ↑' if x > 0 else 'Against Prediction ↓' for x in contributions_list]
                        })
                        contribution_df['Abs Contribution'] = abs(contribution_df['LIME Contribution'])
                        contribution_df = contribution_df.sort_values('Abs Contribution', ascending=False)
                        
                        # Display top 10 features
                        st.dataframe(
                            contribution_df[['Feature', 'LIME Contribution', 'Impact']].head(10),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Summary interpretation
                        st.markdown("### 💡 Interpretation")
                        top_positive = contribution_df[contribution_df['LIME Contribution'] > 0].head(3)
                        top_negative = contribution_df[contribution_df['LIME Contribution'] < 0].head(3)
                        
                        if len(top_positive) > 0:
                            st.success(f"**Features supporting '{predicted_class.split('(')[0].strip()}' diagnosis:**")
                            for _, row in top_positive.iterrows():
                                st.write(f"- **{row['Feature']}**: Contribution = {row['LIME Contribution']:.3f}")
                        
                        if len(top_negative) > 0:
                            st.warning(f"**Features against '{predicted_class.split('(')[0].strip()}' diagnosis:**")
                            for _, row in top_negative.iterrows():
                                st.write(f"- **{row['Feature']}**: Contribution = {row['LIME Contribution']:.3f}")
                        
                        # Show probabilities for all classes
                        st.markdown("### 🎨 Prediction Probabilities for All Classes")
                        
                        for idx, class_name in enumerate(class_names):
                            with st.expander(f"View explanation for {class_name.split('(')[0].strip()} (Probability: {prediction_proba[idx]*100:.1f}%)"):
                                lime_values_class = explanation.as_list(label=idx)
                                
                                features_class = []
                                contributions_class = []
                                
                                for item in lime_values_class[:10]:  # Top 10 features
                                    feature_desc = item[0]
                                    contribution = item[1]
                                    
                                    for fname in feature_names:
                                        if fname in feature_desc:
                                            features_class.append(fname)
                                            contributions_class.append(contribution)
                                            break
                                
                                class_df = pd.DataFrame({
                                    'Feature': features_class,
                                    'Contribution': contributions_class
                                })
                                
                                fig_class = px.bar(
                                    class_df,
                                    x='Contribution',
                                    y='Feature',
                                    orientation='h',
                                    color='Contribution',
                                    color_continuous_scale=['#008bfb', '#ffffff', '#ff0051'],
                                    color_continuous_midpoint=0
                                )
                                fig_class.update_layout(height=400)
                                st.plotly_chart(fig_class, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error generating LIME explanations: {str(e)}")
                        st.info("Showing feature importance instead...")
                        
                        # Fallback to feature importance
                        if feature_importance_df is not None:
                            st.markdown("### 📊 Feature Importance (Global)")
                            fig = px.bar(
                                feature_importance_df.head(10),
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                color='Importance',
                                color_continuous_scale='Blues'
                            )
                            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("LIME library or training data not available. Showing feature importance instead.")
                if feature_importance_df is not None:
                    st.markdown("### 📊 Feature Importance (Global)")
                    fig = px.bar(
                        feature_importance_df.head(10),
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=500)
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Model Insights
    with tab2:
        st.header("Model Performance & Feature Importance")
        
        if feature_importance_df is not None:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🏆 Top 10 Most Important Features")
                
                top_features = feature_importance_df.head(10)
                
                fig = px.bar(
                    top_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=500,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Feature Importance Distribution")
                
                fig = px.pie(
                    feature_importance_df.head(10),
                    values='Importance',
                    names='Feature',
                    hole=0.4
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Full feature importance table
            st.subheader("📋 Complete Feature Importance Table")
            st.dataframe(
                feature_importance_df.style.background_gradient(subset=['Importance'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )
        
        # Model parameters
        st.markdown("---")
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Model Type": "Random Forest Classifier",
                "Number of Trees": 200,
                "Max Depth": 10,
                "Min Samples Split": 10,
                "Min Samples Leaf": 5
            })
        
        with col2:
            st.json({
                "Max Features": "sqrt",
                "Class Weight": "balanced",
                "Random State": 42,
                "Test Accuracy": f"{test_accuracy*100:.2f}%"
            })
    
    # TAB 3: Feature Guide
    with tab3:
        st.header("📚 Feature Descriptions & Normal Ranges")
        
        feature_guide = {
            'Demographics': {
                'AGE': {'desc': 'Patient age in years', 'range': '50-100 years', 'type': 'Demographic'},
                'PTEDUCAT': {'desc': 'Years of formal education', 'range': '0-30 years', 'type': 'Demographic'},
                'PTGENDER_encoded': {'desc': 'Biological sex (0=Female, 1=Male)', 'range': '0 or 1', 'type': 'Demographic'},
                'APOE4': {'desc': 'Number of APOE4 alleles (genetic risk factor)', 'range': '0, 1, or 2', 'type': 'Genetic'}
            },
            'Cognitive Assessments': {
                'MMSE': {'desc': 'Mini-Mental State Examination', 'range': '24-30: Normal, 18-23: Mild, <18: Severe', 'type': 'Cognitive'},
                'MOCA': {'desc': 'Montreal Cognitive Assessment', 'range': '26-30: Normal, <26: Impairment', 'type': 'Cognitive'},
                'ADAS11': {'desc': 'Alzheimer\'s Disease Assessment Scale (11 items)', 'range': '0-70 (lower is better)', 'type': 'Cognitive'},
                'ADAS13': {'desc': 'Alzheimer\'s Disease Assessment Scale (13 items)', 'range': '0-85 (lower is better)', 'type': 'Cognitive'},
                'CDRSB': {'desc': 'Clinical Dementia Rating Sum of Boxes', 'range': '0: Normal, 0.5-4: MCI, >4: Dementia', 'type': 'Clinical'}
            },
            'Memory & Function': {
                'FAQ': {'desc': 'Functional Activities Questionnaire', 'range': '0-30 (higher indicates more impairment)', 'type': 'Functional'},
                'RAVLT_immediate': {'desc': 'Rey Auditory Verbal Learning Test - Immediate recall', 'range': '0-75', 'type': 'Memory'},
                'RAVLT_learning': {'desc': 'Rey Auditory Verbal Learning Test - Learning score', 'range': '0-15', 'type': 'Memory'},
                'RAVLT_forgetting': {'desc': 'Rey Auditory Verbal Learning Test - Forgetting score', 'range': '0-15', 'type': 'Memory'},
                'LDELTOTAL': {'desc': 'Logical Memory Delayed Total score', 'range': '0-25', 'type': 'Memory'},
                'TRABSCOR': {'desc': 'Trail Making Test score', 'range': '0-300 seconds (lower is better)', 'type': 'Executive Function'}
            }
        }
        
        for category, features in feature_guide.items():
            st.subheader(f"📌 {category}")
            
            for feature_name, info in features.items():
                with st.expander(f"**{feature_name}** - {info['type']}"):
                    st.write(f"**Description:** {info['desc']}")
                    st.write(f"**Normal Range:** {info['range']}")
                    
                    # Show importance if available
                    if feature_importance_df is not None:
                        importance_row = feature_importance_df[feature_importance_df['Feature'] == feature_name]
                        if not importance_row.empty:
                            importance = importance_row.iloc[0]['Importance']
                            st.metric("Feature Importance", f"{importance:.4f}")
        
        st.markdown("---")
        st.info("""
        **Note:** These ranges are general guidelines. Clinical interpretation should always 
        be done by qualified healthcare professionals considering the complete clinical picture.
        """)

if __name__ == "__main__":
    main()
