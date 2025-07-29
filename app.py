import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Try to import plotly, fall back to matplotlib if not available
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    import matplotlib.pyplot as plt
    PLOTLY_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .churn-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .no-churn-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessors():
    """Load the trained model and preprocessing objects"""
    try:
        # Load the trained model
        model = tf.keras.models.load_model('model.h5')
        
        # Load preprocessing objects
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('label_encoder_gender.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        
        with open('onehot_encoder_geo.pkl', 'rb') as f:
            onehot_encoder_geo = pickle.load(f)
        
        return model, scaler, label_encoder_gender, onehot_encoder_geo
    except Exception as e:
        st.error(f"Error loading model or preprocessors: {str(e)}")
        return None, None, None, None

def preprocess_input(data, scaler, label_encoder_gender, onehot_encoder_geo):
    """Preprocess the input data for prediction"""
    try:
        # Create a copy of the data
        processed_data = data.copy()
        
        # Encode gender
        processed_data['Gender'] = label_encoder_gender.transform([processed_data['Gender']])[0]
        
        # One-hot encode geography
        geo_encoded = onehot_encoder_geo.transform([[processed_data['Geography']]]).toarray()
        geo_feature_names = onehot_encoder_geo.get_feature_names_out(['Geography'])
        
        # Remove original geography column and add encoded columns
        processed_data.pop('Geography')
        for i, feature_name in enumerate(geo_feature_names):
            processed_data[feature_name] = geo_encoded[0][i]
        
        # Convert to DataFrame and ensure correct order
        df = pd.DataFrame([processed_data])
        
        # Scale the features
        scaled_data = scaler.transform(df)
        
        return scaled_data
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 Customer Churn Prediction</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and preprocessors
    model, scaler, label_encoder_gender, onehot_encoder_geo = load_model_and_preprocessors()
    
    if model is None:
        st.error("Failed to load model. Please ensure all model files are present.")
        return
    
    # Sidebar for input
    st.sidebar.header("📝 Customer Information")
    st.sidebar.markdown("Enter customer details to predict churn probability:")
    
    # Input fields
    credit_score = st.sidebar.slider(
        "Credit Score", 
        min_value=300, 
        max_value=900, 
        value=650, 
        help="Customer's credit score (300-900)"
    )
    
    geography = st.sidebar.selectbox(
        "Geography", 
        options=["France", "Spain", "Germany"],
        help="Customer's country"
    )
    
    gender = st.sidebar.selectbox(
        "Gender", 
        options=["Male", "Female"],
        help="Customer's gender"
    )
    
    age = st.sidebar.slider(
        "Age", 
        min_value=18, 
        max_value=100, 
        value=35,
        help="Customer's age"
    )
    
    tenure = st.sidebar.slider(
        "Tenure (Years)", 
        min_value=0, 
        max_value=10, 
        value=5,
        help="Number of years as a customer"
    )
    
    balance = st.sidebar.number_input(
        "Account Balance", 
        min_value=0.0, 
        max_value=300000.0, 
        value=50000.0,
        help="Customer's account balance"
    )
    
    num_of_products = st.sidebar.slider(
        "Number of Products", 
        min_value=1, 
        max_value=4, 
        value=2,
        help="Number of bank products used"
    )
    
    has_cr_card = st.sidebar.selectbox(
        "Has Credit Card", 
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether customer has a credit card"
    )
    
    is_active_member = st.sidebar.selectbox(
        "Is Active Member", 
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether customer is an active member"
    )
    
    estimated_salary = st.sidebar.number_input(
        "Estimated Salary", 
        min_value=0.0, 
        max_value=200000.0, 
        value=75000.0,
        help="Customer's estimated annual salary"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📊 Customer Profile")
        
        # Display customer information
        customer_data = {
            "Credit Score": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": f"{tenure} years",
            "Balance": f"${balance:,.2f}",
            "Number of Products": num_of_products,
            "Has Credit Card": "Yes" if has_cr_card else "No",
            "Is Active Member": "Yes" if is_active_member else "No",
            "Estimated Salary": f"${estimated_salary:,.2f}"
        }
        
        # Create a nice display of customer data
        for key, value in customer_data.items():
            st.write(f"**{key}:** {value}")
    
    with col2:
        st.header("🔮 Prediction")
        
        # Prepare input data for prediction
        input_data = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary
        }
        
        # Make prediction
        if st.button("🎯 Predict Churn", type="primary"):
            with st.spinner("Analyzing customer data..."):
                # Preprocess the input
                processed_data = preprocess_input(input_data, scaler, label_encoder_gender, onehot_encoder_geo)
                
                if processed_data is not None:
                    # Make prediction
                    prediction_prob = model.predict(processed_data)[0][0]
                    prediction_binary = 1 if prediction_prob > 0.5 else 0
                    
                    # Display results
                    if prediction_binary == 1:
                        st.markdown(f"""
                        <div class="prediction-box churn-risk">
                            <h3>⚠️ High Churn Risk</h3>
                            <p><strong>Churn Probability: {prediction_prob:.2%}</strong></p>
                            <p>This customer is likely to churn. Consider retention strategies.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box no-churn-risk">
                            <h3>✅ Low Churn Risk</h3>
                            <p><strong>Churn Probability: {prediction_prob:.2%}</strong></p>
                            <p>This customer is likely to stay. Continue providing excellent service.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Probability visualization
                    if PLOTLY_AVAILABLE:
                        # Probability gauge chart
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number+delta",
                            value = prediction_prob * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            title = {'text': "Churn Probability (%)"},
                            delta = {'reference': 50},
                            gauge = {
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 25], 'color': "lightgreen"},
                                    {'range': [25, 50], 'color': "yellow"},
                                    {'range': [50, 75], 'color': "orange"},
                                    {'range': [75, 100], 'color': "red"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Simple progress bar as fallback
                        st.subheader("Churn Probability")
                        st.progress(prediction_prob)
                        st.write(f"**{prediction_prob:.2%}**")
    
    # Additional information
    st.markdown("---")
    st.header("ℹ️ About This Model")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏗️ Model Architecture")
        st.write("""
        - **Type**: Artificial Neural Network (ANN)
        - **Framework**: TensorFlow/Keras
        - **Layers**: 3 (64→32→1 neurons)
        - **Activation**: ReLU, Sigmoid
        """)
    
    with col2:
        st.subheader("📈 Performance")
        st.write("""
        - **Optimizer**: Adam
        - **Loss**: Binary Crossentropy
        - **Training**: Early Stopping
        - **Monitoring**: TensorBoard
        """)
    
    with col3:
        st.subheader("🎯 Use Cases")
        st.write("""
        - **Customer Retention**: Identify at-risk customers
        - **Marketing**: Target retention campaigns
        - **Business Strategy**: Reduce churn rates
        - **Revenue**: Improve customer lifetime value
        """)

if __name__ == "__main__":
    main()
