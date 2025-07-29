# Customer Churn Prediction using Artificial Neural Networks

A machine learning project that predicts customer churn using an Artificial Neural Network (ANN) built with TensorFlow and Keras. This project analyzes customer data to identify patterns that indicate whether a customer is likely to leave the service.

## 📊 Project Overview

Customer churn prediction is crucial for businesses to:
- Identify at-risk customers before they leave
- Implement targeted retention strategies
- Reduce customer acquisition costs
- Improve overall customer lifetime value

This project uses a deep learning approach with a neural network to predict customer churn based on various customer attributes.

## 🗂️ Dataset

The project uses the `Churn_Modelling.csv` dataset containing customer information with the following features:

- **RowNumber**: Record index
- **CustomerId**: Unique customer identifier
- **Surname**: Customer surname
- **CreditScore**: Customer credit score
- **Geography**: Customer location (France, Spain, Germany)
- **Gender**: Customer gender (Male/Female)
- **Age**: Customer age
- **Tenure**: Number of years as a customer
- **Balance**: Account balance
- **NumOfProducts**: Number of products used
- **HasCrCard**: Whether customer has a credit card (1/0)
- **IsActiveMember**: Whether customer is an active member (1/0)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Target variable - whether customer churned (1/0)

## 🏗️ Model Architecture

The neural network consists of:
- **Input Layer**: Features after preprocessing
- **Hidden Layer 1**: 64 neurons with ReLU activation
- **Hidden Layer 2**: 32 neurons with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation (binary classification)

### Model Configuration:
- **Optimizer**: Adam (learning rate: 0.01)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Callbacks**: Early Stopping (patience=10), TensorBoard logging

## 🛠️ Technologies Used

- **Python 3.11**
- **TensorFlow 2.15.0** - Deep learning framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Data preprocessing and model evaluation
- **Matplotlib** - Data visualization
- **Streamlit** - Web application framework
- **TensorBoard** - Model monitoring and visualization

## 📁 Project Structure

```
ann/
├── Churn_Modelling.csv          # Dataset
├── experiments.ipynb            # Main notebook with model development
├── model.h5                     # Trained model
├── requirements.txt             # Python dependencies
├── scaler.pkl                   # Fitted StandardScaler
├── label_encoder_gender.pkl     # Gender label encoder
├── onehot_encoder_geo.pkl       # Geography one-hot encoder
├── logs/                        # TensorBoard logs
│   └── fit/
├── venv/                        # Virtual environment
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dhanushm18/ann.git
   cd ann
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook experiments.ipynb
   ```

2. **View TensorBoard logs:**
   ```bash
   tensorboard --logdir logs/fit
   ```

3. **For Streamlit application (if implemented):**
   ```bash
   streamlit run app.py
   ```

## 📈 Data Preprocessing

The preprocessing pipeline includes:

1. **Feature Selection**: Removing non-predictive columns (RowNumber, CustomerId, Surname)
2. **Label Encoding**: Converting Gender to numerical values
3. **One-Hot Encoding**: Converting Geography to binary features
4. **Feature Scaling**: Standardizing numerical features using StandardScaler
5. **Train-Test Split**: 80-20 split with random state 42

## 🎯 Model Performance

The model uses the following evaluation approach:
- **Training/Validation Split**: 80/20
- **Early Stopping**: Prevents overfitting
- **TensorBoard Monitoring**: Real-time training visualization

## 📊 Monitoring and Visualization

- **TensorBoard**: Monitor training progress, loss curves, and model metrics
- **Model Checkpoints**: Best model saved as `model.h5`
- **Preprocessing Artifacts**: Scalers and encoders saved for inference

## 🔮 Making Predictions

To make predictions on new data:

1. Load the trained model and preprocessing objects
2. Apply the same preprocessing pipeline
3. Use the model for inference

```python
import pickle
import tensorflow as tf

# Load model and preprocessors
model = tf.keras.models.load_model('model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# ... load other preprocessors

# Preprocess new data and predict
# predictions = model.predict(processed_data)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Dhanush M**
- GitHub: [@dhanushm18](https://github.com/dhanushm18)
- Email: dmdhanushm17@gmail.com

## 🙏 Acknowledgments

- Dataset source and inspiration
- TensorFlow and Keras documentation
- Open source community contributions
