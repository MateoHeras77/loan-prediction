# Loan Risk Prediction Dashboard

This project is a Streamlit dashboard that uses a machine learning model to predict loan risk based on customer information.

## Features

- Interactive web interface for entering customer data
- Real-time loan risk prediction
- Visualization of prediction results and confidence scores
- Responsive design for use on different devices

## Setup Instructions

1. Ensure you have Python 3.7+ installed on your system

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure the model file is in the correct location:
```
Heart_Disease_Risk_Prediction/models/best_risk_classifier.pkl
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

5. The application will open in your default web browser

## Usage

1. Fill in the customer information form with the relevant details
2. Click 'Predict Risk' to get the prediction result
3. View the predicted risk level and associated confidence score
4. Analyze the key risk factors displayed in the dashboard

## Model Information

The model used in this application is an ensemble classifier trained on historical loan data. It predicts whether a loan applicant falls into the 'High Risk' or 'Low Risk' category based on features such as:

- Personal information (Age, Marital Status)
- Financial status (Income)
- Employment details (Experience, Profession)
- Housing information (House Ownership, Years in Current House)
- Other factors (Car Ownership, Location)

## Troubleshooting

If you encounter issues loading the model, ensure:
- The file path is correct
- The model was trained with a compatible scikit-learn version
- All required preprocessing steps match the ones applied during model training
