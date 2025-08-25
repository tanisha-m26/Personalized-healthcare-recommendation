# Personalized Healthcare Recommendation System

A **real-time interactive healthcare recommendation system** built using **Python, Streamlit, and machine learning**. This dashboard allows users to explore healthcare data, view visualizations, and get personalized recommendations based on patient metrics.

---


## 🛠 Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/personalized-healthcare-recommendation.git
cd personalized-healthcare-recommendation
Create a virtual environment (optional but recommended)

```bash


pip install -r requirements.txt

pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn


🚀 Running Locally

Start the Streamlit app:


streamlit run app.py


Then open your browser at:


Local URL: http://localhost:8501
You should see the dashboard with options:

Home: Explore dataset and visualizations.

Feature Importance: View importance of features in the model.

Model Metrics: View classification report & confusion matrix.

Live Prediction: Input patient metrics to get personalized recommendations.

🧪 Training the Model
If you want to retrain the model:
```bash
python train_model.py

This will generate/update models/healthcare_model.pkl.

📊 Features
Real-time patient data input and prediction

Interactive visualizations (histograms, scatter plots)

Feature importance charts

Model performance metrics (classification report, confusion matrix)

Sidebar navigation with logo and metrics overview

🌐 Deployment
You can deploy the app using Streamlit Cloud:

Push your repository to GitHub:


git add .
git commit -m "Initial commit"
git push origin main
Go to Streamlit Cloud(https://share.streamlit.io/) and click New App.

Select your GitHub repo and branch, then deploy.

The app URL will be provided for access anywhere.

🔗 Useful Links

 Streamlit_Documentation :https://docs.streamlit.io/
Scikit-learn_Documentation :https://scikit-learn.org/stable/documentation.html
Plotly_Express_Documentation :https://plotly.com/python/plotly-express/