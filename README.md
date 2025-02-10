# SMS-spam-detection
Email/SMS Spam Classifier
This project is a machine learning-based spam detection application. It classifies email or SMS messages as either "Spam" or "Not Spam" using a trained machine learning model.

# Table of Contents
 1)Project Overview
 2)Features
 3)Setup and Installation
 4)How to Run the App
 5)Project Files
 6)Technologies Used
 7)Acknowledgments
 
# Project Overview
The Email/SMS Spam Classifier uses Natural Language Processing (NLP) and a machine learning model to analyze and classify text messages. It helps users filter out unwanted spam messages effectively.

Key Steps:
Text preprocessing (tokenization, stemming, removing stopwords).
Feature extraction using TF-IDF Vectorizer.
Prediction using a trained machine learning model.

# Features
Interactive UI: Built using Streamlit for simplicity and usability.
Real-time Spam Classification: Enter a message to instantly check if it’s spam.
Machine Learning Model: Pretrained model for accurate classification.

# Setup and Installation
Clone the repository:

bash
git clone <repository-link>
cd <repository-folder>
Install dependencies: Ensure Python 3.7+ is installed, then run:

bash
pip install -r requirements.txt
Download NLTK data: Ensure the required NLTK datasets are available:

bash
python -m nltk.downloader punkt stopwords
Add the model and vectorizer files: Place the model.pkl and vectorizer.pkl files in the root directory.

# How to Run the App
Run the Streamlit application:

bash
streamlit run app.py
Open your web browser and go to:

arduino
http://localhost:8501
Input a message into the text box and click the "Predict" button to see the result.

# Project Files
app.py: Streamlit app for user interaction.
model.pkl: Pretrained machine learning model for spam detection.
vectorizer.pkl: TF-IDF vectorizer for feature extraction.
Spam-detection.ipynb: Jupyter notebook with the model training and evaluation process.
spam.csv: Dataset used for training and testing the model.

# Technologies Used
Programming Language: Python
Libraries:
Streamlit (for the web app)
NLTK (for NLP)
Scikit-learn (for machine learning)
Pickle (for saving/loading the model)
Machine Learning Techniques: TF-IDF Vectorization, Logistic Regression/other ML algorithms.

# Acknowledgments
The dataset was sourced from Kaggle, containing labeled SMS messages for spam classification.
NLTK and Scikit-learn libraries for their robust NLP and ML tools.
