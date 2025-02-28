import pandas as pd
import numpy as np
import re
import string
import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load Dataset
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data.iloc[:, :2]  # Keeping only relevant columns
data.columns = ['label', 'message']

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

data['message'] = data['message'].apply(preprocess_text)

# Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
print(f"Model Used: RandomForestClassifier")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")

# Save Model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

# Streamlit App
def main():
    st.title("📩 SMS Spam Classifier")
    st.markdown("""
    **🔹 Enter a message below and find out if it's spam or not!**
    """)
    st.divider()
    
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
    
    # User Input
    user_input = st.text_area("📨 Enter your SMS:")
    example_text = "Example: Congratulations! You've won a lottery. Click here to claim."
    st.markdown(f"*{example_text}*")
    
    if st.button("🚀 Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid SMS.")
        else:
            processed_input = preprocess_text(user_input)
            vectorized_input = vectorizer.transform([processed_input])
            prediction = model.predict(vectorized_input)[0]
            
            st.divider()
            if prediction == 1:
                st.error("📛 **Spam Detected!** This message is likely spam.")
            else:
                st.success("✅ **Safe Message!** This message is not spam.")
    
    st.divider()
    st.markdown("💡 *Built using Machine Learning & Streamlit* 🚀")

if __name__ == "__main__":
    main()
