import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Website Title
st.title("📧 Smart Spam Email Classifier")

st.write("Enter email text below to check category.")

# User input
email_text = st.text_area("Enter Email Text")

if st.button("Predict"):

    if email_text.strip() == "":
        st.warning("Please enter email text.")
    else:
        # Transform input
        data = vectorizer.transform([email_text])

        # Prediction
        result = model.predict(data)[0]

        st.success(f"Category: {result}")

        # Simple risk logic
        if result != "Normal Mail":
            st.error("⚠️ This mail looks suspicious!")
        else:
            st.info("✅ This email looks safe.")
# python train_model.py
#python -m streamlit --version
#python -m streamlit run app.py
# pip install streamlit pandas scikit-learn
#pip install streamlit  
#python -m pip install streamlit