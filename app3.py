import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load vectorizer and models
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model_rec = joblib.load("model_recommend_LogReg.joblib")
model_rating = joblib.load("model_rating_XGBRegressor.joblib")

st.title("🛍️ Women's Clothing Review Assistant")

st.markdown("Enter a product review below. We'll predict whether the customer would recommend it and what rating they might give.")

review_input = st.text_area("✍️ Review Text", "Love this product! Super comfortable and fits perfectly.")

if st.button("Predict"):
    vec_input = vectorizer.transform([review_input])

    rec_pred = model_rec.predict(vec_input)[0]
    rate_pred = model_rating.predict(vec_input)[0]

    st.markdown(f"**🔁 Recommend Prediction:** {'Yes' if rec_pred==1 else 'No'}")
    st.markdown(f"**⭐ Predicted Rating:** {round(rate_pred, 2)}")

    # Wordcloud
    wc = WordCloud(width=600, height=300, background_color='white').generate(review_input)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt.gcf())

st.markdown("---")
st.markdown("📌 Tip: Try writing positive and negative reviews to see how the model responds!")
