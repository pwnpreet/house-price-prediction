import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import joblib
import nltk


from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import ngrams, pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer


ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://i.imgur.com/y2vnPN3.jpeg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#  Load models --------------------
lr = joblib.load("lr_model.pkl")
Ridge = joblib.load("Ridge_model.pkl")
lasso = joblib.load("lasso_model.pkl")
scaled = joblib.load("scaled.pkl")
corpus = joblib.load("corpus.pkl")  
portstem = joblib.load("ps.pkl") 


# App Title --------------------
st.title("🏠 House Price Prediction App")

# Loading dataset
from sklearn.datasets import fetch_california_housing
house= fetch_california_housing()
try:
    data= fetch_california_housing(as_frame= True)
    df= data.frame
    st.success("✅ Data loaded succesfully!")
except:
    st.error("😕 Data not found")
    st.stop()

# sidebar section
section = st.sidebar.radio("Select Dataset Section", ["Dataset Preview", "Dataset Information", "Numerical Information"])

if section == "Dataset Preview":
    view_options= st.sidebar.radio("Select Show to View Data", ["Hide", "Show"])
    if view_options == "Show":
        st.sidebar.subheader("👀 Dataset Preview")
        st.sidebar.write(df.head())
elif section == "Dataset Information":
    st.sidebar.subheader("📝 Dataset Information")
    col1, col2= st.sidebar.columns(2)
    col1.metric(label="Number of rows", value=df.shape[0])
    col2.metric(label="Number of columns", value=df.shape[1])
elif section == "Numerical Information":
    with st.sidebar.expander("📊 Numerical description of data", expanded= False):
        st.write(df.describe()) 


# chatbot section 
st.markdown("---")
st.subheader("💬 ChatBot Assistant")

user_input = st.text_input("Ask something about the models or dataset:")

def chatbot_response(user_text):
    text = user_text.lower()
    for q, a in corpus:
        words = q.lower().split()[:2]   
        if all(w in text for w in words):
            return a
    return dict(corpus).get("default", "Sorry, I didn't get that.")

if st.button("Ask"):
    response = chatbot_response(user_input)
    st.session_state["last_response"] = response
    st.success(f"🤖: {response}")


credentials = {
    "admin": "a",
    "manager": "m",
    "trainer": "t"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in= False   

if not st.session_state.logged_in:
    # Login Section 
    st.subheader("🔒 Login Required!")
    username= st.text_input("Enter your username")
    password= st.text_input("Enter your password", type="password")
    btn= st.button("Login")

    if btn:
        if username in credentials and credentials[username] == password:
            st.success(f"😀 welcome, {username}") 
            st.session_state.logged_in= True
        else:
            st.error("😕 Invalid username or password") 

if st.session_state.logged_in:
    st.write("🎉 Now you have access to complete data") 

    # FAQs section 
    st.subheader("📝 Data Analysis FAQ's") 

    # ques1
    with st.expander("Q1: How are median income values distributed?"):
        st.write("Most districts have a median income between 2–5")
        fig1 = px.histogram(df, x="MedInc", nbins=20, title="Distribution by Ocean Proximity")
        fig1.update_layout(paper_bgcolor= 'rgba(0,0,0,0)', plot_bgcolor= 'rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width= True)

    # ques2
    with st.expander("Q2: How many data samples and features does the dataset have?"):
        st.write("The dataset contains 20,640 samples and 8 numeric features.")
        data_count= {"Features": 8, "Target": 1}
        fig2 = px.pie(names=data_count.keys(), values=data_count.values(), title="Feature vs Target Ratio")
        fig2.update_layout(paper_bgcolor= 'rgba(0,0,0,0)', plot_bgcolor= 'rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width= True)

    # ques3
    with st.expander("Q3: What is the average median house value in California?"):
        st.write("The average median house value is around $206,000.")
        avg_values = df[['MedInc', 'AveRooms', 'AveOccup', 'MedHouseVal']].mean().reset_index()
        avg_values.columns = ['Feature', 'Average']
        fig3 = px.bar(avg_values, x='Feature', y='Average', title='Average Values of Selected Features')
        fig3.update_layout(paper_bgcolor= 'rgba(0,0,0,0)', plot_bgcolor= 'rgba(0,0,0,0)')
        st.plotly_chart(fig3, use_container_width= True)

    # ques4
    with st.expander("Q4: How are house values distributed across California?"):
        st.write("Most houses have a median value between $100,000 and $250,000.")
        fig4= px.histogram(df, x="MedHouseVal", nbins= 20, title= "Distribution of median House Values")
        fig4.update_layout(paper_bgcolor= 'rgba(0,0,0,0)', plot_bgcolor= 'rgba(0,0,0,0)')
        st.plotly_chart(fig4, use_container_width= True)

    # Prediction section 
    st.subheader("🐾 Predictions")
    st.write("Using Linear, Ridge and Lasso Regression Models")

    #  User Inputs --------------------
    MedInc = st.number_input("Median Income (in 10k)", min_value=0.0)
    HouseAge = st.number_input("House age", min_value=1.0)
    AveRooms = st.number_input("Average Rooms", min_value=1.0)
    AveBedrms = st.number_input("Average Bedrooms", min_value=0.0)
    Population = st.number_input("Population", min_value=1.0)
    AveOccup = st.number_input("Average Occupation", min_value=1.0)
    Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0)
    Longitude = st.number_input("Longitude", min_value=-124.0, max_value=-114.0)

    #  Predictions --------------------
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    features = scaled.transform(features)

    if st.button("🌟Predict House Value"):
        pred_linear = lr.predict(features)[0]
        pred_ridge = Ridge.predict(features)[0]
        pred_lasso = lasso.predict(features)[0]

        st.success(f"Linear Regression Prediction: ${pred_linear*100000:.2f}")
        st.success(f"Ridge Regression Prediction:  ${pred_ridge*100000:.2f}")
        st.success(f"Lasso Regression Prediction:  ${pred_lasso*100000:.2f}")



#  sidebar NLP- Techneques  -------------------------------
st.sidebar.title("🔍 NLP Techniques")
options = st.sidebar.multiselect(
    "Select technique to display:",
    ["Stemmed", "Word Tokenize", "Lemmatized", "Stopword", "Ngrams", "POS Tags"],
)
show_nlp = st.sidebar.button("Show")


def process_text(text):
    words = word_tokenize(text.lower())
    results = {
        "Word Tokenize": " ".join(words),
        "Stemmed": " ".join([f"'{ps.stem(w)}'" for w in words]),
        "Lemmatized": " ".join([f"'{lemmatizer.lemmatize(w)}'" for w in words]),
        "Stopword": " ".join([f"'{w}'" for w in words if w not in stop_words]),
        "Ngrams":  ", ".join([f"({gram[0]}, {gram[1]})" for gram in ngrams(words, 2)]),
        "POS Tags": " ".join([f"[{w:5}->{t}]" for w, t in pos_tag(words)])
    }
    return results

if show_nlp:
    if 'last_response' in st.session_state:
        response = st.session_state['last_response']
        result = process_text(response)

        if options:
            st.sidebar.subheader("🧠 NLP Output")
            for technique in options:
                st.sidebar.info(f"**{technique}:** {result[technique]}")
        else:
            st.sidebar.warning("Please select at least one NLP technique.")
    else:
        st.sidebar.error("❗ Please ask something in chatbot first.")   
