import streamlit as st # type: ignore
import requests
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from configparser import ConfigParser  # 替代 SafeConfigParser
import matplotlib.pyplot as plt
import base64
import pandas as pd
import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
import os
# Load SpaCy model
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
  
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Load pre-trained models and vectorizers
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("lda_vectorizer.pkl", "rb") as f:
    lda_vectorizer = pickle.load(f)

with open("lda_model.pkl", "rb") as f:
    lda = pickle.load(f)


# 读取分割文件并合并
def load_split_results(directory):
    all_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.startswith("Results_part_")]
    all_files.sort()  # 按顺序读取
    df_list = [pd.read_csv(file) for file in all_files]
    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

# 加载分割数据
split_results_dir = "split_results"
df = load_split_results(split_results_dir)


tfidf_matrix = np.array([list(map(float, row.split(','))) for row in df['TFIDF_Matrix']])
lda_distribution = np.array([list(map(float, row.split(','))) for row in df['LDA_Distribution']])


# Spotify API Configuration
CLIENT_ID = "cdbe1048ac9243c09caa8d7a369b929d"
CLIENT_SECRET = "d0d510e0bfc54ebfa2bd084dc0c59e26"

# Function to clean and tokenize descriptions
def clean_and_tokenize(description):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'\b(visit|podcast|com|episode)\b', '', text)  # Remove specific words
        text = re.sub(r'[\W_]+', ' ', text)  # Remove special characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text

    def tokenize(text):
        doc = nlp(text)
        return ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

    return tokenize(clean_text(description))

# Function to recommend podcasts
def recommend_podcast(description, tfidf_matrix, lda_distribution, df):
    # Clean and tokenize input description
    processed_desc = clean_and_tokenize(description)

    # TF-IDF feature computation
    tfidf_vector = tfidf_vectorizer.transform([processed_desc]).toarray()

    # LDA feature computation
    lda_bow = lda_vectorizer.transform([processed_desc])
    lda_vector = lda.transform(lda_bow)

    # Similarity computation
    tfidf_similarities = cosine_similarity(tfidf_vector, tfidf_matrix).flatten()
    lda_similarities = cosine_similarity(lda_vector, lda_distribution).flatten()

    # Find the most similar podcast indices
    tfidf_most_similar_index = tfidf_similarities.argmax()
    lda_most_similar_index = lda_similarities.argmax()

    # Retrieve the recommended podcast IDs
    tfidf_recommendation = df.iloc[tfidf_most_similar_index]["Episode_ID"]
    lda_recommendation = df.iloc[lda_most_similar_index]["Episode_ID"]
    tfidf_recommendation_topic = df.iloc[tfidf_most_similar_index]["Dominant_Topic"]
    lda_recommendation_topic = df.iloc[lda_most_similar_index]["Dominant_Topic"]
    return {
        "TF-IDF Recommendation": tfidf_recommendation,
        "LDA Recommendation": lda_recommendation,
        "TF-IDF Topic": tfidf_recommendation_topic,
        "LDA Topic": lda_recommendation_topic
    }

# Function to get Spotify Access Token
def get_spotify_token():
    url = "https://accounts.spotify.com/api/token"
    auth_header = f"{CLIENT_ID}:{CLIENT_SECRET}"
    headers = {
        "Authorization": f"Basic {base64.b64encode(auth_header.encode()).decode()}",
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        return None

def search_podcasts(podcast_name, token):
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": podcast_name, "type": "show", "limit": 6}
    response = requests.get(url, headers=headers, params=params)
    results = response.json()
    if "shows" in results and results["shows"]["items"]:
        return [
            {"id": show["id"], "name": show["name"], "publisher": show["publisher"]}
            for show in results["shows"]["items"]
        ]
    return []

def get_episodes(show_id, token):
    url = f"https://api.spotify.com/v1/shows/{show_id}/episodes"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"limit": 50}
    response = requests.get(url, headers=headers, params=params)
    return response.json().get("items", [])



def get_episode_details(episode_id, token):    #TODO
    url = f"https://api.spotify.com/v1/episodes/{episode_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch episode details. Status Code: {response.status_code}")
        return None
def find_similar_podcast_lda(lda_result, token):
    """Retrieve episode details based on LDA recommendation."""
    return get_episode_details(lda_result, token)

def find_similar_podcast_tfd(tfd_result, token):
    """Retrieve episode details based on TF-IDF recommendation."""
    return get_episode_details(tfd_result, token)
def format_duration(duration_ms):
    """
    Formats a duration given in milliseconds into a human-readable string.
    
    :param duration_ms: Duration in milliseconds (int)
    :return: Formatted duration as a string (e.g., "5 minutes and 30 seconds")
    """
    total_seconds = duration_ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes} minutes and {seconds} seconds"


# Streamlit app
# Set wide layout
# Streamlit App Interface
# Layout for inputs and outputs
# Streamlit App Interface
# Layout for inputs and outputs





@st.cache_data
def cached_get_episodes(show_id, token):
    return get_episodes(show_id, token)


# Set page layout
st.set_page_config(layout="wide")

# Initialize session state for page navigation
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"  # Default to home page

# Add navigation bar
st.markdown(
    """
    <style>
     
    [data-testid="stSidebar"] {
        min-width: 150px; /* Adjust the sidebar width */
        max-width: 150px; /* Adjust the sidebar width */
    }
    unsafe_allow_html=True,
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #f8f9fa;
        padding: 10px 20px;
        border-bottom: 1px solid #ddd;
        font-family: Arial, sans-serif;
    }
    .logo {
        display: flex;
        align-items: center;
    }
    .logo img {
        height: 40px;
        margin-right: 10px;
    }
    .logo span {
        font-size: 24px;
        font-weight: bold;
        color: #000; /* Black color for text */
    }
    .nav-links {
        display: flex;
        gap: 20px;
    }
    .nav-links button {
        background: none;
        border: none;
        color: #555;
        font-weight: bold;
        cursor: pointer;
        font-size: 16px;
    }
    .nav-links button:hover {
        color: #000;
    }
    </style>
    <div class="top-bar">
        <div class="logo">
            <img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Spotify_logo_with_text.svg" alt="Spotify">
            <span>Spotify Podcast Recommender</span>
        </div>
  
    </div>
    """,
    unsafe_allow_html=True,
)

# Navigation bar buttons
with st.sidebar:
    if st.button("Home"):
        st.session_state["current_page"] = "home"
    if st.button("About"):
        st.session_state["current_page"] = "about"
    if st.button("Contact"):
        st.session_state["current_page"] = "contact"

# Function to render the Home page
def render_home_page():
   # st.title("Spotify Podcast Recommender")
   # st.write("Analyzing podcast and recommending similar content...")

    col1, col2, col3 = st.columns([1, 4, 1])  # Add a third column for the image
    with col1:
        st.header("Search here")
        token = get_spotify_token()  # Assume function exists
        selected_podcast_data = None

        if token:
            # Input Podcast or Episode Name
            podcast_name = st.text_input("What do you want?")
            if podcast_name:
                podcasts = search_podcasts(podcast_name, token)[:10]
                if podcasts:
                    options = [f"{p['name']} by {p['publisher']}" for p in podcasts]
                    selected_podcast = st.selectbox("Select a podcast or episode", options)
                    selected_index = options.index(selected_podcast)
                    selected_podcast_data = podcasts[selected_index]
                else:
                    st.error("No related podcasts or episodes found.")
            else:
                st.warning("Enter a podcast or episode name to search.")
        else:
            st.error("Failed to authenticate with Spotify API.")

        # Trigger analysis with a button
        if selected_podcast_data and st.button("Analyze Episode", key="analyze_episode_button"):
            episodes = get_episodes(selected_podcast_data["id"], token)
            if episodes:
                selected_episode_data = next(
                    (ep for ep in episodes if podcast_name.lower() in ep["name"].lower()),
                    episodes[0],
                )
                st.session_state["selected_episode_data"] = selected_episode_data
            else:
                st.error("No episodes found for the selected podcast.")

    with col2:
        st.title("Selected Podcast")
        if "selected_episode_data" in st.session_state:
            selected_episode_data = st.session_state["selected_episode_data"]
            description = selected_episode_data["description"]
            name = selected_episode_data["name"]
            release_date = selected_episode_data["release_date"]
            total_duration_seconds = selected_episode_data["duration_ms"] // 1000
            minutes = total_duration_seconds // 60
            seconds = total_duration_seconds % 60
            duration = f"{minutes} minutes and {seconds} seconds"
            podcast_link = selected_episode_data.get("external_urls", {}).get("spotify", "No link available")

            st.subheader("Information:")
            st.write(f"**Analyzed Episode**: {name}")
            st.write(f"**Duration**: {duration}")
            st.write(f"**Description**: {description}")
            
            if podcast_link != "No link available":
                st.write(f"**Link**: [Listen on Spotify]({podcast_link})")
            else:
                st.write(f"**Link**: {podcast_link}")  # Show "No link available" if the link is missing


            recommendations = recommend_podcast(description, tfidf_matrix, lda_distribution, df)
            lda_result = recommendations["LDA Recommendation"]
            tfd_result = recommendations["TF-IDF Recommendation"]
            lda_topic = recommendations['LDA Topic']
            tdf_topic = recommendations['TF-IDF Topic']
            lda_recommendation = find_similar_podcast_lda(lda_result, token)
            tfd_recommendation = find_similar_podcast_tfd(tfd_result, token)

            lda_duration = format_duration(lda_recommendation["duration_ms"])
            tfd_duration = format_duration(tfd_recommendation["duration_ms"])

            tabs = st.tabs(["LDA Analysis", "TF-IDF Analysis"])

            with tabs[0]:
                st.header("Recommended Episode by LDA")
                col_lda_text, col_lda_image = st.columns([3, 1])  # Split into text and image columns
                with col_lda_text:
                    st.write(f"**Name**: {lda_recommendation['name']}")
                    st.write(f"**Duration**: {lda_duration}")
                    st.write(f"**Description**: {lda_recommendation['description']}")
                    st.write(f"**Link**:https://open.spotify.com/episode/{lda_result}")
                    wordcloud_path = f"Image/{lda_topic}.png"
                    st.subheader("Word Cloud for LDA Topic")
                    st.image(wordcloud_path, use_container_width=True)
                with col_lda_image:
                    lda_images = lda_recommendation.get("images", [])
                    lda_image_url = lda_images[0]["url"] if lda_images and isinstance(lda_images[0], dict) and "url" in lda_images[0] else None
                    if lda_image_url:
                        st.image(lda_image_url, use_container_width=True)
                    else:
                        st.write("No image available.")

            with tabs[1]:
                st.header("Recommended Episode by TF-IDF")
                col_tfd_text, col_tfd_image = st.columns([3, 1])  # Split into text and image columns
                with col_tfd_text:
                    st.write(f"**Name**: {tfd_recommendation['name']}")
                    st.write(f"**Duration**: {tfd_duration}")
                    st.write(f"**Description**: {tfd_recommendation['description']}")
                    st.write(f"**Link**:https://open.spotify.com/episode/{tfd_result}")
                    wordcloud_path = f"Image/{tdf_topic}.png"
                    st.subheader("Word Cloud for LDA Topic")
                    st.image(wordcloud_path, use_container_width=True)
                with col_tfd_image:
                    tfd_images = tfd_recommendation.get("images", [])
                    tfd_image_url = tfd_images[0]["url"] if tfd_images and isinstance(tfd_images[0], dict) and "url" in tfd_images[0] else None
                    if tfd_image_url:
                        st.image(tfd_image_url, use_container_width=True)
                    else:
                        st.write("No image available.")

    with col3:
        if "selected_episode_data" in st.session_state:
            images = selected_episode_data.get("images", [])
            image_url = images[0]["url"] if images and isinstance(images[0], dict) and "url" in images[0] else None
            if image_url:
                st.image(image_url, use_container_width=True)
            else:
                st.write("No image available for this episode.")

# Function to render the About page
def render_about_page():
    st.title("About:")
    st.write("     ")
    st.subheader("Intro")
    st.write("This is the Group Project of STAT628 Module4, built by Zekai Xu and Zhenke Peng")
    st.write(" In this module, we are working on the data from Spotify, the top leading platform for music and podcasting streaming, through its API. Our project is focusing on creating insightful metrics for podcast episode to help in clustering and recommendation task from data in Spotify. In addition, we will establish a web app to visualize the metrics and the recommendation for new podcast episode.    ")
    st.subheader("Approach")
    st.write("First, about 80,000 podcast episodes from Spotify were scraped and their metadata including names, ID and descriptions were collected. ")
    st.write("     ")  
    st.write("Next, we preprocessed the descriptions of these episodes by cleaning the text (removing URLs, punctuation, and converting text to lowercase) and tokenized the cleaned text using the 'en_core_web_sm' model from the SpaCy library. Then we constructed two metrics, LDA and TF-IDF, to analyze the descriptions. These metrics were used to generate corresponding feature matrices.  ")
    st.write("     ")
    st.write("Finally, we built an interactive app using Streamlit. The app allows users to search for a podcast and select one episode, fetches its description in real-time via the Spotify API. Features would be extracted from the selected description through two metrics and computer cosine similarity with the precomputed feature matrices to identify and recommend the nearest episodes based on both LDA and TF-IDF similarity scores. ")
    st.write("     ")
    st.subheader("Overall workflow of the project:")
    st.image("Image/Workflow.png", use_container_width=True)  # Display the workflow image

# Function to render the Contact page
def render_contact_page():
    st.title("Contact Us")
    st.subheader("website feedback, questions or accessibility issues:  zpeng66@wisc.edu")
    st.write("     ")
    st.subheader("Learn more about the project: https://github.com/Brucepz/spotify-podcast-classifier")
   

# Route pages based on current_page
if st.session_state["current_page"] == "home":
    render_home_page()
elif st.session_state["current_page"] == "about":
    render_about_page()
elif st.session_state["current_page"] == "contact":
    render_contact_page()
