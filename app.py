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
# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Load pre-trained models and vectorizers
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("lda_vectorizer.pkl", "rb") as f:
    lda_vectorizer = pickle.load(f)

with open("lda_model.pkl", "rb") as f:
    lda = pickle.load(f)

# Load the preprocessed dataset
df = pd.read_csv("Results.csv")
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

    return {
        "TF-IDF Recommendation": tfidf_recommendation,
        "LDA Recommendation": lda_recommendation
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

# Streamlit app


def get_episode_details(episode_id, token):    #TODO
    url = f"https://api.spotify.com/v1/episodes/{episode_id}"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch episode details. Status Code: {response.status_code}")
        return None



# Streamlit App Interface
# Layout for inputs and outputs

st.set_page_config(layout="wide")

# 添加 CSS 移除页面内边距
st.markdown(
    """
    <style>
    .css-18e3th9 {
        padding-left: 0rem;
        padding-right: 0rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
def find_similar_podcast_lda(lda_result, token):  #TODO
    """
    根据 LDA 推荐的 episode ID 获取相应的 episode 信息。

    :param lda_result: 推荐的 episode ID（字符串）
    :param token: Spotify API 的访问令牌
    :return: 包含推荐 episode 详细信息的字典
    """
    return get_episode_details(lda_result, token)


def find_similar_podcast_tfd(tfd_result, token):
    """
    根据 TF-IDF 推荐的 episode ID 获取相应的 episode 信息。

    :param tfd_result: 推荐的 episode ID（字符串）
    :param token: Spotify API 的访问令牌
    :return: 包含推荐 episode 详细信息的字典
    """
    return get_episode_details(tfd_result, token)


with st.container():
    col1, col2 = st.columns([1, 6])  # 调整列比例
    with col1:
        st.header("Inputs")
        token = get_spotify_token()
        if token:
            # 输入 Podcast Name
            podcast_name = st.text_input("Podcast Name")
            if podcast_name:
                # 搜索相关 Podcasts
                podcasts = search_podcasts(podcast_name, token)
                if podcasts:
                    # 显示下拉选择框
                    options = [f"{p['name']} by {p['publisher']}" for p in podcasts]
                    selected_podcast = st.selectbox("Select a Podcast", options)
                    selected_index = options.index(selected_podcast)
                    selected_podcast_data = podcasts[selected_index]

                    # 将按钮放在下拉框下面
                    if st.button("Analyze Podcast", key="analyze_podcast_button"):
                        st.success(f"Podcast Found: {selected_podcast_data['name']} by {selected_podcast_data['publisher']}")
                else:
                    selected_podcast_data = None
                    st.error("No related podcasts found.")
            else:
                selected_podcast_data = None
        else:
            st.error("Failed to authenticate with Spotify API.")
            selected_podcast_data = None

    # Right Panel for Title and Outputs
    with col2:
        st.title("Spotify Podcast Analysis & Recommendations")
        if selected_podcast_data:
            st.write("Analyzing podcast and recommending similar content...")

            episodes = get_episodes(selected_podcast_data["id"], token)
            if not episodes:
                st.error(f"No episodes found for podcast '{selected_podcast_data['name']}'.")
            else:
                # 获取 Podcast 的第一个 Episode
                episode = episodes[0]
                description = episode['description']
                name = episode['name']
                release_date = episode['release_date']
                total_duration_seconds = episode['duration_ms'] // 1000  # Convert milliseconds to seconds

                # 调用推荐函数
                recommendations = recommend_podcast(description, tfidf_matrix, lda_distribution, df)   #TODO

                # Convert duration to minutes and seconds
                minutes = total_duration_seconds // 60
                seconds = total_duration_seconds % 60
                duration = f"{minutes} minutes and {seconds} seconds"

                # 示例 LDA 和 TFD 分析结果
                lda_result = recommendations['LDA Recommendation']    #TODO
                tfd_result = recommendations['TF-IDF Recommendation']

                # 分别推荐最相似的 Podcast
                lda_recommendation = find_similar_podcast_lda(lda_result, token)     
                tfd_recommendation = find_similar_podcast_tfd(tfd_result, token)
      
                # 添加选项卡显示 LDA 和 TFD 分析
                tabs = st.tabs(["LDA Analysis", "TFD Analysis"])

                with tabs[0]:
                    st.header("LDA Analysis")
                    st.write("LDA analysis results will be displayed here.")
                    st.write(f"**Analyzed Episode**: {name}")
                    st.write(f"**Duration**: {duration}")
                    
                    st.write(f"**Description**: {description}")
                    #能否加上podcast name？

                    # 推荐 Podcast (LDA)
                    st.subheader("Recommended Podcast by LDA")
                    st.write(f"**Name**: {lda_recommendation['name']}")
            #        st.write(f"**Publisher**: {lda_recommendation['publisher']}")
                    st.write(f"**Release Date**: {lda_recommendation['release_date']}")
                    st.write(f"**Duration**: {lda_recommendation['duration_ms']}")  
                    st.write(f"**Description**: {lda_recommendation['description']}")
                    st.write(f"**Link**: https://open.spotify.com/episode/{lda_result}")

                with tabs[1]:
                    st.header("TFD Analysis")
                    st.write("TFD analysis results will be displayed here.")
                    st.write(f"**Analyzed Podcast**: {name}")
                    st.write(f"**Duration**: {duration}")
                    st.write(f"**Description**: {description}")

                    # 推荐 Podcast (TFD)
                    st.subheader("Recommended Podcast by TFD")
                    st.write(f"**Name**: {tfd_recommendation['name']}")
                #    st.write(f"**Publisher**: {tfd_recommendation['publisher']}")
                    st.write(f"**Release Date**: {tfd_recommendation['release_date']}")
                    st.write(f"**Duration**: {tfd_recommendation['duration_ms']}")
                    st.write(f"**Description**: {tfd_recommendation['description']}")
                    st.write(f"**Link**: https://open.spotify.com/episode/{tfd_result}")
