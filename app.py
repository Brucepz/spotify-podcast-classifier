import streamlit as st # type: ignore
import requests
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from wordcloud import WordCloud
from configparser import ConfigParser  # 替代 SafeConfigParser
import matplotlib.pyplot as plt
import base64

# Spotify API Configuration
CLIENT_ID = "cdbe1048ac9243c09caa8d7a369b929d"
CLIENT_SECRET = "d0d510e0bfc54ebfa2bd084dc0c59e26"

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

# Function to search for podcasts
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

# Function to get episodes of a podcast
def get_episodes(show_id, token):
    url = f"https://api.spotify.com/v1/shows/{show_id}/episodes"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"limit": 50}
    response = requests.get(url, headers=headers, params=params)
    return response.json().get("items", [])

# Function to perform LDA topic modeling and generate a word cloud
def lda_topic_modeling(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    cleaned_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    cleaned_text = " ".join(cleaned_tokens)

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    X_bow = vectorizer.fit_transform([cleaned_text])

    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda.fit(X_bow)

    # Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(cleaned_text)
    return wordcloud, vectorizer.get_feature_names_out()

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

def find_similar_podcast_lda(lda_result, token):
    """
    根据 LDA 的分析结果，找到最相似的 Podcast
    示例实现：返回模拟数据
    """
    return {
        "name": "LDA Recommended Podcast",
        "publisher": "LDA Example Publisher",
        "description": "This is a description of the podcast recommended by LDA.",
        "release_date": "2024-12-01",
        "duration": "30 minutes and 15 seconds"
    }

def find_similar_podcast_tfd(tfd_result, token):
    """
    根据 TFD 的分析结果，找到最相似的 Podcast
    示例实现：返回模拟数据
    """
    return {
        "name": "TFD Recommended Podcast",
        "publisher": "TFD Example Publisher",
        "description": "This is a description of the podcast recommended by TFD.",
        "release_date": "2024-11-30",
        "duration": "25 minutes and 40 seconds"
    }

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

                # Convert duration to minutes and seconds
                minutes = total_duration_seconds // 60
                seconds = total_duration_seconds % 60
                duration = f"{minutes} minutes and {seconds} seconds"

                # 示例 LDA 和 TFD 分析结果
                lda_result = "LDA Analysis Result"
                tfd_result = "TFD Analysis Result"

                # 分别推荐最相似的 Podcast
                lda_recommendation = find_similar_podcast_lda(lda_result, token)
                tfd_recommendation = find_similar_podcast_tfd(tfd_result, token)

                # 添加选项卡显示 LDA 和 TFD 分析
                tabs = st.tabs(["LDA Analysis", "TFD Analysis"])

                with tabs[0]:
                    st.header("LDA Analysis")
                    st.write("LDA analysis results will be displayed here.")
                    st.write(f"**Analyzed Podcast**: {name}")
                    st.write(f"**Duration**: {duration}")
                    st.write(f"**Description**: {description}")

                    # 推荐 Podcast (LDA)
                    st.subheader("Recommended Podcast by LDA")
                    st.write(f"**Name**: {lda_recommendation['name']}")
                    st.write(f"**Publisher**: {lda_recommendation['publisher']}")
                    st.write(f"**Release Date**: {lda_recommendation['release_date']}")
                    st.write(f"**Duration**: {lda_recommendation['duration']}")
                    st.write(f"**Description**: {lda_recommendation['description']}")

                with tabs[1]:
                    st.header("TFD Analysis")
                    st.write("TFD analysis results will be displayed here.")
                    st.write(f"**Analyzed Podcast**: {name}")
                    st.write(f"**Duration**: {duration}")
                    st.write(f"**Description**: {description}")

                    # 推荐 Podcast (TFD)
                    st.subheader("Recommended Podcast by TFD")
                    st.write(f"**Name**: {tfd_recommendation['name']}")
                    st.write(f"**Publisher**: {tfd_recommendation['publisher']}")
                    st.write(f"**Release Date**: {tfd_recommendation['release_date']}")
                    st.write(f"**Duration**: {tfd_recommendation['duration']}")
                    st.write(f"**Description**: {tfd_recommendation['description']}")
