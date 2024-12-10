import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

# 初始化 Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="5706b8395cea48c19b93de9c88953bfc",
    client_secret="05eaeb4142374fec8b96ad839f0af848"
))

# 定义一个函数，搜索播客
def search_podcasts(query, limit, parent_category):
    results = sp.search(q=query, type="show", limit=limit)
    podcasts = []
    for show in results['shows']['items']:
        podcasts.append({
            "Parent_Category": parent_category,  # 大类别
            "Podcast_Name": show['name'],
            "Description": show['description'],
            "Publisher": show['publisher'],
            "Episode_Count": show['total_episodes'],  # 播客的总集数
            "ID": show['id']
        })
    return podcasts

# 批量搜索多个关键词
def batch_search_podcasts(category_dict, limit=10, delay=1):
    all_podcasts = []
    for parent_category in category_dict.keys():
        print(f"Searching podcasts under category: {parent_category}")
        podcasts = search_podcasts(query=parent_category, limit=limit, parent_category=parent_category)
        all_podcasts.extend(podcasts)
        # 添加延迟，防止超出请求限制
        time.sleep(delay)  # 设置延迟
    return all_podcasts

# 保存为 CSV
def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(df)} podcasts to {filename}")

# 定义大类别
category_dict = {
    "Arts & Entertainment": None,
    "Business & Technology": None,
    "Educational": None,
    "Games": None,
    "Lifestyle & Health": None,
    "News & Politics": None,
    "Sports & Recreation": None,
    "True Crime": None
}

# 搜索播客
podcasts = batch_search_podcasts(category_dict, limit=10)
# 保存结果到 CSV
save_to_csv(podcasts, r"D:\Desktop\628 Mdoule4\ID & Category.csv")
