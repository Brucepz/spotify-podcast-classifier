import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="5706b8395cea48c19b93de9c88953bfc",
    client_secret="05eaeb4142374fec8b96ad839f0af848"
))


def fetch_episodes(show_id, parent_category, limit=50, delay=1):
    episodes = []
    offset = 0
    while True:
        try:
            results = sp.show_episodes(show_id, limit=limit, offset=offset)
            for episode in results['items']:
                episodes.append({
                    "Podcast_ID": show_id,
                    "Parent_Category": parent_category,
                    "Episode_ID": episode['id'],
                    "Episode_Name": episode['name'],
                    "Description": episode['description'],
                    "Release_Date": episode['release_date'],
                    "Is_Explicit": episode['explicit'],
                    "Language": episode['language'],
                    "Available_Markets_Count": len(episode.get('available_markets', [])),
                    "External_URL": episode['external_urls'].get('spotify', '')
                })
            if len(results['items']) < limit:
                break
            offset += limit
            time.sleep(delay)
        except Exception as e:
            print(f"Error fetching episodes for Podcast ID {show_id}: {e}")
            break
    return episodes

def load_podcast_info(filename):
    df = pd.read_csv(filename) 
    return df[["ID", "Parent_Category"]]


def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"Saved {len(df)} episodes to {filename}")


def main(podcast_csv, output_csv):
    podcast_info = load_podcast_info(podcast_csv)
    all_episodes = []
    for index, row in podcast_info.iterrows():
        show_id = row['ID']
        parent_category = row['Parent_Category']
        print(f"Fetching episodes for Podcast ID: {show_id} (Parent Category: {parent_category})")
        episodes = fetch_episodes(show_id, parent_category)
        all_episodes.extend(episodes)
    save_to_csv(all_episodes, output_csv)


if __name__ == "__main__":
    podcast_csv = r"D:\Desktop\628 Mdoule4\ID & Category.csv" 
    output_csv = r"D:\Desktop\628 Mdoule4\episode_detail.csv"  
    main(podcast_csv, output_csv)
