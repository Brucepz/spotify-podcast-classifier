import pandas as pd
import re
import spacy


input_csv = r"D:\Desktop\628 Mdoule4\episode_detail.csv"  
output_csv = r"D:\Desktop\628 Mdoule4\Cleaned_tokenized.csv" 
df = pd.read_csv(input_csv)

# clean
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  
    text = re.sub(r'http\S+|www\.\S+', '', text)  
#    text = re.sub(r'\b(visit|podcast|com|episode)\b', '', text)  
    text = re.sub(r'[\W_]+', ' ', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

df['Cleaned_Description'] = df['Description'].apply(clean_text)

# tokenize
nlp = spacy.load("en_core_web_sm")
def tokenize(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

df['Tokenized_Description'] = df['Cleaned_Description'].apply(tokenize)


df.to_csv(output_csv, index=False, encoding="utf-8")
print(f"Cleaned and tokenized descriptions saved to {output_csv}")
