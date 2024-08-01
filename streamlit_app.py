import streamlit as st
import spacy
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import re
import emoji
import nltk
from nltk.corpus import stopwords
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from googletrans import Translator

# Initialize NLP and other resources
nlp = spacy.load('en_core_web_lg')
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
classifier = pipeline('text-classification', model="parsawar/profanity_model_3.1")

# Load the vector data
vector_csv_path = '.\vector_data.csv'

def load_csv_with_fallback(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin-1')
    return df

vector_df = load_csv_with_fallback(vector_csv_path)

def load_words(file_path):
    with open(file_path, 'r') as file:
        return set(line.strip().lower() for line in file)

forbidden_words_path = '.\abusive_words.txt'
forbidden_words = load_words(forbidden_words_path)

def remove_special_characters(text):
    pattern = r'(?<!\w)[^\w\s](?!\w)'
    return re.sub(pattern, '', text)

def convert_to_lowercase(text):
    return text.lower()

def convert_emojis(text):
    return emoji.demojize(text)

def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

def preprocess_text(text):
    text = remove_special_characters(text)
    text = convert_to_lowercase(text)
    text = convert_emojis(text)
    text = remove_stopwords(text)
    return text

def transliterate_and_translate(text):
    transliterated_text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    translator = Translator()
    translation = translator.translate(transliterated_text, src='hi', dest='en')
    return transliterated_text, translation.text

def contains_forbidden_word(input_text, forbidden_words):
    words = input_text.split()
    return any(word.lower() in forbidden_words for word in words)

def process_user_input(user_input, vector_df=vector_df, nlp=nlp, forbidden_words=forbidden_words):
    transliterated_text = None
    translated_text = None

    def save_entries(entries):
        new_entries_df = pd.DataFrame(entries)
        vector_df_updated = pd.concat([vector_df, new_entries_df], ignore_index=True)
        vector_df_updated.to_csv(vector_csv_path, index=False)
        return vector_df_updated

    if re.search('[\u0900-\u097F]', user_input):
        transliterated_text, translated_text = transliterate_and_translate(user_input)
        translated_for_processing = translated_text

        original_vector = nlp(user_input).vector
        entries = [{
            "cleaned_text": user_input,
            "label": 1,
            "vector": ','.join(map(str, original_vector))
        }]

        vector_df = save_entries(entries)
    else:
        translated_for_processing = user_input

    if contains_forbidden_word(translated_for_processing, forbidden_words):
        return None, None, None, None

    cleaned_user_input = preprocess_text(translated_for_processing)
    user_doc = nlp(cleaned_user_input)
    user_vector = user_doc.vector

    vectors = vector_df['vector'].apply(lambda x: np.array([float(i) for i in x.split(',')]) if isinstance(x, str) else np.zeros(len(user_vector))).values
    vectors = np.stack(vectors)

    similarities = cosine_similarity([user_vector], vectors)

    most_similar_index = similarities[0].argmax()
    most_similar_entry = vector_df.iloc[most_similar_index]
    similarity_score = similarities[0][most_similar_index]

    threshold = 0.8

    if (similarity_score >= threshold):
        assigned_label = most_similar_entry['label']
        hf_label_score = None
        assigned_based_on_similarity = True
    else:
        result = classifier(cleaned_user_input)
        classified_label = result[0]['label']
        hf_label_score = result[0]['score']

        assigned_label = classified_label
        assigned_based_on_similarity = False

    new_vector_entry = {
        "cleaned_text": translated_for_processing,
        "label": assigned_label,
        "vector": ','.join(map(str, user_vector))
    }

    new_vector_entry_df = pd.DataFrame([new_vector_entry])
    vector_df = pd.concat([vector_df, new_vector_entry_df], ignore_index=True)

    vector_df.to_csv(vector_csv_path, index=False)

    return similarity_score, assigned_label, hf_label_score, assigned_based_on_similarity

def main():
    st.title("Text Profanity Detection")

    text_input = st.text_area("Enter text:", "")

    if st.button("Submit"):
        if not text_input:
            st.warning("Please enter some text.")
        else:
            similarity_score, assigned_label_similarity, hf_label_score, assigned_based_on_similarity = process_user_input(text_input, vector_df, nlp, forbidden_words)
            
            if similarity_score is None:
                st.error("Your text contains restricted words. Please remove them and try again.")
            else:
                if assigned_based_on_similarity:
                    st.success(f"Label: {assigned_label_similarity}")
                    st.info(f"Score: {similarity_score}")
                    st.info(f"Determined using: vector similarity")
                else:
                    st.success(f"Label: {assigned_label_similarity}")
                    st.info(f"Score: {hf_label_score}")
                    st.info(f"Determined using: Hugging Face model")

if __name__ == '__main__':
    main()
