import re
import nltk
import pandas as pd
import emoji
import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from azure.storage.blob import BlobServiceClient
import io

# Download stopwords list if not already present
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stopwords set
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)

# Function to remove standalone special characters
def remove_special_characters(text):
    pattern = r'(?<!\w)[^\w\s](?!\w)'
    return re.sub(pattern, '', text)

# Function to convert text to lowercase
def convert_to_lowercase(text):
    return text.lower()

# Function to convert emojis to textual representation
def convert_emojis(text):
    return emoji.demojize(text)

# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# Function to preprocess text
def preprocess_text(text):
    text = remove_special_characters(text)
    text = convert_to_lowercase(text)
    text = convert_emojis(text)
    text = remove_stopwords(text)
    return text

# Sample list of profane words
PROFANE_WORDS = [
    '2 girls 1 cup', 'anal', 'anus', 'areole', 'arian', 'arrse', 'arse', 'arsehole', 
    'aryan', 'ass', 'ass-fucker', 'assbang', 'assbanged', 'asses', 'assfuck', 'assfucker', 
    'assfukka', 'asshole', 'assmunch', 'asswhole', 'auto erotic', 'autoerotic', 'ballsack', 
    'bastard', 'bdsm', 'beastial', 'beastiality', 'bellend', 'bestial', 'bestiality', 'bimbo', 
    'bimbos', 'bitch', 'bitches', 'bitchin', 'bitching', 'blow job', 'blowjob', 'blowjobs', 
    'blue waffle', 'bondage', 'boner', 'boob', 'boobs', 'booobs', 'boooobs', 'booooobs', 
    'booooooobs', 'booty call','assshole', 'breasts', 'brown shower', 'brown showers', 'buceta', 
    'bukake', 'bukkake', 'bull shit', 'bullshit', 'busty', 'butthole', 'carpet muncher', 
    'cawk', 'chink', 'cipa', 'clit', 'clitoris', 'clits', 'cnut', 'cock', 'cockface', 
    'cockhead', 'cockmunch', 'cockmuncher', 'cocks', 'cocksuck', 'cocksucked', 'cocksucker', 
    'cocksucking', 'cocksucks', 'cokmuncher', 'coon', 'cow-girl', 'cow-girls', 'cowgirl', 
    'cowgirls', 'crap', 'crotch', 'cum', 'cuming', 'cummer', 'cumming', 'cums', 'cumshot', 
    'cunilingus', 'cunillingus', 'cunnilingus', 'cunt', 'cuntlicker', 'cuntlicking', 'cunts', 
    'damn', 'deep throat', 'deepthroat', 'dick', 'dickhead', 'dildo', 'dildos', 'dink', 
    'dinks', 'dlck', 'dog style', 'dog-fucker', 'doggie style', 'doggie-style', 'doggiestyle', 
    'doggin', 'dogging', 'doggy style', 'doggy-style', 'doggystyle', 'dong', 'donkeyribber', 
    'doofus', 'doosh', 'dopey', 'douch3', 'douche', 'douchebag', 'douchebags', 'douchey', 
    'drunk', 'duche', 'dumass', 'dumbass', 'dumbasses', 'dyke', 'dykes', 'eatadick', 
    'eathairpie', 'ejaculate', 'ejaculated', 'ejaculates', 'ejaculating', 'ejaculatings', 
    'ejaculation', 'ejakulate', 'enlargement', 'erect', 'erection', 'erotic', 'erotism', 
    'essohbee', 'extacy', 'extasy', 'f_u_c_k', 'f-u-c-k', 'f.u.c.k', 'f4nny', 'facial', 
    'fack', 'fag', 'fagg', 'fagged', 'fagging', 'faggit', 'faggitt', 'faggot', 'faggs', 
    'fagot', 'fagots', 'fags', 'faig', 'faigt', 'fanny', 'fannybandit', 'fannyflaps', 
    'fannyfucker', 'fanyy', 'fart', 'fartknocker', 'fat', 'fatass', 'fcuk', 'fcuker', 'motherfucker',
    'fcuking', 'feck', 'fecker', 'felch', 'felcher', 'felching', 'fellate', 'fellatio', 
    'feltch', 'feltcher', 'femdom', 'fingerfuck', 'fingerfucked', 'fingerfucker', 
    'fingerfuckers', 'fingerfucking', 'fingerfucks', 'fingering', 'fisted', 'fistfuck', 
    'fistfucked', 'fistfucker', 'fistfuckers', 'fistfucking', 'fistfuckings', 'fistfucks', 
    'fisting', 'fisty', 'flange', 'flogthelog', 'floozy', 'foad', 'fondle', 'foobar', 
    'fook', 'fooker', 'foot job', 'footjob', 'foreskin', 'freex', 'frigg', 'frigga', 
    'fubar', 'fuck', 'fuck-ass', 'fuck-bitch', 'fuck-tard', 'fucka', 'fuckass', 'fucked', 
    'fucker', 'fuckers', 'fuckface', 'fuckhead', 'fuckheads', 'fuckhole', 'fuckin', 
    'fucking', 'fuckings', 'fuckingshitmotherfucker', 'fuckme', 'fuckmeat', 'fucknugget', 
    'fucknut', 'fuckoff', 'fuckpuppet', 'fucks', 'fucktard', 'fucktoy', 'fucktrophy', 
    'fuckup', 'fuckwad', 'fuckwhit', 'fuckwit', 'fuckyomama', 'fudgepacker', 'fuk', 
    'fuker', 'fukker', 'fukkin', 'fukking', 'fuks', 'fukwhit', 'fukwit', 'futanari', 
    'futanary', 'fux', 'fux0r', 'fvck', 'fxck', 'g-spot', 'gae', 'gai', 'gang bang', 
    'gang-bang', 'gangbang', 'gangbanged', 'gangbangs', 'ganja', 'gassyass', 'gay', 
    'gaylord', 'gays', 'gaysex', 'gey', 'gfy', 'ghay', 'ghey', 'gigolo', 'glans', 
    'goatse', 'god', 'god-dam', 'god-damned', 'godamn', 'godamnit', 'goddam', 'goddammit', 
    'goddamn', 'goddamned', 'gokkun', 'golden shower', 'goldenshower', 'gonad', 'gonads', 
    'gook', 'gooks', 'gringo', 'gspot', 'gtfo', 'guido', 'h0m0', 'h0mo', 'hamflap', 
    'hand job', 'handjob', 'hardcoresex', 'hardon', 'he11', 'hebe', 'heeb', 'hell', 
    'hemp', 'hentai', 'heroin', 'herp', 'herpes', 'herpy', 'heshe', 'hitler', 'hiv', 
    'hoar', 'hoare', 'hobag', 'hoer', 'hom0', 'homey', 'homo', 'homoerotic', 'homoey', 
    'honky', 'hooch', 'hookah', 'hooker', 'hoor', 'hootch', 'hooter', 'hooters', 
    'hore', 'horniest', 'horny', 'hotsex', 'howtokill', 'howtomurdep', 'hump', 'humped', 
    'humping', 'hussy', 'hymen', 'inbred', 'incest', 'injun', 'j3rk0ff', 'jack off', 
    'jack-off', 'jackass', 'jackhole', 'jackoff', 'jap', 'japs', 'jerk', 'jerk off', 
    'jerk-off', 'jerk0ff', 'jerked', 'jerkoff', 'jism', 'jiz', 'jizm', 'jizz', 'jizzed', 
    'junkie', 'junky', 'kawk', 'kike', 'kikes', 'kill', 'kinbaku', 'kinky', 'kinkyJesus', 
      'knob', 'kock', 'kondom', 'konk', 'kunt', 'kuntlicker', 'kuntlicking', 'kuntz', 
    'kyke', 'l3tters', 'l33t', 'l3tters', 'leather', 'lesbian', 'lezzie', 'lube', 
    'masturbate', 'masturbation', 'mofo', 'mofos', 'mutha', 'muthafucka', 'muthafuckas', 
    'muthafuckin', 'muthafucking', 'n1gga', 'n1gger', 'nazi', 'nigga', 'nigger', 
    'niggers', 'nutsack', 'orally', 'p0rn', 'p0rn0', 'p3nis', 'p4k', 'paki', 'pano', 
    'panties', 'pecker', 'peeing', 'piss', 'pissed', 'pissin', 'pissin', 'pissing', 
    'playboy', 'poof', 'poon', 'poop', 'porn', 'porn0', 'pornography', 'pr0n', 
    'puss', 'pussy', 'qu33r', 'queer', 'rape', 'rectum', 'retard', 'rimjob', 'roastbeef', 
    'rubandtug', 's3x', 's3x0', 's3xual', 's3xuality', 's3xy', 'sadist', 'shemale', 
    'sh1t', 'sh1tcock', 'sh1tdick', 'sh1tface', 'sh1tfaced', 'sh1thead', 'sh1thole', 
    'sh1ts', 'sh1tster', 'sh1tty', 'sh1z', 'shlong', 'shyte', 'sickfuck', 'slut', 
    'sluts', 'smut', 'snatch', 'sodomy', 'spacelube', 'spunk', 'stfu', 'strapon', 
    'suck', 'sucker', 'sucking', 't1tt1e5', 't1tties', 'teabagger', 'testicle', 
    'threesome', 'titt', 'tits', 'titwank', 'tosser', 'tranny', 'trannies', 'turd', 
    'twat', 'vag', 'vagina', 'wank', 'wanker', 'wetback', 'wh0r3', 'whore', 'w0r3', 
    'wtf', 'yank', 'yiffy', 'z0mg', 'zoophile', 'zoophilia'
]

# Initialize Blob Service Client
connection_string = 'DefaultEndpointsProtocol=https;AccountName=stagebucket;AccountKey=your_account_key;EndpointSuffix=core.windows.net'
container_name = 'linkcxo'
blob_name = 'user_inputs.csv'

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

def append_to_csv(text, label):
    # Create a DataFrame with the new data
    df = pd.DataFrame({'text': [text], 'label': [label]})

    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, header=False)

    # Upload the CSV to Azure Blob Storage
    blob_client = container_client.get_blob_client(blob_name)
    
    try:
        # Download the existing CSV
        existing_blob = blob_client.download_blob()
        existing_data = existing_blob.readall().decode('utf-8')
    except:
        # If blob doesn't exist, create it with new data
        existing_data = ''
    
    # Append new data
    updated_csv = existing_data + csv_buffer.getvalue()

    # Upload the updated CSV
    blob_client.upload_blob(updated_csv, overwrite=True)

# Function to detect profanity using regex and return the profane words found
def detect_profanity_with_regex(text, profane_words=PROFANE_WORDS):
    pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, profane_words)) + r')\b', re.IGNORECASE)
    profane_words_found = pattern.findall(text)
    
    if profane_words_found:
        return True, profane_words_found  # Return True and the list of profane words found
    return False, []

# Load the Hugging Face model for classification
classifier = pipeline('text-classification', model="parsawar/profanity_model_3.1")

# Streamlit app layout
st.title("Profanity Detection App")

# Input text
user_input = st.text_area("Enter text to check for profanity")

# Preprocess and detect profanity
if st.button("Check for Profanity"):
    if user_input:
        # Clean and preprocess the input text
        cleaned_text = preprocess_text(user_input.strip())

        # Detect profanity using regex
        profane_detected, profane_words_found = detect_profanity_with_regex(cleaned_text)

        if profane_detected:
            profane_words_string = ', '.join(profane_words_found)
            response = {
                'label': "1",  # Profanity detected
                'profane_words': profane_words_string
            }
            st.write("Profanity detected:", profane_words_string)
        else:
            # If no profane words are found, classify the text using Hugging Face model
            result = classifier(cleaned_text)
            classified_label = result[0]['label']
            hf_label_score = result[0]['score']
            response = {
                'label': classified_label,
                'score': hf_label_score,
                'profane_words': None
            }
            st.write(f"Classification result: {classified_label} (Score: {hf_label_score})")

        # Append the result to the CSV
        append_to_csv(user_input, response['label'])
    else:
        st.error("Please enter some text.")

