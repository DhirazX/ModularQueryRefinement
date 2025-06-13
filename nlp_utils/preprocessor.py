# import nltk
import re

def preprocess(text: str):
    # Converting text to lower case
    text = text.lower()

    # Removing Punctuations
    text = re.sub(r'[^\w\s]', '', text)

    #Removing numbers
    text = re.sub(r'\d+', '', text)

    #Removing Whitespace
    text = text.strip()

    return text