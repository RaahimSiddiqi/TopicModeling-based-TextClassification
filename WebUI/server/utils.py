import re
import spacy
nlp = spacy.load("en_core_web_sm")

contractions = {
    r"ain't": "am not",
    r"aren't": "are not",
    r"can't": "cannot",
    r"could've": "could have",
    r"couldn't": "could not",
    r"didn't": "did not",
    r"doesn't": "does not",
    r"don't": "do not",
    r"hadn't": "had not",
    r"hasn't": "has not",
    r"haven't": "have not",
    r"he'd": "he would",
    r"he'll": "he will",
    r"he's": "he is",
    r"how'd": "how did",
    r"how'll": "how will",
    r"how's": "how is",
    r"i'd": "I would",
    r"i'll": "I will",
    r"i'm": "I am",
    r"i've": "I have",
    r"isn't": "is not",
    r"it'd": "it would",
    r"it'll": "it will",
    r"it's": "it is",
    r"let's": "let us",
    r"mightn't": "might not",
    r"mustn't": "must not",
    r"shan't": "shall not",
    r"she'd": "she would",
    r"she'll": "she will",
    r"she's": "she is",
    r"should've": "should have",
    r"shouldn't": "should not",
    r"that's": "that is",
    r"that'd": "that would",
    r"there's": "there is",
    r"they'd": "they would",
    r"they'll": "they will",
    r"they're": "they are",
    r"they've": "they have",
    r"wasn't": "was not",
    r"we'd": "we would",
    r"we'll": "we will",
    r"we're": "we are",
    r"we've": "we have",
    r"weren't": "were not",
    r"what'll": "what will",
    r"what're": "what are",
    r"what's": "what is",
    r"what've": "what have",
    r"where's": "where is",
    r"who'd": "who would",
    r"who'll": "who will",
    r"who're": "who are",
    r"who's": "who is",
    r"who've": "who have",
    r"won't": "will not",
    r"would've": "would have",
    r"wouldn't": "would not",
    r"you'd": "you would",
    r"you'll": "you will",
    r"you're": "you are",
    r"you've": "you have"
}

quran_spellings = [
    r'qur-an/qur-an-guidan',
    r'qurahn',
    r'qurin',
    r'quraan',
    r'qur’ān',
    r'qurnayn',
    r'qur’an/sunn',
    r'qurân',
    r'qur’aan',
    r'qurʾan',
    r'qurâ€™an',
    r'qurän',
    r'qur;an',
    r'qurʻán',
    r'quruan',
    r'qurʼān',
    r'qur-an',
    r'qurr-on',
    r'qur?an',
    r'quraaniyyoon',
    r'qurran',
    r'qur1an',
    r'quraun',
    r'quraniyyun',
    r'qurýan',
    r'quraaan',
    r'quraaniyoon',
    r'qurʼan',
    r'qur‘ān',
    r'qur`ân',
    r'qur`an',
    r'qurann',
    r'qur`ān',
    r'qurān',
    r'qur’aniyyoon',
    r'qur\x01an',
    r'qur%27an',
    r'quràan',
    r'qura’n',
    r'qur´an',
    r'quranen',
    r'qur‘an',
    r'qur’an',
    r'qurani/quran',
    r'quranan',
    r'quraniyoon',
    r'quran(an',
    r'quran/kuran',
    r'qurʾān',
    r'qur"an',
    r'quraydhan',
    r'quran-an',
    r'quran_an',
    r'qur\\`an'
]


def clean(doc):
    # Make all letters lowercase
    doc=doc.lower()

    # Remove URLs and emails
    doc = re.sub(r'\S*://\S*', ' ' , doc)
    doc = re.sub(r'www\.\S*', ' ' , doc)
    doc = re.sub(r'\S*\.com\S*', ' ' , doc)
    doc = re.sub(r'\S*\.gov\S*', ' ' , doc)
    doc = re.sub(r'\S*\.org\S*', ' ' , doc)
    doc = re.sub(r'\S*\.net\S*', ' ' , doc)
    doc = re.sub(r'\S*\.co\.\S*', ' ' , doc)
    doc = re.sub(r'\S*\.lib\.\S*', ' ' , doc)
    doc = re.sub(r'\S*@\S*', ' ' , doc)

    # Remove the s from possessive nouns
    doc = re.sub(r'\S+\'s', '', doc)

    # expand contractions and also normalise the spelling of quran
    # also adds a space before and after the replaced word
    pattern = r'(' + '|'.join(re.escape(contraction) for contraction in contractions.keys()) + r')'
    doc = re.sub(pattern, lambda x: ' ' + contractions[x.group()] + ' ', doc)

    #Might be useful to normalise the arabic words with apostrophes, it MUST be ran after fixing contractions
    doc = re.sub(r'\'', '', doc)

    # replace every punctuation mark, special character, whitespace character and digit with a space
    doc = re.sub(r'[^a-zA-Z]', ' ', doc)

    doc = nlp(doc)
    # remove stopwords and lemmatize the tokens
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop]
    # join the filtered tokens back into a document/string
    filtered_document = ' '.join(filtered_tokens)
    # # remove single/double lettered words
    filtered_document = re.sub(r'\b\w{1,2}\b', '', filtered_document)
    # remove any excess whitespace
    filtered_document = re.sub(r'\s+', ' ', filtered_document).strip()
    return filtered_document