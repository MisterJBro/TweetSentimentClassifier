import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from deep_translator import GoogleTranslator
import contractions
from cleantext import clean
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
import random, os
import numpy as np
import string
from tqdm import tqdm
import demoji
from spellchecker import SpellChecker
from textblob import Word
from functools import reduce

# Remove URLs
url_regex = re.compile(r'https?://\S+', re.IGNORECASE)
def remove_urls(sent):
    return url_regex.sub(' ',sent)

# Remove user mentions
user_regex = re.compile(r'(?<![\w.-])@[A-Za-z][\w-]+', re.IGNORECASE)
def remove_users(sent):
    return user_regex.sub(' ', sent)

# Remove ampersand html tag &amp;
ampersand_regex = re.compile(r'&amp;?', re.IGNORECASE)
def remove_ampersand(sent):
    return ampersand_regex.sub(' and ', sent)

# To lower case
def lower_case(sent):
    return sent.lower()

# Remove all whitespace characters (spaces, tabs, newlines etc.) and replace through one simple whitespace
def remove_newlines_tabs(sent):
    return ' '.join(sent.split()).strip()

# Translate the given post into the target language
def translate(sent, target='en'):
    if sent == '':
        return ''
    return GoogleTranslator(source='auto', target=target).translate(sent)

# Translate the sentence into intermediate language and then back to english, if start language is english.
# If not, translate only into english
def backtranslation(sent, start='en', intermediate='de'):
    if start == 'en':
        return translate(translate(sent, target=intermediate))
    return translate(sent)

# Uses data augmentation techniques to augment the data. Returns an array of augmented string + the input string in english as first string
# Backtranslation for every element in langs, if input not in english, specify start argument
sent_swap_aug = nas.RandomSentAug(aug_p=0.5)
insert_aug = naw.ContextualWordEmbsAug(model_path='roberta-base', action='insert', aug_min=1, aug_max=10, aug_p=0.15)
synonym_aug = naw.SynonymAug(aug_min=1, aug_max=15, aug_p=0.25, stopwords_regex=r'(muslim.*|islam.*|uyghur.*|jannah|covid.*|radical.*|extremist.*|)')
#e_aug = naw.WordEmbsAug(model_type='glove', action='substitute', aug_min=1, aug_max=10, aug_p=0.3 , stopwords_regex=r'(muslim.*|islam.*|uyghur.*|jannah|covid.*|radical.*|extremist.*|)')
pipeline_aug = naf.Sequential([
    synonym_aug, insert_aug
])

def data_augmentation(sent, start='en', disable_all=False, langs=['de', 'ru', 'ja', 'fr'], num_nlp_aug=3, sent_swaps=True):
    if start != 'en':
        sents = [translate(sent)]
    else:
        sents = [sent]

    # Using disable all to use data_augmentation like backtranslation
    if disable_all:
        return sents

    # Backtranslation
    for l in langs:
        sents.append(backtranslation(sent, intermediate=l))

    # Augmentation pipeline with NLPAug
    if num_nlp_aug > 1:
        sents.extend(pipeline_aug.augment(sent, n=num_nlp_aug))
    elif num_nlp_aug == 1:
        sents.append(pipeline_aug.augment(sent, n=num_nlp_aug))

    # Random sentence swap augmentation, if there is atleast two sentences
    r_sent = sent_swap_aug.augment(sent, n=1)
    if r_sent != sent and sent_swaps:
        sents.append(r_sent)

    return list(set(sents))

# Convert hashtags that are which consist of several words, e.g. #WeStandTogether -> #we stand together
# Idea for code from https://stackoverflow.com/questions/68448243/efficient-way-to-split-multi-word-hashtag-in-python
hashtag_regex = re.compile(r'# ?\S*')
def convert_hashtags(sent):
    return hashtag_regex.sub(
        lambda m: ' '.join(re.findall('[A-Z][^A-Z]+|[a-z][^A-Z]+', m.group())),
        sent,
    )

# Convert emojis to their description text
def convert_emojis(sent):
    return demoji.replace_with_desc(sent, sep=' ')

# Remove all numbers
number_regex = re.compile(r'[0-9]+', re.IGNORECASE)
def remove_numbers(sent):
    return number_regex.sub(' ', sent)

# Spell check one specific word
spell = SpellChecker()
def spellcheck_word(word, certainty=0.9, discarding=True, discard_str=''):
    if list(spell.unknown([word])):
        correction, cer = Word(word).spellcheck()[0]
        if cer >= certainty:
            return correction
        else:
            if discarding:
                return discard_str
            else:
                return word
    return word

# Spell check every word and fix it if the model is certain enough or discard it, if it is not fixable
spell.word_frequency.load_words(['islamization', 'islamisation', 'islamophobia', 'islamophobe', 'islamophobic', 'jannah', 'covid', 'covid19', 'corona', 'bbc',
                                 'closethecamps', 'uyghur', 'uyghurs', 'wuhan', 'flattenthecurve', 'islamism', 'btw', 'wtf', 'ramadan', 'concentrationcamps',
                                 'antifa', 'mashallah', 'makkah', 'blacklivesmatters', 'hatespeech', 'closecamps', 'islamics', 'omicron', 'allahuakbar'])
def spellcheck(sent, certainty=0.9, discarding=True, discard_str=' '):
    words = [spellcheck_word(w, certainty=certainty, discarding=discarding, discard_str=discard_str) for w in nltk.word_tokenize(sent)]
    return ' '.join(words)

# Fix unicode error and convert to ascii
def convert_to_ascii(sent):
    return clean(sent, lower=False)

# Fix different apostrophes e.g. ’ into '
apostrophe_regex = re.compile(r'`|’', re.IGNORECASE)
def fix_apostrophes(sent):
    return apostrophe_regex.sub('\'', sent)

# Expand contractions e.g. I've -> I have
def expand_contractions(sent):
    return contractions.fix(sent, slang=True)

# Remove all punctuation
punct_all_regex = re.compile('[%s]' % re.escape(string.punctuation))
def remove_all_punctuation(sent):
    sent = punct_all_regex.sub(' ', sent)
    return sent

# Lemmatization of words
#white = pd.read_csv('../datasets/whitelist/whitelist.csv', names=['content'])
#white = set(white['content'].tolist())
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
lemmatizer = WordNetLemmatizer()
def lemmatize(sent):
    lem = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sent)]
    #lem = [l for l in lem if l in white]
    return ' '.join(lem)

# Remove stop word, however, some stopwords are important keep them or e.g. "muslim against racism" -> "muslim racism", changes the meaning
# So some stopwords should still be included, especially solidarity is expressed with strong word like: 'we' 'us', and anti solidarity with: 'they' etc.
stopwords = stopwords.words('english').copy()
stopwords.remove('against')
stopwords.remove('we')
stopwords.remove('in')
stopwords.remove('our')
stopwords.remove('all')
stopwords.remove('no')
stopwords.remove('not')
stopwords.remove('they')
stopwords.remove('their')
stopwords.remove('them')
stopwords.remove('you')
stopwords.remove('if')
stopwords.remove('her')
stopwords.remove('him')
stopwords.remove('what')
stopwords.remove('than')
stopwords.remove('off')
stopwords.remove('have')

stop_word_regex = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
def remove_stop_words(sent):
    return stop_word_regex.sub('', sent)

# The complete preprocessing pipeline for one sentence
step_one = [
    remove_urls, remove_users, remove_ampersand, convert_hashtags, lower_case, remove_newlines_tabs
]
step_two = [
    convert_emojis, convert_to_ascii, fix_apostrophes, expand_contractions,
    remove_all_punctuation, # spellcheck,
    lemmatize, remove_stop_words,
    remove_numbers, lower_case, remove_newlines_tabs
]

# Preprocess one sentence
def preprocess(sent, lang='en', data_aug=True):
    sent = reduce(lambda res, f: f(res), step_one, sent)
    new_sents = data_augmentation(sent, start=lang, disable_all=not data_aug, langs=['ru', 'ja', 'fr'], num_nlp_aug=1, sent_swaps=True)
    new_sents = [reduce(lambda res, f: f(res), step_two, s) for s in new_sents]
    return new_sents

# Preprocess a whole dataframe, atleast a lang and content column is needed, labels are optional
def preprocess_df(df, data_aug=True):
    new_content = []
    if 'label' in df.columns:
        new_labels = []
        for label, lang, content in tqdm(zip(df['label'].tolist(), df['lang'].tolist(), df['content'].tolist()), total=len(df)):
            new_sents = preprocess(content, lang=lang, data_aug=data_aug)
            for sent in new_sents:
                new_content.append(sent)
                new_labels.append(label)
        df = pd.DataFrame(list(zip(new_content, new_labels)), columns=['content', 'label'])
        # Shuffle in the end
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        for lang, content in tqdm(zip(df['lang'].tolist(), df['content'].tolist()), total=len(df)):
            new_sents = preprocess(content, lang=lang, data_aug=data_aug)
            for sent in new_sents:
                new_content.append(sent)
        df = pd.DataFrame(list(zip(new_content)), columns=['content'])

    return df

if __name__ == "__main__":
    # Set seed
    seed = 2022
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # Importing data
    train_df = pd.read_csv('../datasets/competition/train.csv')
    train = train_df[:400]
    dev = train_df[400:]

    # Preprocess list, use the following options: train, dev
    preprocess_list = ['train', 'dev']
    for name in preprocess_list:
        df = eval(name)

        # Starting preprocessing
        print(f'Starting preprocessing of {name}!')
        df = preprocess_df(df, data_aug=name=='train')

        # Save csv
        df.to_csv(f'../datasets/preprocessed/{name}.csv', sep=',', index=False)
        print('Finished preprocessing!')
