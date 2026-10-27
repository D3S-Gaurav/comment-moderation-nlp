import re

import numpy as np
import pandas as pd

TEXT_COL = 'comment'

CONTRACTIONS = {
    "i'll": "i will", "you'll": "you will", "he'll": "he will",
    "she'll": "she will", "we'll": "we will", "they'll": "they will",
    "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "won't": "will not",
    "can't": "cannot", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "haven't": "have not",
    "hasn't": "has not", "hadn't": "had not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "gimme": "give me", "lemme": "let me",
}

THREAT_WORDS = [
    'kill', 'shoot', 'execute', 'cut your',
    'exterminate', 'nuke', 'torture', 'gun', 'murder', 'bomb'
]
IDENTITY_WORDS = [
    'muslim', 'jewish', 'black', 'white', 'gay', 'lesbian',
    'transgender', 'hindu', 'christian', 'negro', 'islam',
    'queer', 'mexican', 'arab', 'immigrant'
]
TOXIC_WORDS = [
    'idiot', 'stupid', 'moron', 'loser', 'troll', 'lunatic',
    'pathetic', 'ignorant', 'fool', 'liar', 'hypocrite',
    'coward', 'disgusting', 'trash', 'garbage'
]
NORMAL_WORDS = [
    'however', 'therefore', 'according', 'evidence', 'research',
    'government', 'policy', 'law', 'article', 'report'
]

# Whole-word patterns. Plain substring checks counted 'kill' in "skill",
# 'gun' in "begun", 'arab' in "parable", 'law' in "flaw" and so on.
# A trailing s? keeps simple plurals ("guns", "muslims") matching.
CONTRACTION_RE = re.compile(
    r"\b(" + '|'.join(re.escape(k) for k in CONTRACTIONS) + r")\b", re.IGNORECASE)


def word_patterns(words):
    return [re.compile(r'\b' + re.escape(w) + r's?\b') for w in words]


THREAT_RE = word_patterns(THREAT_WORDS)
IDENTITY_RE = word_patterns(IDENTITY_WORDS)
TOXIC_RE = word_patterns(TOXIC_WORDS)
NORMAL_RE = word_patterns(NORMAL_WORDS)


def expand_contractions(text):
    # Keeps the original casing so caps_ratio still has something to measure.
    # The old version lower-cased here, which made caps_ratio 0 for every row.
    if not isinstance(text, str):
        return ""
    return CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)


def count_words(patterns, text):
    # Number of distinct lexicon words present, same meaning as the old sum(w in x ...)
    text = text.lower()
    return sum(1 for p in patterns if p.search(text))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z0-9\s!?.\'\-\/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def engineer_features_linear(df):
    df = df.copy()

    df[TEXT_COL] = df[TEXT_COL].fillna('')

    for col in ['race', 'religion', 'gender']:
        df[col] = df[col].fillna('Unknown')

    df['disability'] = df['disability'].fillna(0).astype(int)

    text = df[TEXT_COL]
    df['word_count']     = text.apply(lambda x: len(x.split()))
    df['char_count']     = text.apply(len)
    df['caps_ratio']     = text.apply(
                               lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df['threat_count']   = text.apply(lambda x: count_words(THREAT_RE, x))
    df['threat_word_ratio'] = df['threat_count'] / (df['word_count'] + 1)
    df['identity_count'] = text.apply(lambda x: count_words(IDENTITY_RE, x))
    df['toxic_count']    = text.apply(lambda x: count_words(TOXIC_RE, x))
    df['normal_count']   = text.apply(lambda x: count_words(NORMAL_RE, x))
    df['threat_x_identity'] = df['threat_count'] * df['identity_count']
    df['has_threat']     = (df['threat_count'] > 0).astype(int)
    df['has_identity']   = (df['identity_count'] > 0).astype(int)
    df['has_toxicity']   = (df['toxic_count'] > 0).astype(int)
    df['seems_normal']   = (df['normal_count'] > 0).astype(int)

    # case=False because the text is no longer lower-cased before this point
    df['has_will_kill']  = text.str.contains(
        r'will kill|will shoot|will murder|will hurt', case=False, regex=True).astype(int)
    df['has_going_to']   = text.str.contains(
        r'going to kill|going to shoot|going to hurt', case=False, regex=True).astype(int)
    df['has_should_die'] = text.str.contains(
        r'should die|deserve to die|hope.*die|want.*dead', case=False, regex=True).astype(int)
    df['has_all_x_are']  = text.str.contains(
        r'all \w+ are|these \w+ are|\w+ people are', case=False, regex=True).astype(int)
    df['has_go_back']    = text.str.contains(
        r'go back|leave.*country|not welcome|get out', case=False, regex=True).astype(int)

    df['if1_is_0']  = (df['if_1'] == 0).astype(int)
    df['if1_is_4']  = (df['if_1'] == 4).astype(int)
    df['if2_is_4']  = (df['if_2'] == 4).astype(int)
    df['if1_gt_10'] = (df['if_1'] > 10).astype(int)
    df['if2_is_10'] = (df['if_2'] == 10).astype(int)
    df['if2_gt_50'] = (df['if_2'] > 50).astype(int)

    df['net_score']         = df['upvote'] - df['downvote']
    df['controversy_ratio'] = df['upvote'] / (df['downvote'] + 1)
    df['total_votes']       = df['upvote'] + df['downvote']
    df['is_controversial']  = (df['controversy_ratio'] > 2).astype(int)

    df['if_1_log'] = np.log1p(df['if_1'])
    df['if_2_log'] = np.log1p(df['if_2'])

    df = pd.get_dummies(df, columns=['race', 'religion', 'gender'], dtype=int)

    return df
