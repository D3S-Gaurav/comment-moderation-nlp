import pandas as pd

from features import clean_text, engineer_features_linear, expand_contractions


def make_df(comments):
    n = len(comments)
    return pd.DataFrame({
        'comment': comments,
        'race': [None] * n, 'religion': [None] * n, 'gender': [None] * n,
        'disability': [False] * n,
        'if_1': [0] * n, 'if_2': [0] * n,
        'upvote': [0] * n, 'downvote': [0] * n,
    })


def test_contractions_keep_case():
    assert expand_contractions("I'm SO angry, you're WRONG") == "i am SO angry, you are WRONG"


def test_contractions_whole_words_only():
    # "i'd" must not be rewritten inside "hi'd"
    assert expand_contractions("hi'd") == "hi'd"
    assert expand_contractions("she's here") == "she is here"


def test_contractions_non_string():
    assert expand_contractions(None) == ""


def test_caps_ratio_survives_preprocessing():
    text = expand_contractions("I'M GOING TO FIND YOU")
    df = engineer_features_linear(make_df([text, "all lower case"]))
    assert df['caps_ratio'][0] > 0.5
    assert df['caps_ratio'][1] == 0


def test_lexicon_ignores_substrings():
    df = engineer_features_linear(make_df([
        "great skill, the game has begun, read the parable, a small flaw",
    ]))
    assert df['threat_count'][0] == 0
    assert df['identity_count'][0] == 0
    assert df['normal_count'][0] == 0


def test_lexicon_counts_words_and_plurals():
    df = engineer_features_linear(make_df(["I will KILL you, I have guns. Idiots."]))
    assert df['threat_count'][0] == 2
    assert df['toxic_count'][0] == 1
    assert df['has_will_kill'][0] == 1


def test_clean_text():
    assert clean_text("Visit http://x.com NOW!!  #wow") == "visit now!! wow"
