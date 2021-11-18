"""
Test cases for nlputils
"""
import numpy as np
import os
import pytest
import sys
import shutil
from pathlib import Path
from fastcore.foundation import L
from sklearn.datasets import fetch_20newsgroups
all_xs, all_y = fetch_20newsgroups(subset="all", remove=('headers', 'footers', 'quotes'),
                    shuffle=True, return_X_y=True)

# curr_path = Path(os.getcwd())
# sys.path.append(curr_path)
# sys.path.append(curr_path.parent)

from nlputils import *


# Setup 
# curr_path = Path(os.getcwd())
curr_path = Path(".")
path = curr_path/"test/sample_data"
dest = curr_path/"test/dest"
if not os.path.exists(dest): os.mkdir(dest)


# If we call setup, it will run for ALL, but I only want it to run for MOST. 
# That's why this is very redundant. 
# It's also not efficient because pytest need to call each time. 
# We can optimize for efficiency by calling "some_teardown" for example
# but we'll see if we want to do that later. 
def some_setup(threshold=35):
    for folder in os.listdir(path): data_chooser(path/folder, dest, threshold)

# # Pytest data chooser part 1
def test_data_chooser_creates_all_folders_dest():
    some_setup()
    sample_folders = os.listdir(path)
    for folder in sample_folders: assert os.path.exists(dest/folder)


def test_data_chooser_have_error_file():
    some_setup()
    assert os.path.exists(dest/"errors.txt")


def test_data_chooser_error_file_have_one_error():
    """The error is done deliberately to have one file. Changes required
    if you change the sample_data, hence not general. """
    some_setup()
    g = []
    with open(dest/"errors.txt", "r") as f: g.append(f.readlines())

    assert len(g) == 1
    assert g[0][0] == str(path/"comp.graphics/38291\t0\t0\n")


def test_data_chooser_below_threshold_not_chosen():
    some_setup()
    assert len(os.listdir(dest/"alt.atheism")) == 3
    assert len(os.listdir(dest/"comp.graphics")) == 1


def test_data_chooser_another_threshold_file_as_expected():
    some_setup(threshold=34)
    assert len(os.listdir(dest/"comp.graphics")) == 2


def test_split_data_without_idx_works():
    data = "This is\nwhat we want\nto see."
    assert split_data(data) == ["This is", "what we want", "to see."]


def test_split_data_with_idx_works():
    data = ["This is\nwhat we want\nto see.", 
            "What do\nyou want\nto see?"]
    assert split_data(data, 1) == ["What do", "you want", "to see?"]


def test_split_data_with_multiple_newline_works():
    data = "This is\n\n\nwhat we want\n\n\nto see."
    assert split_data(data) == ["This is", "what we want", "to see."]


def test_split_data_with_list_raise_error():
    """Particularly, this is really not implemented in the program."""
    data = [["This is\nwhat we want\nto see."], 
            ["What do\nyou want\nto see?"]]
    with pytest.raises(AttributeError): split_data(data, 1)


def test_threshold_subset_works_as_expected_larger_than():
    data = ["This is\nnot what\nyou\'d expect\nif you\ntry this\nagain."]
    assert len(threshold_subset(data)) == 0


def test_threshold_subset_works_as_expected_smaller_than():
    data = ["This is\nnot what\nyou\'d expect\nif you\ntry this\nagain."]
    assert len(threshold_subset(data, 3)) == 1


@pytest.fixture
def three_str(tmpdir): 
    return ["This is\nwhat we want\nto see, at\nleast for now.",   # 4
            "What do\nyou want\nto see?", # 3
            "This is\nnot what\nyou\'d expect\nif you\ntry this\nagain."] # 6


def test_threshold_subset_multiple_works(three_str):
    assert len(threshold_subset(three_str, 3)) == 3
    assert len(threshold_subset(three_str, 4)) == 2


def test_threshold_subset_returns_array_is_index(three_str):
    assert list(threshold_subset(three_str, 4)) == [0, 2]


def test_threshold_subset_return_array_leq_false(three_str):
    """So we want less than or equal rather than greater than or equal. """
    assert list(threshold_subset(three_str, 4, False)) == [1]


def test_clean_data_all_expected_to_go_is_gone_first_line():
    """Ignore the data, it has no meaning and simple gibberish written."""
    data = """Whatever he is: We shall not rule> based on what they said to us <> we shall never know <\twithout proper\tdefinition existed*****************--------------------"""
    assert clean_data(data) == """Whatever he is: We shall not rule based on what they said to us  we shall never know without properdefinition existed"""


@pytest.fixture
def cleandata_str(tmpdir): 
    return "This is\nnot what\nyou\'d expect\nif you\ntry this\nagain."


def test_clean_data_include_newline_false_not_split(cleandata_str):
    assert clean_data(cleandata_str, False) == "This is\nnot what\nyou\'d expect\nif you\ntry this\nagain."


def test_clean_data_include_newline_true_split_accordingly(cleandata_str):
    assert clean_data(cleandata_str, True) == "This is not what you\'d expect if you try this again."


def test_clean_data_replaced_works(cleandata_str):
    """Note in previous test it works when we don't replace \' because Python ignores the backslash.
    But that's not necessary the case when we human reads, and unsure whether it's the case if ML models
    reads, hence we remove it. """
    assert clean_data(cleandata_str, True) == "This is not what you'd expect if you try this again."


vocab = ["I", "you", "be", "want", "fun", "have", "to", "single"]
eg_sentence_str = ["I", "want", "you", "to", "have", "fun"]
result_int = np.array([0, 3, 1, 6, 5, 4])


def test_stoi_list_vocab(): 
    assert (stoi(vocab, eg_sentence_str) == result_int).all()


def test_stoi_ndarray_vocab(): 
    assert (stoi(np.array(vocab), eg_sentence_str) == result_int).all()


def test_stoi_L_vocab(): 
    assert (stoi(L(vocab), eg_sentence_str) == result_int).all()


def test_stoi_dict_vocab():
    vocab1 = {key: index for index, key in enumerate(vocab)}
    assert (stoi(vocab1, eg_sentence_str) == result_int).all()


def test_stoi_str_word_works_single():
    word = "single"
    assert stoi(vocab, word) == np.array([7])


def test_stoi_str_word_works_multiple():
    word = "be single you"
    assert (stoi(vocab, word) == np.array([2, 7, 1])).all()


def test_stoi_list_word_works():
    assert True  # check test_stoi_list_vocab


def test_stoi_tuple_word_works():
    assert (stoi(vocab, tuple(eg_sentence_str)) == result_int).all()


def test_stoi_ndarray_word_works():
    assert (stoi(vocab, np.array(eg_sentence_str)) == result_int).all()


def test_stoi_L_word_works():
    assert (stoi(vocab, L(eg_sentence_str)) == result_int).all()


def test_stoi_dict_not_implemented():
    """
    Even though if the dict is this {0: "this", 1: "is", 2: "true"}
    it might/will work, but we aren't sure whether the Key really 
    represents position, so we don't support it. 
    """
    word = {0: "be", 1: "single", 2: "you"}
    with pytest.raises(NotImplementedError): stoi(vocab, word)


# Teardown (runs for ALL test, each)
def teardown():
    try:
        for f in os.listdir(dest): 
            try: shutil.rmtree(dest/f)
            except NotADirectoryError: os.remove(dest/f)
    except Exception: pass