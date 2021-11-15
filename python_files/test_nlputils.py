"""
Test cases for nlputils
"""
import numpy as np
import os
import pytest
import sys
import shutil
from pathlib import Path

# curr_path = Path(os.getcwd())
# sys.path.append(curr_path)
# sys.path.append(curr_path.parent)

from nlputils import *


# Setup 
curr_path = Path(os.getcwd())
path = curr_path/"test/sample_data"
dest = curr_path/"test/dest"


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
    assert g[0][0] == "/home/fastai2/notebooks/DataGlacier/NLP_GroupProject_DG/python_files/test/sample_data/comp.graphics/38291\t0\t0\n"


def test_data_chooser_below_threshold_not_chosen():
    some_setup()
    assert len(os.listdir(dest/"alt.atheism")) == 3
    assert len(os.listdir(dest/"comp.graphics")) == 1


def test_data_chooser_another_threshold_file_as_expected():
    some_setup(threshold=34)
    assert len(os.listdir(dest/"comp.graphics")) == 2


# Teardown (runs for ALL test, each)
def teardown():
    for f in os.listdir(dest): 
        try: shutil.rmtree(dest/f)
        except NotADirectoryError: os.remove(dest/f)