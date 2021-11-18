import numpy as np
import re
import os
from pathlib import Path
from tqdm.auto import tqdm


def data_chooser(path, destine, threshold=35):
    """
    Clean data with 2 steps: 
    - The first paragraph containing the metadata is deleted. 
    - Those with line numbers less than (and NOT equal to) threshold 
        are removed from the dataset. 
    
    Drawbacks:
    - Some metadata are two paragraphs, the second paragraph aren't 
        removed. 
    - Only UTF-8 encoding are read. Others aren't read and discarded. 
        To use them, requires manual function. They're added to errors.
    - There're three files with extra "Lines: " (there are two matches)
        and only the first one is taken, and the second one is 
        treated as part of the paragraph. 
    - If the first line is not a number, it will not be processed and
        added to errors. 

    :var path: (pathlib.Path) The base path where the data located. 
    :var destine: (pathlib.Path) The base path of the destination
    :var threshold: (int) The number of lines to remove the data, 
        not inclusive.
    """
    dest = destine/path.name
    files = os.listdir(path)
    errors = []

    if not os.path.exists(dest): os.mkdir(dest)

    for file in tqdm(files): 
        lines = (path/file).open()
        f = None
        openit = False

        # Try if we could open the file. 
        try: 
            i, line_num = 0, 0
            for i, l in enumerate(lines): 
                if re.match("Lines:", l) and openit is False: 
                    try: line_num = int(re.findall(r'\d+', l)[-1])
                    except IndexError: pass
                    if line_num < threshold: break
                if re.match("\n", l) and openit is False:
                    openit = True
                    if f: f.close()
                    f = (dest/f"{file}.txt").open("w")
                    continue

                if openit is True: f.write(l)

            try: f.close()
            except Exception: pass

        except UnicodeDecodeError as e: 
            errors.append(str(path/file) + f"\t{i}\t{line_num}" + "\n")
        

    with open(destine/"errors.txt", "a") as f: f.writelines(errors)



def split_data(data, idx=None):
    """Split data for easier visualization. NOT FOR TRAINING."""
    if idx is not None: x = data[idx].split("\n")
    else: x = data.split("\n")
    x = list(filter(lambda a: a != '', x))
    return x


def threshold_subset(all_xs, threshold=10):
    """
    Will return a numpy array containing all the data that is to be kept.
    Threshold means the number of lines (after separated using `split_data`)
    for the data to be retained, inclusive. By default, threshold=10 means
    if data is less than 10 lines it will be discarded. 
    """
    to_keep = []
    for k, x in enumerate(all_xs):
        sentence_len = len(split_data(x))
        if sentence_len >= threshold: to_keep.append(k)

    return np.array(to_keep)


def clean_data(text, include_newline=True):
    """Cleaning data based on certain set of characters.
    If include_newline=True, newline will be replaced by single space."""
    text = re.sub("[:<>\t*-]", "", text)
    if include_newline: text = re.sub("\n", " ", text)
    text = text.split("\'")  # .replace("\\", "") doesn't work. 
    text = "'".join(text)
    return text


def stoi(vocab, word):
    """
    If single word, we will convert it to list. 
    If not single word str, it will make it a list. 
    Then we want the type to be correct, else we raise error. 
    """
    from fastcore.foundation import L

    # Convert array to dict for vocab, if it is
    if type(vocab) in [np.ndarray, list, L]:
        vocab = {key: index for index, key in enumerate(vocab)}

    # type checks and do stoi. 
    if isinstance(word, str): word = word.split(" ")  # convert to list
    if type(word) in [list, tuple, np.ndarray, L]:
        m = np.zeros((len(word), )).astype(np.uint8)
        for k in range(len(word)):
            try: m[k] = vocab[word[k]]
            except KeyError: m[k] = 0  # xxunk assigned, word not in dict. 
        return m
    else: raise NotImplementedError(f"{type(word)} not implemented")