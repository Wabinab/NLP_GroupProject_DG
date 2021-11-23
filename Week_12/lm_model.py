from pathlib import Path
import gc
from fastai.text.all import *
from tqdm.notebook import tqdm
import sys
import re
import numpy as np
from nlputils import *
from sklearn.datasets import fetch_20newsgroups
from tqdm.auto import tqdm
from collections import Counter
import string
import pandas as pd


os.makedirs("models/outputs", exist_ok=True)
os.makedirs('./outputs', exist_ok=True)
all_xs, all_y = fetch_20newsgroups(subset="all", remove=('headers', 'footers', 'quotes'),
                    shuffle=True, return_X_y=True)

k = 7

choice = np.load(f"choice_{k}.npy")
g = threshold_subset(all_xs, k)
all_xs = np.array(all_xs)[g]
all_y = np.array(all_y)[g]
print("Finish thresholding. ")

g = choice < 0.1
all_xs = all_xs[g]
all_y = all_y[g]
print("Finish removing xxunk above threshold.")

# doesn't change all_y
all_xs = np.array([clean_data(x, False) for x in tqdm(all_xs)])
print("Finish cleaning data.")

try: counter = load_pickle(f"counter_{k}.pkl")
except Exception: 
    counter = Counter()
    for data in tqdm(all_xs): counter += Counter(data.split())
    save_pickle(f"counter_{k}.pkl", counter)


def strip_punc(s): 
    return s.translate(str.maketrans('', '', string.punctuation))


# lower all cases
our_vocab = {v.lower() for v in counter}
del counter

# remove all punctuations
our_vocab = {strip_punc(v) for v in our_vocab}


def brit_to_amer(ending, brit_rule, americ_rule):
    r = re.compile(ending)
    compare_list = list(filter(r.match, our_vocab))
    ame_list = [re.sub(brit_rule, americ_rule, s) for s in compare_list]

    non_empty = {}
    for b, a in tqdm(zip(compare_list, ame_list)):
        m = re.compile(rf'{a}')
        z = list(filter(m.match, our_vocab))
        if a in z: non_empty[b] = a

    return non_empty


british_eng = {
    "rumour": "rumor",
    "vapour": "vapor",
    "arbour": "arbor",
    "colour": "color",
    "behaviour": "behavior",
    "saviour": "savior",
    "favour": "favor",
    "armour": "armor",
    "honour": "honor",
    "inferiour": "inferior",
    "labour": "labor",
    "humour": "humor",
    "endeavour": "endeavor",
    "harbour": "harbor",
    "fervour": "fervor",
    "parlour": "parlor",
    "neighbour": "neighbor",
    "flavour": "flavor",
    "belabour": "belabor",
    'survivour': 'survivor',  # end of our --> or.
    'aerial': "antenna",
    'anywhere': 'anyplace', 
    # 'autumn': 'fall'   # which fall we're deciding? fall or fall?
    "solicitor": "attorney",
    'biscuit': 'cookie',
    'bonnet': 'hood',
    'janitor': 'aretaker',
    'constable': 'patrolman',
    'dynamo': 'generator',
    # and others it's just too many one decide to leave it here for now. 
}

# our to or is too dirty to be used. 
british_eng.update(brit_to_amer(r'[a-z]+ise$', r'ise', r'ize')) # ise --> ize. 
british_eng.update(brit_to_amer(r'[a-z]+yse$', 'yse', 'yze'))  # use --> yze
british_eng.update(brit_to_amer(r'[a-z]+ae[a-z]$', 'ae', 'e'))  # ae --> e
del british_eng["michael"], british_eng["laer"], british_eng["caen"]
del british_eng["raes"]
# oe --> e checked nothing useful (mostly useful translated become rubbish)
british_eng.update({'defence': 'defense',
 'sence': 'sense',
 'selfdefence': 'selfdefense',
 'nonsence': 'nonsense',
 'pretence': 'pretense',
 'absence': 'absense',
 'essence': 'essense',
 'licence': 'license',
 'offence': 'offense'})  # ence --> ense after deleting rubbish.
british_eng.update({'catalogue': 'catalog'})  # ogue --> og. 

spelling_mistakes = {
    "bahaviour": "behavior",
    "excercise": "exercise",
    "supprise": "surprise",
    "suprise": "surprise",
    "appologise": "apologize",
    "appologize": "apologize",
    'excersise': "exercise",
    'oterwise': "otherwise",
    "frnachise": "franchise",
    "fulfullment": "fulfillment",
    'usuallu': "usually",
    'specfically': "specifically",
    'espically': "especially",
    "talll": "tall",
    'usally': "usually",
    'ususally': "unusually",
    'adventually': 'eventually',
    'oscialltor': 'oscillator',
    'xcellerator': 'accelerator',
    'reccollecting': 'recollecting',
    'osciallator': 'oscillator',
    'unballance': 'unbalance',
    'congroller': 'controller',
    'weeeeelllllll': 'well',
    'killig': 'killing',
    'oscilliscope': "oscilloscope",
    "ussually": "usually",
    'knoew': 'knew',
    "hense": "hence",
    
}

spelling_mistakes.update(brit_to_amer(r'[a-z]+lll[a-z]+$', 'lll', 'll'))

def word_neutralizer(spell, brit_to_amer, word, return_type="str"):
    """
    Neutralizes word(s). 

    :spell: (dict/equivalent) contains spelling mistakes. Mistakes are the 
        keys and corrected words are the values. 
    :brit_to_amer: (dict/equivalent) british to american. British English 
        as keys, American English as values. (Could be reversed if you
        with it to be the other way round).
    :word: (str/list/equivalent) Word(s) that we want to clean. 
    :return_type: (str) The return type. Make sure what you enter are in
        quotations and they're imported/legible type or this whole function
        will fail. 

    :return: cleaned words. 
    """
    # changes spelling mistakes first before brit_to_amer
    for k, v in spell.items(): word = word.replace(k, v)
    for k, v in brit_to_amer.items(): word = word.replace(k, v)
    return word


assert word_neutralizer(spelling_mistakes, british_eng, "hense colour") == "hence color"
assert word_neutralizer(spelling_mistakes, british_eng, "hense color") == "hence color"
assert word_neutralizer(spelling_mistakes, british_eng, "hence colour") == "hence color"
assert word_neutralizer(spelling_mistakes, british_eng, "hence color") == "hence color"

for k in tqdm(range(len(all_xs))):
    all_xs[k] = word_neutralizer(spelling_mistakes, british_eng, all_xs[k])


df = pd.DataFrame([all_xs, all_y]).T
df.columns = ["text_", "category"]


#%% ULMFiT

bs = 96
dls_lm = TextDataLoaders.from_df(df, is_lm=True, bs=bs,
            n_workers=os.cpu_count() * 2)


learn = language_model_learner(dls_lm, AWD_LSTM, pretrained=True, 
            metrics=[accuracy, Perplexity()], wd=0.1, drop_mult=0.3,
            moms=(0.8, 0.7, 0.8)).to_fp16()
suggested_lr_valley = learn.lr_find()
print(suggested_lr_valley)

lr = 5e-3
lr *= bs/48
learn.fine_tune(10, lr*5)


# O
save_path = Path(".")/"outputs"  # this is gitignored as it's v.big. 
try:
    learn.save("./outputs/fine_tuned")
    learn.save_encoder("./outputs/fine_tuned_enc")
except Exception: print("Save loc 1 failed.")

try: 
    learn.save(save_path/"fine_tuned")
    learn.save_encoder(save_path/"fine_tuned_enc")
except Exception: print("Save loc 2 failed")

curr_path = os.getcwd()
try: 
    learn.save(os.path.join(curr_path, "outputs/fine_tuned"))
    learn.save_encoder(os.path.join(curr_path, "outputs/fine_tuned_enc"))
except Exception: pass


#%% 
try: del learn, dls_lm
except Exception: pass
torch.cuda.empty_cache()
gc.collect()

sl = 72  # sequence length. 
bs = 32

dls = DataBlock(
    blocks=(TextBlock.from_df("text_", seq_len=sl, tok=SpacyTokenizer("en")),
            CategoryBlock),
    get_x=ColReader("text"), get_y=ColReader("category"),
    splitter=RandomSplitter(0.1)
).dataloaders(df, bs=bs, num_workers=os.cpu_count()*2, seq_len=sl)

learn_c = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.3,
                metrics=accuracy, moms=(0.8, 0.7, 0.8)).to_fp16()
learn_c.load_encoder(os.path.join(curr_path, "outputs/fine_tuned_enc"))
learn_c.freeze()

suggested_lr_valley = learn_c.lr_find()
print(suggested_lr_valley)

# Gradual unfreezing + gradual descend of lr. 

learn_c.fit_one_cycle(1, 3e-2, moms=(0.8, 0.7, 0.8))
learn_c.save(os.path.join(curr_path, "outputs/first"))

learn_c.freeze_to(-2)
learn_c.fit_one_cycle(1, slice(1e-2/(2.6**4), 1e-2), moms=(0.8, 0.7, 0.8))
learn_c.save(os.path.join(curr_path, "outputs/second"))

learn_c.freeze_to(-3)
learn_c.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3), moms=(0.8, 0.7, 0.8))
learn_c.save(os.path.join(curr_path, "outputs/third"))

learn_c.unfreeze()
learn_c.fit_one_cycle(3, slice(1e-3/(2.6**4), 1e-3), moms=(0.8, 0.7, 0.8))
learn_c.save(os.path.join(curr_path, "outputs/final_model"))