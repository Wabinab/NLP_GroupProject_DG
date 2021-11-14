import numpy as np
import os
import re
from pathlib import Path 

path = Path("/home/fastai2/Music/20_newsgroups")

# Make directory for our new dataset
dest = path.parent/"20_newsgroups_new"
try: os.mkdir(dest)
except Exception: pass
assert os.path.exists(dest)

from nlputils import *

for folder in os.listdir(path):
    data_chooser(path/folder, dest)