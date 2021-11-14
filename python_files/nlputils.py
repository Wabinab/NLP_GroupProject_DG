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

            for i, l in enumerate(lines): 
                if re.match("Lines:", l) and openit is False: 
                    try: line_num = int(re.findall("\d+", l)[0])
                    except IndexError: errors.append(str(path/file) + "\n")
                    if line_num < 35: break
                if re.match("\n", l) and openit is False:
                    openit = True
                    if f: f.close()
                    f = (dest/f"{file}.txt").open("w")
                    continue

                if openit is True: f.write(l)

            try: f.close()
            except Exception: pass

        except UnicodeDecodeError: 
            errors.append(str(path/file) + "\n")

    with open(destine/"errors.txt", "a") as f: f.writelines(errors)
