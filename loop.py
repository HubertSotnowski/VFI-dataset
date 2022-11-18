import glob

number = 0
import os
import time
from distutils.dir_util import copy_tree
import shutil

for name in glob.glob("./data/*", recursive=True):
    print(name)
    time.sleep(1)
    os.system(f"sh ./_scripts/rrld_pipeline.sh {name} ./{number}")
    copy_tree(f"./{number}/images/", f"dataset/{name}")
    time.sleep(1)
    shutil.rmtree(f"./{number}")
    number += 1
