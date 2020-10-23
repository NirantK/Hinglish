import os
import subprocess

os.system("pip3 install fire")
import platform
import sys
from pathlib import Path
from typing import List
from hinglishutils import get_files_from_gdrive
import fire

requirements = Path("requirements.txt")


def run(command):
    subprocess.call(command)


def install(req_file,:
    # if platform.system() == "Darwin":
    #     continue
    # elif platform.system() == "Linux":
    #     continue

    with Path(req_file).open(mode="r") as f:
        packages = f.readlines()
        packages = [p.strip() for p in packages]


def download_files():
    get_files_from_gdrive("https://drive.google.com/file/d/1-Ki6v1a1jF79qx22gM6JlX1NVD4txTdn/view?usp=sharing", 
                      "train_lm.txt")

    get_files_from_gdrive("https://drive.google.com/file/d/1-MRU7w2_la36qopO8Ob4BoCynOAZc0sZ/view?usp=sharing", 
                        "dev_lm.txt")

    get_files_from_gdrive("https://drive.google.com/file/d/1-NqiU-tL5hW59MFtUXh1exivRokZKfs7/view?usp=sharing", 
                        "test_lm.txt")
    get_files_from_gdrive("https://drive.google.com/file/d/1k4N0JlVOP-crIcCtC6ZI5Va8X3s2-r_D/view?usp=sharing", 
                      "test_labels_hinglish.txt")
    get_files_from_gdrive("https://drive.google.com/file/d/1-FykBMdD7erRhr9370thtySNm6QvnQAA/view?usp=sharing", 
                        "train.json")

    get_files_from_gdrive("https://drive.google.com/file/d/1-F6o4lSub2D-_iCoNPvxxnCiPQ82VJjG/view?usp=sharing", 
                        "test.json")

    get_files_from_gdrive("https://drive.google.com/file/d/1-Esp4UtIZwX44eI8qndngweKZ6p9GLKT/view?usp=sharing", 
                        "valid.json")

    get_files_from_gdrive("https://drive.google.com/file/d/17wFvtj9tfp4QI6FrErAyqL9H1s5-lZkR/view?usp=sharing", 
                        "final_test.json")

def install_everything():
    install(requirements)
    download_files()
    os.system("git clone https://github.com/meghanabhange/transformers.git")
    os.system("cd transformers && pip3 install -e .")
    os.system("cd ..")
    os.system(f"echo WANDB_PROJECT=hinglish && wandb login")
    os.system("cp transformers/examples/language-modeling/run_language_modeling.py .")

if __name__ == "__main__":
    fire.Fire(install_everything)