from setuptools import setup
# read the contents of your README file
from pathlib import Path

# read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name = "fastlogreg",
   version = "0.0.1",
     author = "Nechba Mohammed, Mouhajir Mohamed, and Sedjari Yassine",
description = "Efficient logistic regression in Python with GPU support",
long_description =  long_description,
long_description_content_type = "text/markdown",
url = "https://github.com/NechbaMohammed/fastlogreg",
author_email = "mohammednechba@gmail.com",
       packages=['fastlogreg'],
        zip_safe = False)
