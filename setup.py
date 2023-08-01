from setuptools import setup
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name = "fastlogistic",
   version = "0.1",
     author = "Nechba Mohammed, Mouhajir Mohamed, and Sedjari Yassine",
description = "Efficient logistic regression in Python with GPU support",
long_description = long_description,
long_description_content_type = "text/markdown",
url = "https://github.com/NechbaMohammed/fastlogistic",
author_email = "mohammednechba@gmail.com",
       packages=['fastlogistic'],
        zip_safe = False)
