from setuptools import setup
from pathlib import Path

# Read the contents of your README file
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="FastLogisticReg",
    version="0.0.1",
    author="Nechba Mohammed, Mouhajir Mohamed, and Sedjari Yassine",
    description="Efficient logistic regression in Python with GPU support",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Corrected the content type to "text/markdown"
    url="https://github.com/NechbaMohammed/FastLogisticReg",
    author_email="mohammednechba@gmail.com",
    packages=['FastLogisticReg'],
)
