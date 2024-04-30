from setuptools import setup, find_packages

with open('README.md','r', encoding="utf-8") as f:
    descriptions = f.read()


setup(
    name= "Food Image Classification",
    version= "0.0.1",
    author= "Vraj Bhavsar",
    author_email= "vrajcbhavsar0905@gmail.com",
    description= "small python package for Classification app.",
    long_description= descriptions,
    long_description_content = "text/markdown",
    url = "https://github.com/vrjbhvsr/Food_image_classification",
    package_dir= {"":"src"},
    packages= find_packages(where="src")

)