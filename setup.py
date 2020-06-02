from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))

with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


with open(path.join(this_dir, "VERSION")) as f:
    version = f.read()


setup(
    name="gretel-tools",
    version=version,
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="General tools for data science, machine learning, and data engineering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={},
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        'dataclasses;python_version<"3.7"',
        "smart_open==2.0.0",
        "gensim==3.8.3",
        "numpy",
        "pandas",
    ],
)
