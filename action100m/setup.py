from setuptools import setup, find_packages

NAME = "action100m"
VERSION = "0.1.0"
DESCRIPTION = "Action100M video action dataset pipeline"
URL = "https://github.com/facebookresearch/Action100M"

with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    url=URL,
    python_requires=">=3.11",
    install_requires=requirements,
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "action100m=action100m.scripts.run_pipeline:main",
        ],
    },
)
