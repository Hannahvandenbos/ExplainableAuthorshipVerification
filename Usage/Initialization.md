# Initialization

After downloading the project from github, install the necessary packages by running the following, we recommend to install it in a virtual (conda) environment:

```sh
# Upgrade pip
pip install --upgrade pip
```

```sh
# Install and build GuidedLDA
git clone https://github.com/vi3k6i5/GuidedLDA
cd GuidedLDA
sh build_dist.sh
python setup.py sdist
pip install -e .
cd ..
```

```sh
# Install dependencies
pip install -r requirements.txt
```

```sh
# Download Spacy model
python -m spacy download en_core_web_sm
```
