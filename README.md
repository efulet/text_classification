#Welcome to Text Classification Proofs of Concept

This prototype is a proofs of concept for text classification using different 
Machine Learning methods. For starting, several abstracts from scientific papers 
have been downloaded from ACM Digital Library and IEEE Digital Library. The 
topics/classes are the following:
* Bioinformatics
* Database
* Network
* Programming language (just Programming for this aplication)

The prototype is using 80 abstracts in total (20 by topic) from ACM Digital 
Library for trainig and 20  abstracts (5 by topic) from IEEE Digital Library for
testing.

Currently, this prototype is implementing just Artificial Neural Networks concept.

##Requirements

* Python 2.7 (Other versions haven't been tested yet)
* Setuptools (http://pypi.python.org/pypi/setuptools)
```bash
$> wget https://bootstrap.pypa.io/ez_setup.py -O - | sudo python
```
* Pip (https://pip.pypa.io/en/latest/installing.html)
```bash
$> sudo easy_install pip
```
* Numpy (http://www.scipy.org/scipylib/download.html)
```bash
$> sudo pip install -U numpy
```
* Scikit-learn (http://scikit-learn.org/stable/install.html)

    NOTE: Scikit-learn requires
      - Python >= 2.6 or >= 3.3
      - NumPy >= 1.6.1
      - SciPy >= 0.9
      
```bash
$> sudo pip install -U scikit-learn
```
* NLTK (http://www.nltk.org/install.html)
```bash
$> sudo pip install -U nltk
```
* Open a Python SHELL and type:
```bash
$> python
```
```python
import nltk
nltk.download()
```
* PyBrain (http://pybrain.org/)
```bash
$> cd /usr/local/src
$> git clone git://github.com/pybrain/pybrain.git
$> cd pybrain
$> python setup.py install
```

## Run

Print program options:
```bash
$> python ftc/main.py -h
usage: python main.py

optional arguments:
    -h, --help  show this help message and exit
    -e E        epochs
    -H H        number of neurons in the hidden layer
    -l L        load a network
    -m M        momentum
    -s S        save the network
    -v          verbose
    -w W        weightdecay
```

Example:
```bash
    $> python ftc/main.py -e 500 -v -s mynet.net
```
