Welcome to Artificial Neural Network Text Classification
========================================================

This program is a proof of concept for text classification using a Artificial 
Neural Network. This network classifies papers using their abstracts. The 
topics/classes are the following:

  * Bioinformatics
  * Database
  * Network
  * Programming language (just Programming for this aplication)

The network are using 80 abstracts in total (20 by topic) for training and 
validation. These abstracts were downloaded from ACM Digital Library. 20 
abstracts (5 by topic) from IEEE Digital Library are used for testing the model.


Requirements
------------

+ Python 2.7 (Other versions haven't been tested yet)

+ Setuptools (http://pypi.python.org/pypi/setuptools)

    $> wget https://bootstrap.pypa.io/ez_setup.py -O - | sudo python

+ Pip (https://pip.pypa.io/en/latest/installing.html)

    $> sudo easy_install pip

+ Numpy (http://www.scipy.org/scipylib/download.html)

    $> sudo pip install -U numpy

+ Scikit-learn (http://scikit-learn.org/stable/install.html)

    NOTE: Scikit-learn requires
      - Python >= 2.6 or >= 3.3
      - NumPy >= 1.6.1
      - SciPy >= 0.9
      
    $> sudo pip install -U scikit-learn

+ NLTK (http://www.nltk.org/install.html)

    $> sudo pip install -U nltk

+ Open a Python SHELL and type:
    
    $> python
    
    import nltk
    
    nltk.download()

+ PyBrain (http://pybrain.org/)
    
    $> cd /usr/local/src
    
    $> git clone git://github.com/pybrain/pybrain.git
    
    $> cd pybrain
    
    $> python setup.py install


Run
---

Print program options:
    
    $> python atc/main.py -h
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

Example:
    
    $> python atc/main.py -e 500 -v -s mynet.net
