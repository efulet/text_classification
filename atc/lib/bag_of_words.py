"""
@created_at 2014-11-22
@author Exequiel Fuentes <efulet@gmail.com>
"""


import os
import traceback
import re
import logging

import nltk


class BagOfWords:
    """
    Implements a bag of words. Requires NLTK.
    """
    
    def __init__(self, logger=None):
        self._bag_of_words = {}
        
        self._tokenizer = nltk.RegexpTokenizer(r'\w+')
        self._stopwords = nltk.corpus.stopwords.words("english")
        #self._stemmer = nltk.PorterStemmer() # Original, Porter version 1
        self._stemmer = nltk.SnowballStemmer("english") # Porter version 2
        #self._stemmer = nltk.LancasterStemmer()
        
        self._minlength = 1
        
        self._logger = logger or logging.getLogger(__name__)
    
    def _add_word(self, word):
        """
        Add a word into the dictionary
        """
        if word == None or len(word) == 0:
            return
        
        if word in self._bag_of_words:
            self._bag_of_words[word] += 1
        else:
            self._bag_of_words[word] = 1
    
    def len(self):
        """
        Return bag of words length
        """
        return len(self._bag_of_words)
    
    def words(self):
        """
        Return words in the dictonary
        """
        return self._bag_of_words.keys()
    
    def get_frequency(self, word):
        """
        Return word frequency
        """
        if word in self._bag_of_words:
            return self._bag_of_words[word]
        else:
            return 0
    
    def create_bag_of_words(self, file_path):
        """
        Create a bag of word from a text file.
        """
        # Check file existence
        if os.path.isfile(file_path) == False:
            raise IOError("No such file: " + file_path)
        
        # Load file using NLTK library
        nltk_path = nltk.data.find(file_path)
        raw = open(nltk_path, 'rU').read().decode('utf8')
        
        # http://www.nltk.org/api/nltk.tokenize.html
        tokens = self._tokenizer.tokenize(raw)
        
        # Words to lower case, remove white space and empty words
        tokens = [w.strip().lower() for w in tokens if w.strip()]
        
        # Another normalization task involves identifying non-standard words 
        # including numbers, abbreviations, and dates, and mapping any such 
        # tokens to a special vocabulary.
        
        # Remove numbers
        numbers = re.compile(r'\w*[0-9]\w*')
        #numbers = re.compile(r'[0-9]')
        tokens = [numbers.sub("", w) for w in tokens if numbers.sub("", w)]
        
        # Remove words less than 1 in length
        tokens = [w for w in tokens if len(w) > self._minlength]
        
        # http://nullege.com/codes/search/nltk.corpus.stopwords
        tokens = [w for w in tokens if w not in self._stopwords]
        
        # Stemmers remove morphological affixes from words, leaving only the word stem.
        # It uses Porter method
        # http://www.nltk.org/howto/stem.html
        tokens = [self._stemmer.stem(w) for w in tokens]
        
        # Finally, create the bag of words
        for word in tokens:
            self._add_word(word)
    
    def export_bag_of_words(self):
        """
        Export bag of words as text format
        """
        raise NotImplementedError
