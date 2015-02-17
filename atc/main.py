"""
@created_at 2014-11-17
@author Exequiel Fuentes <efulet@gmail.com>
"""


import fnmatch
import os
import traceback
import sys
import datetime

from lib import *

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)

# http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python
import logging
import logging.config
logging.config.dictConfig({
    'version': 1,              
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level':'INFO',    
            'class':'logging.StreamHandler',
        },  
    },
    'loggers': {
        '': {                  
            'handlers': ['default'],        
            'level': 'INFO',  
            'propagate': True  
        }
    }
})

# TODO move these methods to util class
def check_version():
    if sys.version_info[:2] != (2, 7):
        raise Exception("This application requires pyhton v2.7")

def db_path():
    pathfile = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pathfile, "db")

def log_path():
    pathfile = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pathfile, "..", "log")

def create_inputs(vocabulary, classes, library):
    net_inputs = []
    
    for topic in classes:
        for root, dirnames, filenames in os.walk(os.path.join(db_path(), library, topic)):
            for filename in fnmatch.filter(filenames, '*.txt'):
                current_net_inputs = []
                current_bag_of_words = BagOfWords()
                current_bag_of_words.create_bag_of_words(os.path.join(root, filename))
                current_vocabulary = current_bag_of_words.words()
                
                # 1 when the word is in the bag, 0 otherwise
                for word in vocabulary:
                    if word in current_vocabulary:
                        current_net_inputs.append(1)
                    else:
                        current_net_inputs.append(0)
                
                # Create the input representation:
                # --> ([0,1...], topic_index)
                net_inputs.append((current_net_inputs, classes.index(topic)))
    
    return net_inputs


if __name__ == "__main__":
    try:
        options = Options().parse(sys.argv[1:])
        
        # Define logger settings
        logger = logging.getLogger(__name__)
        
        # Create a file handler
        handler = logging.FileHandler(os.path.join(log_path(), "atc.log"))
        handler.setLevel(logging.INFO)
         
        # Create a logging format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
         
        # Add the handlers to the logger
        logger.addHandler(handler)
        
        # Define the classes
        classes = ["bioinformatics", "database", "network", "programming"]
        
        # Create the bag of words including all abstracts
        start = datetime.datetime.now()
        bag_of_words = BagOfWords(logger)
        for root, dirnames, filenames in os.walk(os.path.join(db_path(), "acm")):
            for filename in fnmatch.filter(filenames, '*.txt'):
                bag_of_words.create_bag_of_words(os.path.join(root, filename))
        
        # Now, bag of words will be our vocabulary
        vocabulary = bag_of_words.words()
        
        # Create inputs to the network
        net_inputs = create_inputs(vocabulary, classes, "acm")
        logger.info("Time creating vocabulary: %f [sec]" % (datetime.datetime.now() - start).total_seconds())
        
        # Create the network
        fnetwork = FNetwork(net_inputs, classes, options, logger)
        
        # Load the network from file
        if options.load:
            logger.info("Loading network from file...")
            fnetwork.load(options.load)
        else:
            # Fit the network
            start = datetime.datetime.now()
            fnetwork.fit()
            logger.info("Time training network: %f [sec]" % (datetime.datetime.now() - start).total_seconds())
            
            # Show the errors
            if options.verbose == False:
                fnetwork.show_error()
            
            # Save the network after fit
            if options.save_as:
                fnetwork.save(options.save_as)
        
        # Show the network layer
        #if options.verbose:
            fnetwork.show_layer()
        
        logger.info("Testing the network using validation dataset from IEEE...")
        start = datetime.datetime.now()
        validation_dataset = create_inputs(vocabulary, classes, "ieee")
        logger.info("Time creating IEEE input: %f [sec]" %(datetime.datetime.now() - start).total_seconds())
        
        validation_target = []
        for i in xrange(len(validation_dataset)):
            validation_target.append(validation_dataset[i][1])
        
        start = datetime.datetime.now()
        validation_pred = fnetwork.predict(validation_dataset)
        logger.info("Time predicting IEEE target: %f [sec]" % (datetime.datetime.now() - start).total_seconds())
        
        logger.info("Classification performance: %.2f" % fnetwork.classification_performance(validation_pred, validation_target))
        logger.info("Explained sum of squares: %.2f" % fnetwork.explained_sum_squares(validation_pred, validation_target))
        logger.info("Mean squared error: %.2f" % fnetwork.mean_squared_error(validation_pred, validation_target))
        
        # Create a metrics instance for showing a report in the console
        fmetrics = FMetrics(logger)
        
        # Create the report
        fmetrics.report(validation_target, validation_pred)
        
        # For keeping the plot
        fnetwork.show_plot()
    except Exception, err:
        logger.error(str(err), exc_info=True)
    finally:
        sys.exit()

