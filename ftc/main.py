"""
@created_at 2015-01-18
@author Exequiel Fuentes <efulet@gmail.com>
"""


import sys
import datetime

from lib.util import Options, SystemUtils
from lib.doc import BagOfWords
from lib.net import Network
from lib.stats import FMetrics

#import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":
    try:
        # Parse options
        options = Options().parse(sys.argv[1:])
        
        # Configure logging
        logger = SystemUtils().configure_log()
        
        # Define the classes
        classes = ["bioinformatics", "database", "network", "programming"]
        
        # Create bag of words
        bag_of_words = BagOfWords()
        vocabulary = bag_of_words.create_vocabulary()
        
        # Create inputs to the network
        net_inputs = bag_of_words.create_ann_inputs(vocabulary, classes, "acm")
        
        # Create the network
        network = Network(net_inputs, classes, options, logger)
        
        # Load the network from file
        if options.load:
            logger.info("Loading network from file...")
            network.load(options.load)
        else:
            # Fit the network
            start = datetime.datetime.now()
            network.fit()
            logger.info("Time training network: %f [sec]" % (datetime.datetime.now() - start).total_seconds())
            
            # Show the errors
            if options.verbose == False:
                network.show_error()
            
            # Save the network after fit
            if options.save_as:
                network.save(options.save_as)
        
        # Show the network layer
        #if options.verbose:
            network.show_layer()
        
        logger.info("Testing the network using validation dataset from IEEE...")
        start = datetime.datetime.now()
        validation_dataset = bag_of_words.create_ann_inputs(vocabulary, classes, "ieee")
        logger.info("Time creating IEEE input: %f [sec]" %(datetime.datetime.now() - start).total_seconds())
        
        validation_target = []
        for i in xrange(len(validation_dataset)):
            validation_target.append(validation_dataset[i][1])
        
        start = datetime.datetime.now()
        validation_pred = network.predict(validation_dataset)
        logger.info("Time predicting IEEE target: %f [sec]" % (datetime.datetime.now() - start).total_seconds())
        
        logger.info("Classification performance: %.2f" % network.classification_performance(validation_pred, validation_target))
        logger.info("Explained sum of squares: %.2f" % network.explained_sum_squares(validation_pred, validation_target))
        logger.info("Mean squared error: %.2f" % network.mean_squared_error(validation_pred, validation_target))
        
        # Create a metrics instance for showing a report in the console
        fmetrics = FMetrics(logger)
        
        # Create the report
        fmetrics.report(validation_target, validation_pred)
        
        # For keeping the plot
        network.show_plot()
    except Exception, err:
        error_msg = str(err)
        try:
            logger.error(error_msg, exc_info=True)
        except Exception, err:
            print(error_msg)
            print str(err)
    finally:
        sys.exit()
