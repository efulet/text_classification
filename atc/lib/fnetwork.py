"""
@created_at 2014-11-22
@author Exequiel Fuentes <efulet@gmail.com>
"""


import logging

from pybrain.datasets import ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.validation import Validator

# Only needed for data generation and graphical output
import pylab as pl

import numpy as np

# Only needed for saving and loading trained network
import pickle
import os

from fnetwork_exception import FNetworkException


class FNetwork:
    # Define the split proportion into 75% training and 25% test data sets
    SPLIT_PROPORTION = 0.25
    
    # Define 5 hidden units
    HIDDEN_NEURONS = 5
    
    # Define the momentum, which is the ratio by which the gradient of the last 
    # timestep is used
    MOMENTUM = 0.1
    
    # Weightdecay corresponds to the weightdecay rate, where 0 is no weight decay at all.
    WEIGHTDECAY = 0.01
    
    # Define epochs
    EPOCHS = 100
    
    def __init__(self, input, classes, options, logger=None):
        """
        :param input: Dataset
        :param classes: Class values
        :param options: Optional vales
        :param logger: logger object [opcional]
        """
        if input == None or len(input) == 0:
            raise FNetworkException("Empty dataset")
        self._input = input
        
        if classes == None or len(classes) == 0:
            raise FNetworkException("Empty class vector")
        self._classes = classes
        
        self._options = options
        
        if self._options.hidden_neurons:
            self._hidden_neurons = self._options.hidden_neurons
        else:
            self._hidden_neurons = FNetwork.HIDDEN_NEURONS
        
        if self._options.momentum:
            self._momentum = self._options.momentum
        else:
            self._momentum = FNetwork.MOMENTUM
        
        if self._options.weightdecay:
            self._weightdecay = self._options.weightdecay
        else:
            self._weightdecay = FNetwork.WEIGHTDECAY
        
        if self._options.epochs:
            self._epochs = self._options.epochs
        else:
            self._epochs = FNetwork.EPOCHS
        
        if self._options.verbose:
            self._verbose = True
        else:
            self._verbose = False
        
        self._logger = logger or logging.getLogger(__name__)
        
        self._dataset = None
        self._X_train = None
        self._X_test = None
        
        self._feed_forward_network = None
        
        self._X_train_results = []
        self._X_test_results = []
    
    def fit(self):
        """
        Fit network using PyBrain library
        """
        # Create the dataset
        # http://pybrain.org/docs/api/datasets/classificationdataset.html
        self._dataset = ClassificationDataSet(len(self._input[0][0]), 1, \
                                           nb_classes=len(self._classes), \
                                           class_labels=self._classes)
        
        # Add samples
        # http://pybrain.org/docs/tutorial/fnn.html
        for sample in self._input:
            self._dataset.addSample(sample[0], [sample[1]])
        
        # Print statistics
        #print self._dataset.calculateStatistics()
        
        # Randomly split the dataset into 75% training and 25% test data sets. 
        # Of course, we could also have created two different datasets to begin with.
        self._X_test, self._X_train = self._dataset.splitWithProportion(FNetwork.SPLIT_PROPORTION)
        
        # For neural network classification, it is highly advisable to encode 
        # classes with one output neuron per class. Note that this operation 
        # duplicates the original targets and stores them in an (integer) field 
        # named 'class'.
        self._X_train._convertToOneOfMany()
        self._X_test._convertToOneOfMany()
        
        if self._verbose:
            # Test our dataset by printing a little information about it.
            self._logger.info("Number of training patterns: %4d" % len(self._X_train))
            self._logger.info("Input dimensions: %4d" % self._X_train.indim)
            self._logger.info("Output dimensions: %4d" % self._X_train.outdim)
            #print "First sample (input, target, class):"
            #print self._X_train['input'][0], self._X_train['target'][0], self._X_train['class'][0]
        
        # Now build a feed-forward network with 5 hidden units. We use the shortcut 
        # buildNetwork() for this. The input and output layer size must match 
        # the dataset's input and target dimension. You could add additional 
        # hidden layers by inserting more numbers giving the desired layer sizes.
        # The output layer uses a softmax function because we are doing classification. 
        # There are more options to explore here, e.g. try changing the hidden 
        # layer transfer function to linear instead of (the default) sigmoid.
        self._feed_forward_network = buildNetwork(self._X_train.indim, \
                                                  self._hidden_neurons, \
                                                  self._X_train.outdim, \
                                                  outclass=SoftmaxLayer)
        
        # Set up a trainer that basically takes the network and training dataset 
        # as input. We are using a BackpropTrainer for this.
        trainer = BackpropTrainer(self._feed_forward_network, dataset=self._X_train, \
                                  momentum=self._momentum, verbose=self._verbose, \
                                  weightdecay=self._weightdecay)
        
        # Start the training iterations
        epoch_results = []
        train_error_results = []
        test_error_results = []
        
        for i in xrange(self._epochs):
            # Train the network for some epochs. Usually you would set something 
            # like 5 here, but for visualization purposes we do this one epoch 
            # at a time.
            trainer.trainEpochs(1)
            
            # http://pybrain.org/docs/api/supervised/trainers.html
            X_train_result = percentError(trainer.testOnClassData(), self._X_train['class'])
            X_test_result = percentError(trainer.testOnClassData(dataset=self._X_test), self._X_test['class'])
            
            # Store the results
            epoch_results.append(trainer.totalepochs)
            train_error_results.append(X_train_result)
            test_error_results.append(X_test_result)
            
            if (trainer.totalepochs == 1 or trainer.totalepochs % 10 == 0 or \
                trainer.totalepochs == self._epochs) and self._verbose:
                    self._logger.info("Epoch: %4d" % trainer.totalepochs +
                          "  Train error: %5.2f%%" % X_train_result +
                          "  Test error: %5.2f%%" % X_test_result)
                    
                    # Now, plot the train and test data
                    pl.figure(1)
                    pl.ioff() # interactive graphics off
                    pl.clf() # clear the plot
                    pl.hold(True) # overplot on
                    
                    pl.plot(epoch_results, train_error_results, 'b', 
                            epoch_results, test_error_results, 'r')
                    pl.xlabel('Epoch number')
                    pl.ylabel('Error')
                    pl.legend(['Training result', 'Test result'])
                    pl.title('Training/Test results')
                    
                    pl.ion() # interactive graphics on
                    pl.draw() # update the plot
        
        if self._verbose:
            # Print network coefs
            #self._logger.info(self._feed_forward_network['in'].outputbuffer[self._feed_forward_network['in'].offset])
            #self._logger.info(self._feed_forward_network['hidden0'].outputbuffer[self._feed_forward_network['hidden0'].offset])
            #self._logger.info(self._feed_forward_network['out'].outputbuffer[self._feed_forward_network['out'].offset])
            
            # Finally, keep showing the plot.
            pl.ioff()
        
        # Store the results
        self._X_train_results = (epoch_results, train_error_results)
        self._X_test_results = (epoch_results, test_error_results)
    
    def predict(self, validation_dataset):
        """
        Generate predictions
        
        :param validation_dataset: Validation dataset
        """
        y_pred = []
        
        for i in xrange(len(validation_dataset)):
            output = self._feed_forward_network.activate(validation_dataset[i][0])
            class_index = max(xrange(len(output)), key=output.__getitem__)
            y_pred.append(class_index)
        
        return y_pred
    
    def classification_performance(self, output, target):
        """
        Returns the hit rate of the outputs compared to the targets.
        http://pybrain.org/docs/api/tools.html#pybrain.tools.validation.Validator.classificationPerformance
        """
        return Validator.classificationPerformance(np.array(output), np.array(target))
    
    def explained_sum_squares(self, output, target):
        """
        Returns the explained sum of squares (ESS).
        http://pybrain.org/docs/api/tools.html#pybrain.tools.validation.Validator.ESS
        """
        return Validator.ESS(np.array(output), np.array(target))
    
    def mean_squared_error(self, output, target):
        """
        Returns the mean squared error. The multidimensional arrays will get 
        flattened in order to compare them.
        http://pybrain.org/docs/api/tools.html#pybrain.tools.validation.Validator.MSE
        
        """
        return Validator.MSE(np.array(output), np.array(target))
    
    def show_plot(self):
        pl.show()
    
    def show_error(self):
        """
        Show training and test process versus epochs
        """
        pl.figure(1)
        
        pl.plot(self._X_train_results[0], self._X_train_results[1], 'b', 
                self._X_test_results[0], self._X_test_results[1], 'r')
        pl.xlabel('Epoch number')
        pl.ylabel('Error')
        pl.legend(['Training result', 'Test result'])
        pl.title('Training/Test results')
        pl.draw()
    
    def show_layer(self):
        """
        Show network layers in text format
        """
        for mod in self._feed_forward_network.modules:
            print "Module:", mod.name
            
            if mod.paramdim > 0:
                print "--parameters:", mod.params
            
            for conn in self._feed_forward_network.connections[mod]:
                print "-connection to", conn.outmod.name
                if conn.paramdim > 0:
                    print "- parameters", conn.params
            
            if hasattr(self._feed_forward_network, "recurrentConns"):
                print "Recurrent connections"
                for conn in self._feed_forward_network.recurrentConns:
                    print "-", conn.inmod.name, " to", conn.outmod.name
                    if conn.paramdim > 0:
                        print "- parameters", conn.params
    
    def save(self, file_path):
        """
        Save network
        """
        try:
            file_net = None
            
            file_net = open(file_path, 'w')
            pickle.dump(self._feed_forward_network, file_net)
        except Exception, err:
            raise FNetworkException(str(err))
        finally:
            if file_net != None:
                file_net.close()
    
    def load(self, file_path):
        """
        Load network from file
        """
        try:
            file_net = None
            
            if os.path.isfile(file_path) == False:
                raise FNetworkException("No such file: " + file_path)
            
            file_net = open(file_path,'r')
            self._feed_forward_network = pickle.load(file_net)
        except Exception, err:
            raise FNetworkException(str(err))
        finally:
            if file_net != None:
                file_net.close()
