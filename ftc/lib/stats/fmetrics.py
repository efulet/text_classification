"""
@created_at 2015-01-18
@author Exequiel Fuentes Lettura <efulet@gmail.com>
"""


from sklearn import metrics

import matplotlib.pyplot as plt

from lib.util import SystemUtils


class FMetrics:
    """"""
    
    def __init__(self, logger=None):
        """"""
        self._logger = logger or SystemUtils().configure_log()
    
    def report(self, y_test, y_pred):
        """
        Create a report with several metrics
        """        
        # **********************************************************************
        # The mean_absolute_error function computes the mean absolute error, 
        # which is a risk function corresponding to the expected value of the 
        # absolute error loss.
        #self._logger.info("Mean absolute error: %.4f" % metrics.mean_absolute_error(y_test, y_pred))
        
        # **********************************************************************
        # The mean_squared_error function computes the mean square error, 
        # which is a risk function corresponding to the expected value of the 
        # squared error loss or quadratic loss
        #self._logger.info("Mean square error: %.4f" % metrics.mean_squared_error(y_test, y_pred))
        
        # Accuracy classification score.
        #self._logger.info("Accuracy score: %.4f" % metrics.accuracy_score(y_test, y_pred))
        
        # **********************************************************************
        # The confusion_matrix function computes the confusion matrix to 
        # evaluate the accuracy on a classification problem
        cm = metrics.confusion_matrix(y_test, y_pred)
        self._logger.info("Confusion matrix:")
        self._logger.info(cm)
        
        # Show confusion matrix in a separate window
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
