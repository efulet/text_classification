"""
@created_at 2014-12-07
@author Exequiel Fuentes <efulet@gmail.com>
"""


from sklearn import metrics

import matplotlib.pyplot as plt

class FMetrics:
    def __init__(self, logger=None):
        self._logger = logger or logging.getLogger(__name__)
    
    def report(self, y_test, y_pred):
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
