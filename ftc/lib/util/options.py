"""
@created_at 2015-01-18
@author Exequiel Fuentes Lettura <efulet@gmail.com>
"""


from argparse import ArgumentParser


class Options:
    """"""
    
    def __init__(self):
        """"""
        self.parser = ArgumentParser(usage='python main.py')
        self._init_parser()
    
    def _init_parser(self):
        """"""
        self.parser.add_argument("-e", metavar="E", type=int,
                                 dest="epochs",
                                 default=100,
                                 help="epochs")
        
        self.parser.add_argument("-H", metavar="H", type=int,
                                 dest="hidden_neurons",
                                 default=5,
                                 help="number of neurons in the hidden layer")
        
        self.parser.add_argument("-l", metavar="L", type=str,
                                 dest="load",
                                 help="load a network")
        
        self.parser.add_argument("-m", metavar="M", type=float,
                                 dest="momentum",
                                 default=0.1,
                                 help="momentum")
        
        self.parser.add_argument("-s", metavar="S", type=str,
                                 dest="save_as",
                                 help="save the network")
        
        self.parser.add_argument("-v", dest="verbose",
                                 action='store_true',
                                 help="verbose")
        
        self.parser.add_argument("-w", metavar="W", type=float,
                                 dest="weightdecay",
                                 default=0.01,
                                 help="weightdecay")
    
    def parse(self, args=None):
        """"""
        return self.parser.parse_args(args)
