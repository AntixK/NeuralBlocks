import pandas as pd
from IPython.display import display, clear_output


class Logger:
    def __init__(self):
        self.logger = []
        self.counter = 0


    def append(self, list):
        self.logger.append(list)
        self.counter +=1

    def print(self):
        clear_output()
        df = pd.DataFrame(self.logger)
        df.columns  =self.columns
        display(df)
