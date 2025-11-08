import os

class Logger:
    def __init__(self, loc):
        self.loc = loc
        try:
            os.remove(loc)
        except FileNotFoundError:
            pass

    def log(self, msg):
        with open(self.loc, 'a') as file:
            file.write(msg + '\n')
