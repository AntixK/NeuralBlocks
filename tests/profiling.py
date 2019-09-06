from torch.autograd.profiler import profile
from abc import ABC, abstractmethod

def dec(func):
    def wrapper(self, *args):
        func(self, *args)

    return wrapper

class A(ABC):
    def sheep(self):
        print("sheep")

    @abstractmethod
    def s(self):
        pass


class B(A):
    def baba(self):
        print("Baba")

    @dec
    def black(self, text):
        print("Black " + text)

    def s(self):
        pass

b = B()
b.baba()
b.black("sheep")
# b.sheep()