from abc import ABCMeta, abstractclassmethod


class Product():
    def __init__(self):
        pass

class TaskModuler(ABCMeta):
    def __init__(self):
        pass
    
    def __call__(self, product: Product):
        pass
    
    def create(self, product: Product):
        pass
    
    def destroy(self, product: Product):
        pass
    
class Processor():
    pass
    