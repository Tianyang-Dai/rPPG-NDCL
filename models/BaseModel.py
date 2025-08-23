from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, config):
        self.isTrain = True
        self.config = config

    @abstractmethod
    def optimize_parameters(self, data):
        pass

    @abstractmethod
    def forward(self, data):
        pass
