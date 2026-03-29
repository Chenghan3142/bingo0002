from abc import ABC, abstractmethod

class BaseDataProvider(ABC):
    @abstractmethod
    def fetch_sentiment_data(self, ticker: str):
        pass

    @abstractmethod
    def fetch_fundamental_data(self, ticker: str):
        pass

    @abstractmethod
    def fetch_macro_data(self):
        pass

    @abstractmethod
    def fetch_smart_money_data(self, ticker: str):
        pass
