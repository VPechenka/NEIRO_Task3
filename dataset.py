from datetime import timedelta

from tinkoff.invest import CandleInterval, Client, Quotation
from tinkoff.invest.schemas import CandleSource
from tinkoff.invest.utils import now

import matplotlib.pyplot as plt
from torch import tensor, float32
from numpy import array

from pprint import pprint

TOKEN = "ТОКЕН"


class CoastData:
    def __init__(self, days=365):
        self.days = days

        self.coast_data = self.get_close_data()

        self.start_data = 0
        self.ratio = 0

        self.normalize()

        self.training_data = []
        self.target_data = []

        self.all_testing_data = []
        self.testing_data = []
        self.test_target_data = []
        self.compare_data = []

        self.create_datasets()

    def get_close_data(self) -> list[float]:
        coast_data = []

        with Client(TOKEN) as client:
            candles_data = [*client.get_all_candles(
                instrument_id="BBG004730N88",  # << Сбер >>
                from_=now() - timedelta(days=self.days),
                interval=CandleInterval.CANDLE_INTERVAL_HOUR,
                candle_source_type=CandleSource.CANDLE_SOURCE_UNSPECIFIED,
            )]

            for candle_data in candles_data:
                coast_data.append(self.to_rubles(candle_data.close))

        return coast_data

    @staticmethod
    def to_rubles(quotation: Quotation) -> float:
        return quotation.units + quotation.nano / 1_000_000_000

    def normalize(self) -> None:
        min_data = min(self.coast_data)
        max_data = max(self.coast_data) - min_data
        for index in range(len(self.coast_data)):
            self.coast_data[index] = (self.coast_data[
                                          index] - min_data) / max_data

        self.start_data = min_data
        self.ratio = max_data

    def create_datasets(self) -> None:
        training_data = self.coast_data[:-(24 * 3 + 1 + 100)]

        for index in range(0, len(training_data) - (24 * 3 + 1)):
            self.training_data.append(training_data[index:index + (24 * 3)])
            self.target_data.append(training_data[index + (24 * 3 + 1)])

        testing_data = self.coast_data[-(24 * 3 + 1 + 100):]
        for index in range(0, len(testing_data) - (24 * 3 + 1)):
            self.testing_data.append(testing_data[index:index + (24 * 3)])
            self.test_target_data.append(testing_data[index + (24 * 3 + 1)])
            self.compare_data.append(testing_data[index + (24 * 3)])

        self.all_testing_data = testing_data[(24 * 3 + 1):]

    def get_training_data(self):
        return (
            tensor(array(self.training_data), dtype=float32),
            tensor(array(self.target_data), dtype=float32)
        )

    def get_testing_data(self):
        return (
            tensor(array(self.testing_data), dtype=float32),
            self.test_target_data,
            self.compare_data
        )

    def get_all_testing_data(self):
        return (
            tensor(array(self.all_testing_data), dtype=float32),
            tensor(array(self.testing_data), dtype=float32)
        )

    def get_normalize_data(self):
        return self.start_data, self.ratio

    def show_graphic(self):
        candles_data = tensor(self.coast_data, dtype=float32)
        plt.figure(figsize=(10, 6))
        plt.plot(candles_data.numpy(), label='True')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    data = CoastData()
    data.show_graphic()
