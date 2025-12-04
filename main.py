from torch import tensor, no_grad, float32, save as t_save, load as t_load
from torch.nn import Module, RNN, Linear, MSELoss
from torch.optim import Adam

import matplotlib.pyplot as plt

from json import dump, load
from numpy import array

from dataset import CoastData

LR = 0.00001
DAYS = 100


class SimpleRNN(Module):
    def __init__(self):
        super(SimpleRNN, self).__init__()

        self.rnn1 = RNN(1, 24 * 3 * 3, batch_first=True)
        self.fc = Linear(24 * 3 * 3, 1)

    def forward(self, x):
        out, _ = self.rnn1(x)
        out = out[-1]
        out = self.fc(out)
        return out


class Net:
    def __init__(self):
        self.model = SimpleRNN()

        self.criterion = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=LR)

        self.coast_data = CoastData(days=DAYS)

        self.loss_list = []
        self.error_rate_list = []

        print("The network has been initialized")

    def load_settings(self):
        self.model.load_state_dict(
            t_load(f'data/checkpoints.pt', weights_only=True)
        )

    def load_data(self):
        with open(
                f"data/loss_list.json", "r", encoding="utf8"
        ) as file:
            self.loss_list += load(file)

        with open(
                f"data/error_rate_list.json", "r", encoding="utf8"
        ) as file:
            self.error_rate_list += load(file)

    def save_settings(self):
        t_save(self.model.state_dict(), f'data/checkpoints.pt')

    def save_data(self):
        with open(
                f"data/loss_list.json", "w", encoding="utf8"
        ) as file:
            dump(self.loss_list, file, indent=4)

        with open(
                f"data/error_rate_list.json", "w", encoding="utf8"
        ) as file:
            dump(self.error_rate_list, file, indent=4)

    def training(self, epoch=10, testing_step=1):
        training_data, target_data = self.coast_data.get_training_data()

        print(f"Training dataset size: {training_data.shape[0]}")

        self.testing(with_save=False)

        epoch_counter = 0
        while epoch_counter != epoch:
            epoch_counter += 1

            self.model.train()

            loss_data = []

            for index in range(len(training_data)):
                outputs = self.model(training_data[index].unsqueeze(1))

                target = tensor([target_data[index]])
                loss = self.criterion(outputs, target)

                loss_data.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss = sum(loss_data) / len(loss_data)
            print(f'Epoch [{epoch_counter:5}], Loss: [{loss:.6f}]')
            self.loss_list.append(loss)

            if epoch % testing_step == 0:
                self.testing()

    def testing(self, with_save=True):
        testing_data, target_data, compare_date = (
            self.coast_data.get_testing_data())

        self.model.eval()

        test_counter = 0
        error_rate = 0

        with no_grad():
            for index in range(len(testing_data)):
                predictions = self.model(
                    testing_data[index].unsqueeze(1)).numpy()
                error_rate += abs(predictions[0] - target_data[index])

                if (target_data[index] - compare_date[index]
                ) * (predictions[0] - compare_date[index]) > 0:
                    test_counter += 1

        print(f"Direction:  {(test_counter / len(testing_data)) * 100:.1f} %")
        _, ratio = self.coast_data.get_normalize_data()
        error_rate = (error_rate / len(testing_data)) * ratio
        print(f"Error rate: {error_rate:.2f} RUB\n")
        if with_save:
            self.error_rate_list.append(float(error_rate))

    def show_test(self):
        all_testing_data, testing_data = self.coast_data.get_all_testing_data()

        predictions = []

        with no_grad():
            for index in range(len(testing_data)):
                predictions.append(
                    self.model(
                        testing_data[index].unsqueeze(1)
                    ).numpy()
                )

        predictions = tensor(array(predictions), dtype=float32)

        start_data, ratio = self.coast_data.get_normalize_data()

        all_testing_data *= ratio
        predictions *= ratio

        all_testing_data += start_data
        predictions += start_data

        plt.figure(figsize=(10, 6))

        plt.plot(all_testing_data, label='Target data')
        plt.plot(predictions, label='Predicted')

        plt.legend()
        plt.show()

    def show_training(self):
        plt.figure(figsize=(10, 6))

        error_rate = []
        for data in self.error_rate_list:
            error_rate.append(data / 100)

        print(error_rate)

        plt.plot(self.loss_list, label='Loss')
        plt.plot(error_rate, label='Error rate')

        plt.legend()
        plt.show()


def main():
    net = Net()

    # net.load_settings()
    # net.load_data()

    # net.training(epoch=10)
    # net.testing(with_save=False)

    # net.save_settings()
    # net.save_data()

    net.show_test()
    net.show_training()


if __name__ == "__main__":
    main()
