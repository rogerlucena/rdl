import pandas as pd
from random import randrange

NUMBER_ANNOUNCERS = 10


def read_data(file_name):
    data = []
    i = 0
    with open(file_name) as data_file:
        read_data = data_file.read()
        data[i] = read_data
        i += 1
    return data


# Baselines - cheating
def random_strategy():
    return randrange(NUMBER_ANNOUNCERS)


def argmax(array):
    return max(range(len(array)), key=lambda i: array[i])


def static_best(memory):
    rate_sum = [sum(x) for x in zip(*memory)]
    best_announcer = argmax(rate_sum)

    def best_average():
        return best_announcer

    return best_average


def optimal(click_rates):
    return argmax(click_rates)


# Algorithms
def UCB():
    pass


def LinUCB():
    pass


# Strategy class
# class Strategy:
#     def __init__(self):
#         pass

#     def get_choice(self):
#         pass

#     def update(self):
#         pass

if __name__ == "__main__":
    print(random_strategy())
    memory = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    print(static_best(memory)())
    print(optimal(range(10)))