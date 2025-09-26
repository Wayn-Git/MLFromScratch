import random
import pandas as pd

def train_test_split(df, test_size=20):
    indicies = df.index.to_list()
    test_indecies = random.sample(population=indicies, k=test_size)
    test_data = df.iloc[test_indecies]
    train_data = df.drop(test_indecies)

    train_data = train_data.reset_index()
    test_data = test_data.reset_index()

    return train_data, test_data