import sys
import argparse
import numpy as np
from utils import load_data
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Dataset separation")
    parser.add_argument('--train_size', type=int, default=80, choices=range(0, 100),
                        help="Percentage reserved to training")
    args = parser.parse_args()
    test_size = (100 - int(args.train_size)) / 100

    data_path = None
    while (data_path is None):
        data_path = input("dataset: ")
        data = load_data(data_path)
        if data is None:
            print("Error: dataset not found.", file=sys.stderr)
            data_path = None

    data = data[:, 1:]
    data[:, 0] = np.where(data[:, 0] == "M", 1, 0)
    data = data[:, :].astype(float)

    X_train, X_validation = train_test_split(data, test_size=test_size, random_state=42)
    np.savetxt("data/data_train.csv", X_train, delimiter=',', fmt='%s')
    print(f"{args.train_size}% of the dataset has been saved at data/data_train.csv.")
    np.savetxt("data/data_validation.csv", X_validation, delimiter=',', fmt='%s')
    print(f"{100 - args.train_size}% of the dataset has been saved at data/data_validate.csv.")


if __name__ == "__main__":
    main()
