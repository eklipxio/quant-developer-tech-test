import csv
from plot_series import plot_series
import numpy as np


def read_cell(file_path, row_index, col_index):
    """
    Reads a 'cell' in CSV file and return value.

    Parameters:
    - file_path: file
    - row_index: column
    - col_index: row

    Returns:
    - result_list: value at (row_index,col_index)
    """
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i == row_index:
                return row[col_index]
    return None  # In case the indices are out of bounds


def csv_to_dicts(file_path, col_debut=None, col_end=None, exclude_empty=True):
    """
    Reads CSV file and return a dictionary.

    Parameters:
    - file_path: file
    - col_debut: starting column to return
    - col_end: ending column to return, excluded

    Returns:
    - result_list: list of dictionaries
    """
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)[col_debut:col_end]  # First row from col_debut to col_end as keys
        result_list = []

        for row in csv_reader:
            values = row[col_debut:col_end]  # Row values starting from col_debut to col_end as keys
            # Create a dictionary and remove keys with blank values if exclude_empty is True
            row_dict = {key: value for key, value in zip(header, values) if value} if exclude_empty else dict(
                zip(header, values))
            result_list.append(row_dict)

    return result_list


def Test():
    clean_path = 'clean.csv'
    noisy_path = 'noisy.csv'

    # Reading the csv file and printing the output
    clean = csv_to_dicts(clean_path, 2)
    noisy = csv_to_dicts(noisy_path, 2)

    print(read_cell(clean_path,1,1))

    key_clean = np.array([float(key) for key in clean[0]])
    value_clean = np.array([float(value) for value in clean[0].values()])
    key_noisy = np.array([float(key) for key in noisy[0]])
    value_noisy = np.array([float(value) for value in noisy[0].values()])

    plot_series(key_clean,value_clean, "Clean Data", "Strike", "Volatility",False)
    plot_series(key_noisy,value_noisy, "Noisy Data", "Strike", "Volatility",False)


def main():
    Test()


if __name__ == "__main__":
    main()
