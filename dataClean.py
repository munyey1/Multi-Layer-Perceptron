import numpy as np
import pandas as pd
import random
from scipy import stats


def clean_data():
    # read the dataset into a dataframe
    df = pd.read_excel(r'DataSet.xlsx')
    # drop the date column
    df = df.drop(['Date'], axis=1)

    # eliminate string outliers
    df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # eliminate outliers 3 s.d. away from the mean
    df = df[(np.abs(stats.zscore(df[df.columns])) < 3).all(axis=1)]
    # mix the dataset randomly
    df_mixed = df.sample(frac=1, random_state=random.randint(1, 100))

    # calculate the number of rows for each subset
    num_rows = len(df_mixed)
    n_data_60 = int(0.6 * num_rows)
    n_data_20 = int(0.2 * num_rows)

    # split dataframe into two subsets, 80:20
    df_train_val = df_mixed[:n_data_60 + n_data_20]
    df_test_20 = df_mixed[n_data_60 + n_data_20:]

    # standardise the dataset carrying training and validation data
    df_train_val_std = 0.8 * ((df_train_val - df_train_val.min()) / (df_train_val.max() - df_train_val.min())) + 0.1

    # split 80 into 60:20
    df_train_60_std = df_train_val_std[:n_data_60]
    df_val_20_std = df_train_val_std[n_data_60:]

    # standardised test subset with own min, max
    df_test_20_std = 0.8 * ((df_test_20 - df_test_20.min()) / (df_test_20.max() - df_test_20.min())) + 0.1

    # write the subsets to separate sheets in an Excel file
    with pd.ExcelWriter('CleanedData.xlsx') as writer:
        df_train_60_std.to_excel(writer, sheet_name='Training Subset', index=False)
        df_val_20_std.to_excel(writer, sheet_name='Validation Subset', index=False)
        df_test_20_std.to_excel(writer, sheet_name='Testing Subset', index=False)
