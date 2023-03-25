import pandas as pd
from MLP import MLP
from dataClean import clean_data

# clean the dataset
clean_data()

# get the training, validation and testing subset
df = pd.read_excel(r'CleanedData.xlsx', sheet_name=None)
df_train = df.get('Training Subset')
df_valid = df.get('Validation Subset')
df_test = df.get('Testing Subset')

# get the number of predictors, assuming the last column of the dataset is the predictand
n_predictors = len(df_train.iloc[[0]].values[0]) - 1

# allow the user to enter the number of hidden nodes in the MLP
# allow the user to enter the number of epochs to train
n_nodes = int(input("Enter the number of hidden nodes: "))
epochs = int(input("Enter the number of epochs: "))

# create MLP object and give it the training and validation subsets, along with the number of hidden
# nodes and predictors
mlp1 = MLP(df_train, df_valid, n_nodes, n_predictors)
# train the MLP with user given epochs and get the training, validation and testing errors
trn_vld_errors = mlp1.train(epochs)
test_error = mlp1.test(df_test)

# write the MLP performance to a text file
with open("MLPperformance.txt", "a") as f:
    f.write(f"Hidden nodes: {n_nodes},Epochs: {epochs}, Training error: {trn_vld_errors[0]}, Validation error: {trn_vld_errors[1]},"
            f"Test error: {test_error}\n")
