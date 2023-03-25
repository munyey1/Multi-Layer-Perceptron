import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats


class MLP:
    p = 0.1

    # initialise the MLP
    def __init__(self, df_train, df_valid, hidden_nodes, n_predictors):
        # store the datasets and number of hidden nodes and predictors
        self.df_train = df_train
        self.df_valid = df_valid
        self.hidden_nodes = hidden_nodes
        self.n_predictors = n_predictors

        # randomise the weights
        self.in_weights = np.random.uniform(-2/n_predictors, 2/n_predictors, (n_predictors, hidden_nodes))
        self.in_weights = np.round(self.in_weights, 2)
        self.out_weights = np.random.uniform(-2/n_predictors, 2/n_predictors, hidden_nodes)
        self.out_weights = np.round(self.out_weights, 2)

        # randomise the biases
        self.biases = np.random.uniform(-2/n_predictors, 2/n_predictors, hidden_nodes + 1)
        self.biases = np.round(self.biases, 2)
        # set the bias of the output node
        self.biases[-1] = round(random.uniform(-2/hidden_nodes, 2/hidden_nodes), 2)

    def forward_pass(self, values):
        # values - predictor values
        s_value = 0
        u_values = []
        u_output = 0
        # for each hidden node calculate it's S value and U value
        # calculate the output at the end
        for i in range(self.hidden_nodes):
            # for each predictor
            for j in range(self.n_predictors):
                # calculate S value, predictor * weight
                s_value += values[j] * self.in_weights[j][i]
            s_value += self.biases[i]
            # calculate U value for each hidden node using the sigmoid activation function
            u_value = 1 / (1 + np.exp(-s_value))
            u_values.append(u_value)
            u_output += u_value * self.out_weights[i]

        output = 1 / (1 + np.exp(-u_output))
        # return output and U values as they are needed for updating the weights and biases
        return output, u_values

    def backpass(self, values, output_values):
        momentum = 0.9
        # calculate delta values for nodes
        output = output_values[0]
        output_delta = (values[5] - output) * (output * (1 - output))

        # for each hidden node, calculate the delta values and update the weights and biases
        for i in range(self.hidden_nodes):
            delta = (self.out_weights[i] * output_delta) * (output_values[1][i] * (1 - output_values[1][i]))
            # for each predictor, update each weight it is connected to
            for j in range(self.n_predictors):
                temp = self.in_weights[j][i]
                self.in_weights[j][i] += MLP.p * delta * values[j]
                weight_change = self.in_weights[j][i] - temp
                self.in_weights[j][i] += momentum * weight_change

            # update the out-going weights of the hidden node
            temp = self.out_weights[i]
            self.out_weights[i] += MLP.p * output_delta * output_values[1][i]
            weight_change = self.out_weights[i] - temp
            self.out_weights[i] += momentum * weight_change
            # update biases
            self.biases[i] += MLP.p * delta

        # update the output node bias accordingly
        self.biases[-1] += MLP.p * output_delta

    def train(self, epochs):
        train_errors = []
        valid_errors = []
        train_error = 0
        valid_error = 0
        train_len = len(self.df_train)
        valid_len = len(self.df_valid)
        # for each epoch
        for i in range(epochs):
            # for each row in the training subset
            for j in range(train_len):
                # get predictors and calculate values for the hidden nodes
                t_values = self.df_train.iloc[[j]].values[0]
                train_values = self.forward_pass(t_values)
                output = train_values[0]
                # update weights and biases
                self.backpass(t_values, train_values)
                train_error += (t_values[5] - output)**2
            # calculate training error using RMSE
            train_error = np.sqrt(train_error / train_len)
            train_errors.append(train_error)

            # for each row in the validation subset
            for k in range(valid_len):
                # get predictors and calculate values for the hidden nodes
                v_values = self.df_valid.iloc[[k]].values[0]
                valid_values = self.forward_pass(v_values)
                v_output = valid_values[0]
                valid_error += (v_values[5] - v_output)**2
            # calculate validation error using RMSE
            valid_error = np.sqrt(valid_error / valid_len)
            valid_errors.append(valid_error)

            # check if the validation error has been improved compared to the previous 2 epochs
            if (len(valid_errors) > 1) and (valid_error >= max(valid_errors[-2:])):
                # check if model has improved on the validation subset for more than 10 epochs
                if i - np.argmax(valid_errors) > 10:
                    # end training loop
                    break

        print("Training RMSE: ", train_error)
        print("Validation RMSE: ", valid_error)
        # plot the errors of the training and validation subsets
        plt.plot(range(1, len(train_errors) + 1), train_errors, 'b-')
        plt.plot(range(1, len(valid_errors) + 1), valid_errors, 'r-')
        plt.xlabel("Epoch")
        plt.ylabel("Error Average")
        plt.title("Error Average per Epoch")
        plt.legend(["Training error", "Validation error"])
        plt.show()
        return train_error, valid_error

    def test(self, df_test):
        # initialise variables
        test_len = len(df_test)
        test_error = 0
        test_errors = []
        predicted_outputs = []
        real_values = []

        # for each row in the test subset calculate the errors
        for i in range(test_len):
            t_values = df_test.iloc[[i]].values[0]
            # calculate the predicted predictand
            output = self.forward_pass(t_values)[0]
            predicted_outputs.append(output)
            # calculate test error
            test_error += (t_values[5] - output)**2
            real_values.append(t_values[5])

        # calculate RMSE
        test_error = np.sqrt(test_error / test_len)
        test_errors.append(test_error)

        # get slope and intercept of the predicted outputs and real values
        slope, intercept = stats.linregress(predicted_outputs, real_values)

        # function to calculate the slope and intercept of a point
        def slope_intercept(x):
            return slope * x + intercept

        # apply function to all predicted outputs
        mymodel = list(map(slope_intercept, predicted_outputs))

        # plot regression line and predicted outputs on graph
        plt.plot(predicted_outputs, real_values, 'b.')
        plt.plot(predicted_outputs, mymodel, color='red')
        plt.ylabel("PanE Value")
        plt.xlabel("Predicted Value")
        plt.show()

        print('Testing RMSE: ', test_error)
        # plot the errors of the testing subset
        plt.plot(range(1, len(real_values) + 1), real_values, 'b-')
        plt.plot(range(1, len(predicted_outputs) + 1), predicted_outputs, 'y-')
        plt.xlabel("Test Number")
        plt.ylabel("PanE Value")
        plt.title("Error Average per Test")
        plt.legend(["Expected output", "Predicted output"])
        plt.show()

        return test_error
