import torch.nn as nn
import roboticstoolbox as rtb

INPUT_SIZE = 3
OUTPUT_SIZE = 4
NUM_HIDDEN_LAYERS = 3
HIDDEN_SIZE = 25
ACTIVATION = "sigmoid"

ACTIVATIONS = {"sigmoid": nn.Sigmoid(),
               "relu": nn.ReLU(),
               "tanh": nn.Tanh()}


class RobotANNModel(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, hidden_size, activation):
        super(RobotANNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation

        self.layers = []
        # Create input layer
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))

        # Create hidden layers
        for i in range(self.num_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(self.activation)

        # Create output layer
        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

        # Create sequential model
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


def get_ANN_model(input_size=INPUT_SIZE,
                  output_size=OUTPUT_SIZE,
                  num_hidden_layers=NUM_HIDDEN_LAYERS,
                  hidden_size=HIDDEN_SIZE,
                  activation=ACTIVATION):
    activation = ACTIVATIONS[activation]
    model = RobotANNModel(input_size, output_size, num_hidden_layers, hidden_size, activation)
    return model


def get_robot_model():
    return rtb.models.AL5D()
