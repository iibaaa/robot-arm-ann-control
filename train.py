import argparse
import time

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch
import numpy as np
import os

try:
    from .model import get_ANN_model
except ImportError:
    from model import get_ANN_model


class TrainModel:

    def __init__(self, args):
        self.args = args
        self.train_dataset = None
        self.test_dataset = None

        self.ann_model = None

        self.load_data_set()
        self.load_model()
        self.train()

    def load_model(self):

        self.ann_model = get_ANN_model(activation=self.args.activation)

        if not self.args.checkpoint == "None":
            chk = os.path.join("weights", self.args.checkpoint)
            if os.path.exists(chk):
                self.ann_model.load_state_dict(torch.load(chk))
                print("Loaded model from checkpoint {}".format(chk))
            else:
                print("Checkpoint {} does not exist".format(chk))

        # Set model to device
        try:
            self.ann_model.to(self.args.device)
            print("Model loaded to {}".format(self.args.device))
        except:
            self.ann_model.to(torch.device("cpu"))
            print("Model loaded to CPU")

    def load_data_set(self):
        joint_space_name = self.args.dataset_name + "_joint_space.npy"
        world_space_name = self.args.dataset_name + "_world_space.npy"

        joint_space = np.load(os.path.join(self.args.dataset_path, joint_space_name))
        world_space = np.load(os.path.join(self.args.dataset_path, world_space_name))

        # Split dataset into train and test
        train_joint_space, test_joint_space, train_world_space, test_world_space = train_test_split(joint_space,
                                                                                                    world_space,
                                                                                                    test_size=0.2,
                                                                                                    random_state=self.args.seed)

        # Create train and test dataset
        self.train_dataset = TensorDataset(torch.tensor(train_world_space, dtype=torch.float32),
                                           torch.tensor(train_joint_space, dtype=torch.float32))
        self.test_dataset = TensorDataset(torch.tensor(test_world_space, dtype=torch.float32),
                                          torch.tensor(test_joint_space, dtype=torch.float32))

        print("Loaded dataset from {}".format(os.path.join(self.args.dataset_path, self.args.dataset_name)))

    def train(self):
        criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(self.ann_model.parameters(), lr=self.args.learning_rate)

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)

        test_losses = []
        train_losses = []
        mean_errors = []
        accuracy = []

        print("Training model started")
        start_time = time.time()
        try:
            for epoch in range(self.args.epochs):
                train_running_loss = 0.0
                test_running_loss = 0.0

                # Train Model
                self.ann_model.train()

                for batch_idx, (world_space, joint_space) in enumerate(train_loader):
                    world_space = world_space.to(self.args.device)
                    joint_space = joint_space.to(self.args.device)

                    optimizer.zero_grad()
                    output = self.ann_model(world_space)
                    loss = criterion(output, joint_space)
                    loss.backward()
                    optimizer.step()

                    train_running_loss += loss.item()

                # Test Model
                self.ann_model.eval()
                accuracy_sum = 0.0
                mean_error = 0.0
                total = 0.0

                for test_batch_idx, (world_space, joint_space) in enumerate(test_loader):
                    world_space = world_space.to(self.args.device)
                    joint_space = joint_space.to(self.args.device)

                    with torch.no_grad():
                        output = self.ann_model(world_space)
                        loss = criterion(output, joint_space)

                        # Calculate running loss
                        test_running_loss += loss.item()

                        # Calculate mean error
                        mean_error += torch.mean(torch.abs(output - joint_space)).item()

                        total += world_space.size(0)

                        # Calculate accuracy
                        threshold_ = np.radians(self.args.accuracy_threshold)
                        accuracy_sum += torch.sum(torch.abs(output - joint_space) < threshold_).item()

                accuracy.append(accuracy_sum / total)
                mean_errors.append(mean_error / len(test_loader))
                test_losses.append(test_running_loss / len(test_loader))
                train_losses.append(train_running_loss / len(train_loader))

                print(
                    "Epoch: {} | Train Loss: {:.4f} | Test Loss: {:.4f} | Mean Error: {:.4f} | Accuracy: {:.4f}".format(
                        epoch + 1, train_losses[-1], test_losses[-1], mean_errors[-1], accuracy[-1]))

                if accuracy[-1] > self.args.stop_threshold:
                    print(f"Accuracy threshold reached to {self.args.stop_threshold}, "
                          f"stopping training at epoch {epoch + 1}")
                    break

        finally:
            end_time = time.time()
            print("Training time: {:.2f} seconds".format(end_time - start_time))

            if not os.path.exists("weights"):
                os.makedirs("weights")

            output_name = self.args.output_name + ".pth"

            output = os.path.join("weights", output_name)

            torch.save(self.ann_model.state_dict(), output)

            print(f"Model saved to {output}")

            print("Training complete")

            # Save results
            results = dict()
            results["accuracy"] = accuracy
            results["mean_errors"] = mean_errors
            results["train_losses"] = train_losses
            results["test_losses"] = test_losses
            results["epochs"] = epoch + 1
            results["training_time"] = end_time - start_time
            results["accuracy_threshold"] = self.args.accuracy_threshold
            results["stop_threshold"] = self.args.stop_threshold
            results["seed"] = self.args.seed
            results["batch_size"] = self.args.batch_size
            results["learning_rate"] = self.args.learning_rate
            results["learning_rate"] = self.args.learning_rate
            np.save(f"results_{self.args.activation}_{self.args.output_name}.npy", results)


def main(args):
    TrainModel(args)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--dataset_path", type=str, default="dataset")
    arguments.add_argument("--dataset_name", type=str, default="AL5D_100k")
    arguments.add_argument("--epochs", type=int, default=300)
    arguments.add_argument("--batch_size", type=int, default=20)
    arguments.add_argument("--learning_rate", type=float, default=0.001)
    arguments.add_argument("--seed", type=int, default=42)
    arguments.add_argument("--accuracy_threshold", type=float, default=0.25)
    arguments.add_argument("--checkpoint", type=str, default="None")
    arguments.add_argument("--device", type=str, default="cuda:0")
    arguments.add_argument("--output_name", type=str, default="model_sigmoid_mae")
    arguments.add_argument("--stop_threshold", type=float, default=1.0)
    arguments.add_argument("--activation", type=str, default="sigmoid")

    args = arguments.parse_args()
    main(args)
