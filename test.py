import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


try:
    from .model import get_ANN_model, get_robot_model
except ImportError:
    from model import get_ANN_model, get_robot_model

class TestModel:
    def __init__(self, args):
        self.test_dataset = None
        self.joint_dataset = None
        self.args = args

        self.robot_model = get_robot_model()
        self.ann_model = get_ANN_model()
        self.load_model()
        self.load_test_set()
        self.test()

    def load_model(self):
        if self.args.checkpoint == "None":
            print("No checkpoint provided")
            return

        weigh_path = os.path.join("weights", self.args.checkpoint)
        if os.path.exists(weigh_path):
            self.ann_model.load_state_dict(torch.load(weigh_path))
            print("Loaded model from checkpoint {}".format(self.args.checkpoint))

    def load_test_set(self):
        world_space_name = self.args.dataset_name + "_world_space.npy"
        joint_space_name = self.args.dataset_name + "_joint_space.npy"

        self.test_dataset = np.load(os.path.join(self.args.dataset_path, world_space_name))
        self.joint_dataset = np.load(os.path.join(self.args.dataset_path, joint_space_name))

        print("Loaded dataset from {}".format(os.path.join(self.args.dataset_path, self.args.dataset_name)))


    def test(self):
        # Set model to device
        try:
            self.ann_model.to(self.args.device)
            print("Model loaded to {}".format(self.args.device))
        except:
            self.ann_model.to(torch.device("cpu"))
            print("Model loaded to CPU")

        # Select Random Test Samples
        np.random.seed(self.args.seed)
        test_samples_idx = np.random.randint(0, len(self.test_dataset), self.args.test_num_samples)

        # Test Data
        test_data = self.test_dataset[test_samples_idx]
        test_output = self.joint_dataset[test_samples_idx]

        # Predicted Joint Space
        with torch.no_grad():
            predicted_joint_space = self.ann_model(torch.tensor(test_data, dtype=torch.float32).to(self.args.device))

        predicted_joint_space = predicted_joint_space.cpu().numpy()

        # Calculate Target World Space from predicted joint space
        target_world_space = np.empty((self.args.test_num_samples, 3), dtype=np.float32)

        for idx, angles in enumerate(predicted_joint_space):
            pose = self.robot_model.fkine(angles)
            target_world_space[idx, :] = np.array([pose.t[0], pose.t[1], pose.t[2]], dtype=np.float32)

        # Calculate Error
        error = np.abs(target_world_space - test_data)

        # Print Results
        print("Test Results")
        print("Mean Error: {}".format(np.mean(error) * 1000))

        print("X Error: {}".format(np.mean(error[:, 0]) * 1000))
        print("Y Error: {}".format(np.mean(error[:, 1]) * 1000))
        print("Z Error: {}".format(np.mean(error[:, 2]) * 1000))


        # Calculate Joint Error
        joint_error = np.abs(predicted_joint_space - test_output)

        # Print Results in degrees
        print("Joint Error")
        print("Joint 1 Error: {}".format(np.mean(joint_error[:, 0]) * 180 / np.pi))
        print("Joint 2 Error: {}".format(np.mean(joint_error[:, 1]) * 180 / np.pi))
        print("Joint 3 Error: {}".format(np.mean(joint_error[:, 2]) * 180 / np.pi))
        print("Joint 4 Error: {}".format(np.mean(joint_error[:, 3]) * 180 / np.pi))




def main(args):
    TestModel(args)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--dataset_path", type=str, default="dataset")
    arguments.add_argument("--dataset_name", type=str, default="AL5D_100k")
    arguments.add_argument("--seed", type=int, default=0)
    arguments.add_argument("--test_num_samples", type=int, default=100)
    arguments.add_argument("--checkpoint", type=str, default="model_relu.pt")
    arguments.add_argument("--device", type=str, default="cuda:0")

    args = arguments.parse_args()
    main(args)
