import argparse
import os
import torch
import numpy as np
import time


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
        self.ann_model = None
        self.load_model()
        self.load_test_set()
        self.test()

    def load_model(self):
        if self.args.checkpoint == "None":
            print("No checkpoint provided")
            exit()

        self.ann_model = get_ANN_model(activation=self.args.activation)
        weigh_path = os.path.join("weights", self.args.checkpoint)
        if os.path.exists(weigh_path):
            self.ann_model.load_state_dict(torch.load(weigh_path, map_location=torch.device(self.args.device)))
            print("Loaded model from checkpoint {}".format(self.args.checkpoint))
        else:
            print("Checkpoint {} does not exist".format(self.args.checkpoint))
            exit()

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

        s1 = time.time()
        # Predicted Joint Space
        with torch.no_grad():
            predicted_joint_space = self.ann_model(torch.tensor(test_data, dtype=torch.float32).to(self.args.device))

        predicted_joint_space = predicted_joint_space.cpu().numpy()
        s2 = time.time()

        # Calculate Target World Space from predicted joint space
        target_world_space = np.empty((self.args.test_num_samples, 3), dtype=np.float32)
        test_data_for_fkine = []

        for idx, angles in enumerate(predicted_joint_space):
            pose = self.robot_model.fkine(angles)
            target_world_space[idx, :] = np.array([pose.t[0], pose.t[1], pose.t[2]], dtype=np.float32)

        for idx, angles in enumerate(test_output):
            pose = self.robot_model.fkine(angles)
            test_data_for_fkine.append(pose)

        # Inverse Kinematics solution
        inverse_angles = np.empty((self.args.test_num_samples, 4), dtype=np.float32)
        s3 = time.time()

        for idx, target in enumerate(test_data_for_fkine):
            # Set position
            angles = self.robot_model.ikine_LM(target)
            inverse_angles[idx, :] = angles.q
        s4 = time.time()

        target_world_space_inverse = np.empty((self.args.test_num_samples, 3), dtype=np.float32)

        # Calculate Inverse Error
        for idx, angles in enumerate(inverse_angles):
            pose = self.robot_model.fkine(angles)
            target_world_space_inverse[idx, :] = np.array([pose.t[0], pose.t[1], pose.t[2]], dtype=np.float32)

        inverse_error = np.abs(target_world_space_inverse - target_world_space)


        # Calculate Error
        error = np.abs(target_world_space - test_data)

        # Print Results
        print("Model Test Results")
        print("Run Time : {} ms".format((s2-s1)*1000))
        print("Mean Error: {}".format(np.mean(error) * 1000))

        print("X Error: {}".format(np.mean(error[:, 0]) * 1000))
        print("Y Error: {}".format(np.mean(error[:, 1]) * 1000))
        print("Z Error: {}".format(np.mean(error[:, 2]) * 1000))

        print("Inverse Test Results")
        print("Run Time : {} s".format((s4-s3)))
        print("Mean Error: {}".format(np.mean(inverse_error) * 1000))

        print("X Error: {}".format(np.mean(inverse_error[:, 0]) * 1000))
        print("Y Error: {}".format(np.mean(inverse_error[:, 1]) * 1000))
        print("Z Error: {}".format(np.mean(inverse_error[:, 2]) * 1000))

        # Save results
        results = dict()
        results["test_time"] = s2-s1
        results["test_dataset"] = self.args.dataset_name
        results["seed"] = self.args.seed
        results["test_sample"] = self.args.test_num_samples
        results["error"] = error
        results["mean_error"] = np.mean(error) * 1000
        results["x_error"] = np.mean(error[:, 0]) * 1000
        results["y_error"] = np.mean(error[:, 1]) * 1000
        results["z_error"] = np.mean(error[:, 2]) * 1000
        results["ik_time"] = s4-s3
        results["ik_error"] = inverse_error
        results["ik_mean_error"] = np.mean(inverse_error) * 1000
        results["ik_x_error"] = np.mean(inverse_error[:, 0]) * 1000
        results["ik_y_error"] = np.mean(inverse_error[:, 1]) * 1000
        results["ik_z_error"] = np.mean(inverse_error[:, 2]) * 1000

        model_name = self.args.checkpoint.split(".")[0]
        # np.save(f"results/test_results_{model_name}.npy", results)






def main(args):
    TestModel(args)


if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument( "--dataset_path", type=str, default="dataset")
    arguments.add_argument("--dataset_name", type=str, default="AL5D_100k")
    arguments.add_argument("--seed", type=int, default=0)
    arguments.add_argument("--test_num_samples", type=int, default=1000)
    arguments.add_argument("--checkpoint", type=str, default="model_sigmoid.pth")
    arguments.add_argument("--device", type=str, default="cpu")
    arguments.add_argument("--activation", type=str, default="sigmoid")

    args = arguments.parse_args()
    main(args)
