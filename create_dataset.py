import os
import numpy as np
import roboticstoolbox as rtb
from math import degrees, radians
import argparse

try:
    from model import get_robot_model
except ImportError:
    from .model import get_robot_model


def visualize(world_space):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(world_space[:, 0], world_space[:, 1], world_space[:, 2],
               c='r', marker='o',
               label='World Space',
               s=0.5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


    # Plot XY - XZ - YZ planes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(world_space[:, 0], world_space[:, 1],
               c='k', marker='o',
               label='World Space',
               s=0.5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(world_space[:, 0], world_space[:, 2],
               c='k', marker='o',
               label='World Space',
               s=0.5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(world_space[:, 1], world_space[:, 2],
               c='k', marker='o',
               label='World Space',
               s=0.5)

    ax.set_xlabel('Y Label')
    ax.set_ylabel('Z Label')

    plt.show()


def save(joint_space, world_space, args):
    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main_name = f"{args.dataset_name}_{int(args.num_samples/1000)}k"
    # Joint Space
    output_file_joint_space = os.path.join(args.output_path, main_name + "_joint_space.npy")
    np.save(output_file_joint_space, joint_space)
    print("Saved joint space to {}".format(output_file_joint_space))

    # World Space
    output_file_world_space = os.path.join(args.output_path, main_name + "_world_space.npy")
    np.save(output_file_world_space, world_space)
    print("Saved world space to {}".format(output_file_world_space))


def main(args):
    robot = get_robot_model()
    dof = 4
    resolution = radians(args.resolution)

    # Create Seperate joint Space
    joint_space_seperate = dict()

    for i in range(dof):
        q_min = robot.qlim[0][i]
        q_max = robot.qlim[1][i]
        joint_space_seperate[i] = np.arange(q_min, q_max, resolution, dtype=np.float32)
        print("Joint {} space size: {}".format(i, len(joint_space_seperate[i])))

    # Create Joint Space
    joint_space = np.empty((args.num_samples, dof), dtype=np.float32)
    print("Joint space size: {}".format(joint_space.shape))

    np.random.seed(args.num_samples)
    for j in range(dof):
        joint_space[:, j] = np.random.choice(joint_space_seperate[j], size=args.num_samples)

    # Create World Space for xyz
    world_space = np.empty((args.num_samples, 3), dtype=np.float32)

    for idx, angles in enumerate(joint_space):
        pose = robot.fkine(angles)
        world_space[idx, :] = np.array([pose.t[0], pose.t[1], pose.t[2]], dtype=np.float32)

    print("World space size: {}".format(world_space.shape))

    if args.save:
        save(joint_space, world_space, args)

    if args.visualize:
        visualize(world_space)




if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--output_path", type=str, default="dataset")
    arguments.add_argument("--dataset_name", type=str, default="AL5D")
    arguments.add_argument("--num_samples", type=int, default=50000)
    arguments.add_argument("--resolution", type=float, default=0.1)
    arguments.add_argument("--visualize", type=bool, default=False)
    arguments.add_argument("--save", type=bool, default=True)

    args = arguments.parse_args()
    main(args)
