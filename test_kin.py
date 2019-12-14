import numpy as np

from py_ur_kin import *


# Universal Robot model constants
UR3 = 0
UR5 = 1
UR10 = 2

class TestForwardSinglePose:
    # def test_ur3(self):
    #     expected_pose = np.array([
    #         [0., 1., -0., 0.4569],
    #         [0., 0., 0.19425],
    #         [0., 0., 0., 1.],
    #     ])

    #     pose_solution = np.zeros((4,4))
    #     ur_forward(ur_type=UR3, joints=np.zeros((6,1)), pose=pose_solution)

    #     assert np.array_equal(pose_solution, expected_pose)

    # def test_ur5(self):
    #     expected_pose = np.array([
    #         [0., 1., -0., 0.81725],
    #         [1., 0., 0., 0.19145],
    #         [0., 0., -1., -0.005491],
    #         [0., 0., 0., 1.],
    #     ])

    #     pose_solution = np.zeros((4,4))
    #     ur_forward(ur_type=UR5, joints=np.zeros((6,1)), pose=pose_solution)

    #     assert np.array_equal(pose_solution, expected_pose)

    def test_ur10(self):
        expected_pose = np.array([
            0., 1., -0., 1.1843,
            1., 0., 0., 0.256141,
            0., 0., -1., 0.0116,
            0., 0., 0., 1.,
        ])

        pose_solution = np.zeros((16,), dtype=np.float)
        ur_forward(ur_type=UR10, joints=np.zeros((6,)), pose=pose_solution)

        assert np.array_equal(pose_solution, expected_pose)


class TestInverseSinglePose:
    # def test_ur3(self):
    #     expected_pose = np.array([
    #         [0., 1., -0., 0.4569],
    #         [0., 0., 0.19425],
    #         [0., 0., 0., 1.],
    #     ])

    #     pose_solution = np.zeros((4,4))
    #     ur_forward(ur_type=UR3, joint=np.zeros((6,1)), pose=pose_solution)

    #     assert pose_solution == expected_pose

    # def test_ur5(self):
    #     expected_pose = np.array([
    #         [0., 1., -0., 0.81725],
    #         [1., 0., 0., 0.19145],
    #         [0., 0., -1., -0.005491],
    #         [0., 0., 0., 1.],
    #     ])

    #     pose_solution = np.zeros((4,4))
    #     ur_forward(ur_type=UR5, joint=np.zeros((6,1)), pose=pose_solution)

    #     assert pose_solution == expected_pose

    def test_ur10(self):
        # Input pose and expected solutions from compiling/executing ur_kin.cpp directly.
        pose = np.array([
            0.455, 0.292, -0.841, 0.866,
            0.540, -0.841, 0.000, 0.214,
            -0.708, -0.455 -0.540, -0.482,
            0.000, 0.000, 0.000, 1.000,
        ])
        expected_solutions = np.array([
            [0.000329, 5.637287, 1.359465, 0.411913, 1.000537, 0.000329],
            [0.000329, 0.659394, 4.923720, 1.825551, 1.000537, 0.000329],
            [0.000329, 5.585598, 1.801145, 3.163515, 5.282648, 3.141921],
            [0.000329, 1.019023, 4.482041, 5.049194, 5.282648, 3.141921],
            [3.534660, 2.094854, 1.763999, 4.619685, 1.901325, 3.489326],
            [3.534660, 3.777466, 4.519187, 0.181885, 1.901325, 3.489326],
            [3.534660, 2.508555, 1.386846, 1.441544, 4.381860, 0.347733],
            [3.534660, 3.839695, 4.896339, 2.884096, 4.381860, 0.347733],
        ])
        ik_solutions = np.zeros((8, 6))

        solution_cnt = ur_inverse(ur_type=UR10, T=pose, q_sols=ik_solutions, q6_des=0)

        assert solution_cnt == 8
        assert np.allclose(ik_solutions, expected_solutions, atol=1e-6)
