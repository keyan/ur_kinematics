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
            0.000000, 0.000000, 1.000000, 0.000000, 1.000000, 0.000000,
            0.000000, 0.963378, 5.283185, 1.036622, 1.000000, 0.000000,
            0.000000, 0.075612, 1.290638, 2.775343, 5.283185, 3.141593,
            0.000000, 1.315787, 4.992547, 4.116444, 5.283185, 3.141593,
            3.534445, 1.845147, 1.241084, 5.492384, 1.901905, 3.489189,
            3.534445, 3.038321, 5.042102, 0.498191, 1.901905, 3.489189,
            3.534445, 2.159248, 1.055217, 2.222556, 4.381280, 0.347596,
            3.534445, 3.175403, 5.227968, 3.316835, 4.381280, 0.347596,
        ])
        ik_solutions = np.zeros((8, 6))

        solution_cnt = ur_inverse(ur_type=UR10, T=pose, q_sols=ik_solutions, q6_des=0)

        assert solution_cnt == 8
        assert np.array_equal(ik_solutions, expected_solutions)
