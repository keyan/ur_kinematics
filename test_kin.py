import numpy as np

from py_ur_kin import *


# Universal Robot model constants
UR3 = 0
UR5 = 1
UR10 = 2

class TestForwardSinglePose:
    def test_ur3(self):
        expected_pose = np.array([
            0., 1., -0., 0.4569,
            1., 0., 0., 0.19425,
            0., 0., -1., 0.06655,
            0., 0., 0., 1.,
        ])
        assert expected_pose.shape == (16,)

        pose_solution = np.zeros((16,), dtype=np.float)
        ur_forward(ur_type=UR3, joints=np.zeros((6,)), pose=pose_solution)

        assert np.allclose(pose_solution, expected_pose, atol=1e-16)

    def test_ur5(self):
        expected_pose = np.array([
            0., 1., -0., 0.81725,
            1., 0., 0., 0.19145,
            0., 0., -1., -0.005491,
            0., 0., 0., 1.,
        ])
        assert expected_pose.shape == (16,)

        pose_solution = np.zeros((16,), dtype=np.float)
        ur_forward(ur_type=UR5, joints=np.zeros((6,)), pose=pose_solution)

        assert np.allclose(pose_solution, expected_pose, atol=1e-16)

    def test_ur10(self):
        expected_pose = np.array([
            0., 1., -0., 1.1843,
            1., 0., 0., 0.256141,
            0., 0., -1., 0.0116,
            0., 0., 0., 1.,
        ])
        assert expected_pose.shape == (16,)

        pose_solution = np.zeros((16,), dtype=np.float)
        ur_forward(ur_type=UR10, joints=np.zeros((6,)), pose=pose_solution)

        assert np.allclose(pose_solution, expected_pose, atol=1e-16)


class TestForwardNPoses:
    def test_ur3_identical_joints_input(self):
        expected_poses = np.array([
            [
                0., 1., -0., 0.4569,
                1., 0., 0., 0.19425,
                0., 0., -1., 0.06655,
                0., 0., 0., 1.,
            ],
            [
                0., 1., -0., 0.4569,
                1., 0., 0., 0.19425,
                0., 0., -1., 0.06655,
                0., 0., 0., 1.,
            ],
        ])
        assert expected_poses.shape == (2, 16)

        pose_solutions = np.zeros((2, 16), dtype=np.float)
        ur_forward_n(ur_type=UR3, joints=np.zeros((2, 6)), poses=pose_solutions)

        assert np.allclose(pose_solutions, expected_poses, atol=1e-16)

    def test_ur3_diff_joints_input(self):
        expected_poses = np.array([
            [
                0., 1., -0., 0.4569,
                1., 0., 0., 0.19425,
                0., 0., -1., 0.06655,
                0., 0., 0., 1.,
            ],
            [
                -0.30596594, 0.1476573, -0.94052228, 0.0882236,
                0.66201934, -0.67696042, -0.32164418, 0.22006864,
                -0.68418947, -0.72105611, 0.10937484, -0.25187121,
                0., 0., 0., 1.,
            ],
        ])
        assert expected_poses.shape == (2, 16)

        joints = np.array([
            [0, 0, 0, 0, 0, 0],
            [0.37789703, 0.64992169, 0.64684366, 0.3325949,  0.75510405, 0.07019782],
        ])
        assert joints.shape == (2, 6)

        pose_solutions = np.zeros((2, 16), dtype=np.float)
        ur_forward_n(ur_type=UR3, joints=joints, poses=pose_solutions)

        assert np.allclose(pose_solutions, expected_poses, atol=1e-16)


    def test_linear_speedup(self):
        pass


class TestInverseSinglePose:
    """
    Input pose and expected solutions from compiling/executing ur_kin.cpp directly.

    Absolute float comparison discrepancy tolerance is fairly high because there is
    rounding when printing values in cpp.
    """
    def test_ur3(self):
        pose = np.array([
            0.455, 0.292, -0.841, 0.866,
            0.540, -0.841, 0.000, 0.214,
            -0.708, -0.455, -0.540, -0.482,
            0.000, 0.000, 0.000, 1.000,
        ])
        assert pose.shape == (16,)

        ik_solutions = np.zeros((8, 6))

        solution_cnt = ur_inverse(ur_type=UR3, T=pose, q_sols=ik_solutions, q6_des=0)

        assert solution_cnt == 0

    def test_ur5(self):
        pose = np.array([
            0.455, 0.292, -0.841, 0.866,
            0.540, -0.841, 0.000, 0.214,
            -0.708, -0.455, -0.540, -0.482,
            0.000, 0.000, 0.000, 1.000,
        ])
        assert pose.shape == (16,)

        ik_solutions = np.zeros((8, 6))

        solution_cnt = ur_inverse(ur_type=UR5, T=pose, q_sols=ik_solutions, q6_des=0)

        assert solution_cnt == 0

    def test_ur10(self):
        pose = np.array([
            0.455, 0.292, -0.841, 0.866,
            0.540, -0.841, 0.000, 0.214,
            -0.708, -0.455, -0.540, -0.482,
            0.000, 0.000, 0.000, 1.000,
        ])
        assert pose.shape == (16,)

        expected_solutions = np.array([
            [0.000329, 0.000113, 0.999512, 6.283170, 1.000537, 0.000329],
            [0.000329, 0.963024, 5.283673, 1.036097, 1.000537, 0.000329],
            [0.000329, 0.075583, 1.290113, 2.775506, 5.282648, 3.141921],
            [0.000329, 1.315261, 4.993072, 4.116054, 5.282648, 3.141921],
            [3.534660, 1.845668, 1.240550, 5.492562, 1.901325, 3.489326],
            [3.534660, 3.038336, 5.042635, 0.497808, 1.901325, 3.489326],
            [3.534660, 2.159603, 1.054738, 2.222846, 4.381860, 0.347733],
            [3.534660, 3.175300, 5.228447, 3.316625, 4.381860, 0.347733],
        ])
        ik_solutions = np.zeros((8, 6))

        solution_cnt = ur_inverse(ur_type=UR10, T=pose, q_sols=ik_solutions, q6_des=0)

        assert solution_cnt == 8
        assert np.allclose(ik_solutions, expected_solutions, atol=1e-6)

class TestInverseNPoses:
    def test_ur10_identical_joints_input(self):
        poses = np.array([
            [
                0.455, 0.292, -0.841, 0.866,
                0.540, -0.841, 0.000, 0.214,
                -0.708, -0.455, -0.540, -0.482,
                0.000, 0.000, 0.000, 1.000,
            ],
            [
                0.455, 0.292, -0.841, 0.866,
                0.540, -0.841, 0.000, 0.214,
                -0.708, -0.455, -0.540, -0.482,
                0.000, 0.000, 0.000, 1.000,
            ],
        ])
        assert poses.shape == (2, 16)

        expected_solutions = np.array([
            [
                [0.000329, 0.000113, 0.999512, 6.283170, 1.000537, 0.000329],
                [0.000329, 0.963024, 5.283673, 1.036097, 1.000537, 0.000329],
                [0.000329, 0.075583, 1.290113, 2.775506, 5.282648, 3.141921],
                [0.000329, 1.315261, 4.993072, 4.116054, 5.282648, 3.141921],
                [3.534660, 1.845668, 1.240550, 5.492562, 1.901325, 3.489326],
                [3.534660, 3.038336, 5.042635, 0.497808, 1.901325, 3.489326],
                [3.534660, 2.159603, 1.054738, 2.222846, 4.381860, 0.347733],
                [3.534660, 3.175300, 5.228447, 3.316625, 4.381860, 0.347733],
            ],
            [
                [0.000329, 0.000113, 0.999512, 6.283170, 1.000537, 0.000329],
                [0.000329, 0.963024, 5.283673, 1.036097, 1.000537, 0.000329],
                [0.000329, 0.075583, 1.290113, 2.775506, 5.282648, 3.141921],
                [0.000329, 1.315261, 4.993072, 4.116054, 5.282648, 3.141921],
                [3.534660, 1.845668, 1.240550, 5.492562, 1.901325, 3.489326],
                [3.534660, 3.038336, 5.042635, 0.497808, 1.901325, 3.489326],
                [3.534660, 2.159603, 1.054738, 2.222846, 4.381860, 0.347733],
                [3.534660, 3.175300, 5.228447, 3.316625, 4.381860, 0.347733],
            ],
        ])
        ik_solutions = np.zeros((2, 8, 6))
        n_sols = np.zeros((2,), dtype=np.int)

        ur_inverse_n(ur_type=UR10, T=poses, q_sols=ik_solutions, n_sols=n_sols, q6_des=0)

        assert np.allclose(ik_solutions, expected_solutions, atol=1e-6)
        assert np.array_equal([8, 8], n_sols)

    def test_ur3_diff_joints_input(self):
        poses = np.array([
            [
                0.455, 0.292, -0.841, 0.866,
                0.540, -0.841, 0.000, 0.214,
                -0.708, -0.455, -0.540, -0.482,
                0.000, 0.000, 0.000, 1.000,
            ],
            [
                0.09132201, 0.45659133, 0.81887916, 0.4408727,
                0.35095868, 0.71505819, 0.13249488, 0.9656959,
                0.17816832, 0.55892049, 0.4209948,  0.70959087,
                0.39323732, 0.53455839, 0.91823681, 0.27298904,
            ],
        ])
        expected_solutions = np.array([
            [
                [0.000329, 0.000113, 0.999512, 6.283170, 1.000537, 0.000329],
                [0.000329, 0.963024, 5.283673, 1.036097, 1.000537, 0.000329],
                [0.000329, 0.075583, 1.290113, 2.775506, 5.282648, 3.141921],
                [0.000329, 1.315261, 4.993072, 4.116054, 5.282648, 3.141921],
                [3.534660, 1.845668, 1.240550, 5.492562, 1.901325, 3.489326],
                [3.534660, 3.038336, 5.042635, 0.497808, 1.901325, 3.489326],
                [3.534660, 2.159603, 1.054738, 2.222846, 4.381860, 0.347733],
                [3.534660, 3.175300, 5.228447, 3.316625, 4.381860, 0.347733],
            ],
            [
                [0.97684781, 5.30698571, 0.98927075, 2.620268  , 4.83341192, 1.53478269,],
                [0.97684781, 6.26009984, 5.29391456, 3.64569537, 4.83341192, 1.53478269,],
                [4.43855327, 3.22538603, 0.87469635, 5.78582899, 1.5777854 , 1.88699045,],
                [4.43855327, 4.06873894, 5.40848896, 0.40868347, 1.5777854 , 1.88699045,],
                [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,],
                [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,],
                [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,],
                [0.        , 0.        , 0.        , 0.        , 0.        , 0.        ,],
            ],
        ])
        ik_solutions = np.zeros((2, 8, 6))
        n_sols = np.zeros((2,), dtype=np.int)

        ur_inverse_n(ur_type=UR10, T=poses, q_sols=ik_solutions, n_sols=n_sols, q6_des=0)

        assert np.allclose(ik_solutions, expected_solutions, atol=1e-6)
        assert np.array_equal([8, 4], n_sols)
