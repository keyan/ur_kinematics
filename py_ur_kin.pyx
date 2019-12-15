import numpy as np
cimport numpy as np


cdef extern from "ur_kin.h" namespace "ur_kinematics":
    void forward(const double*, double*, int)
    int inverse(const double*, double*, double, int)


cpdef ur_forward(ur_type, joints, pose):
    """
    Given a single set of joint parameters size 1x6, compute the robot pose and
    store result by overwriting the flattened 4x4 `pose` matrix.
    """
    cdef:
        np.ndarray[double, ndim=1, mode='c'] q = np.ascontiguousarray(
            joints, dtype=np.float)
        np.ndarray[double, ndim=1, mode='c'] T = np.ascontiguousarray(
            np.zeros((16,)), dtype=np.float)

    forward(&q[0], &T[0], ur_type)

    pose[:] = T


cpdef ur_inverse(ur_type, T, q_sols, q6_des):
    """
    Given a single flattened 4x4 pose matrix `T` compute the inverse kinematics
    solutions giving robot joint parameters. Each solution is 1x6, there are
    max of 8 solutions, and the solutions are stored in q_sols, which is
    therefore of size 8x6.

    Returns the actual number of solutions.
    """
    cdef:
        np.ndarray[double, ndim=1, mode='c'] pose = np.ascontiguousarray(
            T, dtype=np.float)
        np.ndarray[double, ndim=2, mode='c'] solutions = np.ascontiguousarray(
            np.zeros((8, 6)), dtype=np.float)

    solution_cnt = inverse(&pose[0], &solutions[0, 0], 0, ur_type)
    q_sols[:, :] = solutions

    return solution_cnt
