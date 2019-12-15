from cython.parallel import parallel, prange
import numpy as np
cimport numpy as np


cdef extern from "ur_kin.h" namespace "ur_kinematics":
    void forward(const double*, double*, int) nogil
    int inverse(const double*, double*, double, int) nogil


cpdef void ur_forward(int ur_type, double[::1] joints, double[::1] pose) nogil:
    """
    Given a single set of joint parameters size 1x6, compute the robot pose and
    store result by overwriting the flattened 4x4 `pose` matrix.
    """
    forward(&joints[0], &pose[0], ur_type)


cpdef ur_inverse(ur_type, T, q_sols, q6_des):
    """
    Given a single flattened 4x4 pose matrix `T` compute the inverse kinematics
    solutions giving robot joint parameters. Each solution is 1x6, there are
    max of 8 solutions, and the solutions are stored in q_sols, which is
    therefore of size 8x6.

    Returns the actual number of solutions.
    """
    cdef double[::1] pose = T
    cdef double[:, ::1] solutions = q_sols

    solution_cnt = inverse(&pose[0], &solutions[0, 0], 0, ur_type)
    q_sols[:, :] = solutions

    return solution_cnt


cpdef ur_forward_n(int ur_type, double[:, ::1] joints, double[:, ::1] poses):
    """
    Computes forward poses for N joint positions.
    """
    cdef int i
    cdef int n = joints.shape[0]

    for i in prange(n, nogil=True):
        ur_forward(ur_type, joints[i, :], poses[i, :])
