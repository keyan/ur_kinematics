#ifndef UR_KIN_H
#define UR_KIN_H

namespace ur_kinematics {
  void forward(const double* q, double* T);
  int inverse(const double* T, double* q_sols, double q6_des);
}

#endif
