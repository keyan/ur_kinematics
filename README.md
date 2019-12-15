## Usage

Install python dependencies (ideally in a fresh virtualenv):
```
pip install -r requirements.txt
```

See available make commands:
```
make
```

Bindings are written in Cython in `py_ur_kin.pyx`, then can be imported and used with:
```
import py_ur_kin

py_ur_kin.ur_forward(...)
py_ur_kin.ur_inverse(...)
```

## Design decisions

1. Used Cython (vs. pybind or boost) because Cython's documentation looked easier to parse and was more friendly to Python programmers. Additionally the OpenMP parallization support appeared simple to use.
1. To support all UR models I instantiated a new struct for the model parameters and fetched the correct one on each function call then used that to create function local params. This is a little messy and I would have preferred passing around the struct, but I didn't want to modify more of the kinematic calculation code than I had too.

## Notes

#### What is inverse kinematics?

Given the coordinates of an end effector, find the inputs for the rotators/arms/joints

e.g. we want the robot to cut along some line on an object:
	- "motion planning" allows us to specify the movements of the end-effectors to achieve this
	- "inverse kinemetatics" allows transforming the motion plan into joint parameters needed

#### What is forward kinematics?

Using kinematic equations to determine the behavior/movement of end-effectors given joint params

#### How to create bindings?

Pybind, cython, boost/python

#### Parallel execution?

Maybe use multiprocessing and call the cpp-bindings? Or using Cython's parallel module? https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html#cython.parallel.parallel
