## Usage

Install python dependencies (ideally in a fresh virtualenv):
```
pip install -r requirements.txt
```

See available make commands:
```
make
```

## Notes

- What is inverse kinematics?

Given the coordinates of an end effector, find the inputs for the rotators/arms/joints

e.g. we want the robot to cut along some line on an object:
	- "motion planning" allows us to specify the movements of the end-effectors to achieve this
	- "inverse kinemetatics" allows transforming the motion plan into joint parameters needed

- What is forward kinematics?

Using kinematic equations to determine the behavior/movement of end-effectors given joint params

- How to create bindings?

Pybind, cython, boost/python

- Parallel execution?

Maybe use multiprocessing and call the cpp-bindings? Or using Cython's parallel module? https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html#cython.parallel.parallel
