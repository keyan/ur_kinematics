## Usage

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

## Dependencies

Python dependencies (ideally in a fresh virtualenv):
```
pip install -r requirements.txt
```

Actually building the bindings was tricky due to OpenMP support. Ultimately I was able to build on MacOSX-10 with `ggc-9` and the below OpenMP runtime. But I suspect reproducing on another machine may be difficult:
```
brew install gcc-9
brew install libomp
```

## Design decisions

1. Used Cython (vs. pybind or boost) because Cython's documentation looked easier to parse and was more friendly to Python programmers. Additionally the OpenMP parallization support appeared simple to use, although OpenMP was ultimately a bit of a pain.
1. To support all UR models I instantiated a new struct for the model parameters and fetched the correct one on each function call then used that to create function local params. This is a little messy and I would have preferred passing around the struct, but I didn't want to modify more of the kinematic calculation code than I had too.
1. Parallelization with Cython/OpenMP didn't produce great results. A GIL-enabled Python loop shows similar performance for forward kinematics calculations, while there is a more noticeable speedup for inverse calculations. Trying to increase the batch size enough to see an obvious benefit during forward calculations only led to machine memory exhaustion. The benchmark test occasionally fails because there is not a consistent speedup with increased threads used.

Without parallelization:
```
test_kin.py Starting forward benchmark...
Thread count: 1, Speedup from last run: ~100.0%, Time: 0.3836228847503662
Thread count: 2, Speedup from last run: ~19.62%, Time: 0.3083679676055908
Thread count: 3, Speedup from last run: ~11.89%, Time: 0.271716833114624
Thread count: 4, Speedup from last run: ~13.36%, Time: 0.23540711402893066
Starting inverse benchmark...
Thread count: 1, Speedup from last run: ~100.0%, Time: 2.4680662155151367
Thread count: 2, Speedup from last run: ~0.11%, Time: 2.4654102325439453
Thread count: 3, Speedup from last run: ~0.78%, Time: 2.4461629390716553
Thread count: 4, Speedup from last run: ~0.07%, Time: 2.4443681240081787
```

With parallelization:
```
test_kin.py Starting forward benchmark...
Thread count: 1, Speedup from last run: ~100.0%, Time: 0.3841521739959717
Thread count: 2, Speedup from last run: ~21.33%, Time: 0.3022170066833496
Thread count: 3, Speedup from last run: ~9.31%, Time: 0.2740669250488281
Thread count: 4, Speedup from last run: ~10.33%, Time: 0.24576401710510254
Starting inverse benchmark...
Thread count: 1, Speedup from last run: ~100.0%, Time: 2.645191192626953
Thread count: 2, Speedup from last run: ~43.0%, Time: 1.507807970046997
Thread count: 3, Speedup from last run: ~16.96%, Time: 1.252027988433838
Thread count: 4, Speedup from last run: ~16.61%, Time: 1.0440778732299805
```
