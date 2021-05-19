# FastSLAM64 - FastSLAM 1.0 GPU implementation
This repository contains a real-time CUDA implementation of the FastSLAM 1.0 [[1]](#1) algorithm. The implementation is described in my thesis *_Navigation System for Autonomous Student Formula_* (Czech Technical University in Prague, 2021).


## Dependencies
The algorithm require a CUDA-capable GPU with CUDA installed on the host. Python3 is required to run the examples. The Python dependencies can be installed using:

```
$ python3 -m pip install -r requirements.txt
```
## Running
There are three examples from different datasets -- `simulation.py`, `fsonline.py`, and `utias.py`, which can run by executing the corresponding file with Python.

## Code structure



- `simulation.py`
- `fsonline.py`
- `utias.py`
- `lib/`
    - `Python GPU instrumentation/visualization`
- `cuda/`
    - `FastSLAM 1.0 implementation`

## References
<a id="1">[1]</a>
M. Montemerlo, S. Thrun, D. Koller, B. Wegbreit, et al.
FastSLAM: Afactored solution to the simultaneous localization and mapping problem.
Aaai/iaai, 593598, 2002

