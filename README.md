# Code for the article - Temporal Network Compression via Network Hashing

Article is available at: https://arxiv.org/abs/2307.04890. The follwing code has also been archived at: https://hal.inria.fr/ .

## Description

The repository provides different functions to compute the component matrix, the size of the out-components of the nodes and their hashed versions.
The code provided can be used to compute:
1. The component matrix
- with a full numpy array
- with a sparse scipy matrix
- with HyperLogLog structures

1. The out-components sizes
- with the full numpy array
- with a sparse scipy matrix
- from the HyperLogLog structures

1. The hashed component matrices
1. The aggregation of the hashed matrices to approximate the true component matrix

## Requirements

The code is written in Python3, and uses the following libraries: NumPy, SciPy, Networkx, Datasketch and POT.

Run `pip install -r requirements.txt` to install all the dependencies.

## Getting started

The file 'main.py' includes the code to compute all the aforementioned functions. You can use it with
```bash
$ python3 main.py
```