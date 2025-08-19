## Overview

This repository contains the code and results from testing several effective potentials to describe the low-energy behavior of a given short-range potential. A brief review of the theory, details of the chosen potentials, references, and a summary of the main results are provided in **`summary_project.pdf`**.

## Repository structure

Eigenfunctions, eigenvalues and phase shifts computed in **`code.py`** are saved in:

- **`Coulomb/`** — for a Coulomb potential; 
- **`true/`** — for the chosen short-range potential;  
- **`eff_a_4/`** — for an effective field theory up to order $a^4$, with subfolders for different values of $a$;
- **`eff_a_2/`** — for an effective field theory up to order $a^2$, with subfolders for different values of $a$.

The **`plot`** folder contains plots comparing results across the different theories.

> Example output from a run is shown in **`terminal.txt`**.

## Running the code

To run the script and solve the radial Schrödinger equation with, for instance, the parameters below, use the following command in your terminal:

```bash
$ python code.py --r_max 1200 --x_0 -9.0 --dx 0.01 --Z 1 --n_max 20 --r_ph 800 --b 1
```
where the parameters represent:
```console
$ python code.py -h
  usage: code.py [-h] [--r_max R_MAX] [--x_0 X_0] [--dx DX] [--Z Z] [--n_max N_MAX] [--r_ph R_PH] [--b B]

Solve the radial Schrödinger equation.

options:
  -h, --help     show this help message and exit
  --r_max R_MAX  Maximum radius (default: 1000)
  --x_0 X_0      Minimum x value (default: -8.0)
  --dx DX        Grid spacing (default: 0.01)
  --Z Z          Atomic number (default: 1)
  --n_max N_MAX  Maximum number of eigenvalues (default: 20)
  --r_ph R_PH    Phase shifts evaluation point (default: 800)
  --b B          Inverse of the range of true potential (default: 1)
```

To use the default values run simply:
```bash
$ python code.py 
```
All the lenghts are expressed in atomic units. The value $x_0 = -8$ corresponds to a minimun radius $r_{min} = 3 x 10^{-4} a_0$, with $a_0$ the Bohr radius.

Notice that, in order to obtain convergent results in **`code.py`**, not all combinations of paramter values are valid. For instance, the maximum eigenvalue index $n_{max}$ is related to the maximum radius $r_{max}$, such that the outermost node of the wavefunction lies within within $r_{max}$.


