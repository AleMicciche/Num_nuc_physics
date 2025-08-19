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
python code.py --r_max 1200 --x_0 -9.0 --dx 0.01 --Z 1 --n_max 20 --r_ph 800 --b 1
```
with parameters representing:
```bash
python code.py --r_max 1200 --x_0 -9.0 --dx 0.01 --Z 1 --n_max 20 --r_ph 800 --b 1
```




