## Overview

This repository contains the code and results from testing several effective potentials to describe the low-energy behavior of a given short-range potential. A brief review of the theory, details of the chosen potentials, references, and a summary of the main results are provided in **`summary_project.pdf`**.

## Repository structure

All results produced by **`code.py`** are saved in:

- **`Coulomb/`** — for a Coulomb potential  
- **`true/`** — for the chosen short-range potential  
- **`eff_a_4/`** — for an effective field theory up to order $a^4$, with subfolders for different values of $a$ 
- **`eff_a_2/`** — for an effective field theory up to order $a^2$, with subfolders for different values of $a$

Each folder is organized into subfolders by result type (eigenfunctions, eigenvalues, phase shifts).

> Example output from a run is shown in **`terminal.txt`**.
