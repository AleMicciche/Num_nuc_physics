#This code solves the radial Schroedinger equation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
from scipy.optimize import differential_evolution
import inspect
import os
import argparse

#set display quality 
#plt.rcParams['savefig.dpi'] = 300
#plt.rcParams['figure.dpi'] = 180

#new exception to catch convergence-related errors during the optimization algorithm 
class ConvergenceError(Exception):
    pass

#initialize the grid in the variable x(r), r radial coordinate
def init_grid(N, x_0, dx, Z):
    
    #prepare an array of N elements, with distance dx
    x = np.linspace(x_0, x_0 + (N-1) * dx, N)

    #generate r, sqrt_r, and r^2 based on logarithmic x grid
    r = np.exp(Z*x)
    sqrt_r = np.sqrt(r)
    r2 = r * r

    #print grid information
    print("Radial grid information:\n")
    print("dx = ", dx)
    print("x_min = ", x_0)
    print("n_points = ", N)
    print("r(0) = ", r[0])
    print("r(n_points) = ", r[N-1])
    print("-----------------------------------------------\n")

    return r, sqrt_r, r2


#save the potential in a .parquet and plot it
def save_pot(r, pot, low_lim_plot, up_lim_plot, name, title=None):
    
    if title is None:
        title = name
    
    # Create folder if it doesn't exist
    if not os.path.exists(name):
        os.makedirs(name)
    
    df_pot = pd.DataFrame({"r": r, "V(r)": pot})
    df_pot.to_parquet(f"{name}/potential.parquet", index = False)

    plt.figure()
    plt.title(f"{title} potential")
    plt.plot(r, pot, 'o')
    plt.xlabel("r     (Bohr radii)")
    plt.ylabel("V(r)     (Ry)")
    plt.xlim(low_lim_plot, up_lim_plot)
    plt.tight_layout()
    plt.savefig(f"{name}/potential.png")

#define the function to solve the radial Schroedinger equation with potential pot
def solve_schr(N, dx, grid, pot, n, l, Z, i_ph, verbose=True):

    #assign these values from the defined grid
    r, sqrt_r, r2 = grid

    eps = 1e-10 #tolerance threshlod for convergence of the eigenvalue 
    n_iter = 1000 #iterations after the convergence is evaluated

    #initial lower and upper bounds for eigenvalue
    eup = pot[N-1]
    elw = eup 

    for i in range(N):
        elw = min(elw, ((l + 0.5) * (l + 0.5)) / r2[i] + pot[i] )

    if (eup - elw < eps):
        print("solve_schr: lower and upper bound in the energy interval searching are equal:", eup, elw)
        raise ConvergenceError("Schroedinger solver did not converge due to bounds.")
    
    #initial energy and energy correction
    e = (eup + elw) * 0.5 
    de = 1e+10 
    
    f = np.zeros(N)

    k= 0
    while k < n_iter and abs(de) > eps:

        #Debug: print(f"eup={eup}, elw = {elw}, e ={e}")
    
        #define the function f and determine the position of its last change of sign. f plays the role of (V_eff - E) in the radial schroedinger equation in the coordinate r
        icl = -1 

        f[0] = ((l + 0.5) * (l + 0.5) + r2[0] * (pot[0] - e))
        for i in range(1, N):
            f[i] = ((l + 0.5) * (l + 0.5) + r2[i] * (pot[i] - e))
            
            #handle the unlikely case f[i] == 0
            if f[i] == 0:
                f[i] = 1e-20 

            #check the sign change in f
            if  f[i] != math.copysign(f[i], f[i - 1]):
                icl = i
        
        if icl < 0 or icl >= N - 3:
            print(f"icl={icl:4d}, n_iter = {k+1}, n={n}, f[N-5]={f[N-5]}, f[N-4]={f[N-4]}, f[N-3]={f[N-3]}, f[N-2]={f[N-2]}, f[N-1] ={f[N-1]}")
            print("error in solve_sheq: last change of sign too far")
            sys.exit(1)

        #redefine the function f to be inserted in numerov algorithm
        for i in range(N):
            f[i] = 1 - (dx * dx / 12) * f[i]

        #wavefunction
        y = np.zeros(N)

        nodes = n - l - 1

        #determine the wave function in the first two points (i.e. near r=0)
        if np.allclose(pot, -2 * Z / r):
        #if the potential is coulomb, use the known asymptotic behaviour in r=0 from analytic solutions
            y[0] = (r[0] ** (l + 1)) * (1 - Z * r[0] /(l+1)) / sqrt_r[0]
            y[1] = (r[1] ** (l + 1)) * (1 - Z * r[1] /(l+1)) / sqrt_r[1]
        else:
        #otherwise, use the asymptotic behaviour based simply on the leading r^{-2} dependence of the potential in r=0
            y[0] = (r[0] ** (l + 1)) / sqrt_r[0]
            y[1] = (r[1] ** (l + 1)) / sqrt_r[1]

        #start outward integration from x_0 to x_icl
        n_cross = 0
        for i in range(1, icl):
            y[i + 1] = ((12 - 10 *f[i]) * y[i] - f[i - 1] * y[i - 1]) / f[i + 1]
            if y[i] != math.copysign(y[i], y[i + 1]): 
                n_cross += 1
        fac = y[icl]
        
        #check the number of crossing
        if n_cross != nodes:
            #the eigenfunction do not belongs to the energy eigenvalue e: adjust energy interval
            if n_cross > nodes:
                eup = e
            else:
                elw = e

            e = (eup + elw) * 0.5
        else:
            #the eigenvalue e is correct: inward integration from x_{N-1} to x_icl
            y[N-1] = dx
            y[N-2] = (12 - 10 * f[N-1]) * y[N-1] / f[N-2] 

            for i in range(N - 2, icl, -1):
                y[i - 1] = ((12 - 10 *f[i]) * y[i] - f[i + 1] * y[i + 1]) / f[i - 1]
                if y[i-1] > 1e10:
                    for j in range(N - 1, i - 2, -1):
                        y[j] /= y[i - 1]
                
            #rescale to match at the point icl
            fac /= y[icl]
            for i in range(icl, N):
                y[i] *= fac

            #normalize the wavefunction
            norm = 0
            for i in range(N):
                norm += y[i] * y[i] * r2[i] * dx
            norm = np.sqrt(norm)
            for i in range(N):
                y[i] /= norm
            
            #the left and right derivative of the solution at icl are different. Assume the discontinuity comes from a delta function centered at icl (f[icl] -> fcusp). ycusp the solution in icl for this new f-function
            ycusp = (y[icl - 1] * f[icl - 1] + y[icl + 1] * f[icl + 1]  + 10 * f[icl] * y[icl]) / 12
            
            #from numerov method: f[icl]  y[icl] = fcusp * ycusp 
            dfcusp = f[icl] * (y[icl] / ycusp - 1)

            # eigenvalue update using perturbation theory
            de = - 12 / dx * ycusp * ycusp * dfcusp
            #Debug: print(f"de = {de}")
            if de < 0.0:
                elw = e
            if de > 0.0:
                eup = e

            #Prevent e from going out of bounds
            e = e - de
            e = min(e, eup)
            e = max(e, elw)

            #prevent last change of sign from occuring in the last point
            if e == eup: 
                e = (eup + elw)/2
        
        k += 1

    # ---- convergence not achieved -----
    if abs(de) > eps:
        if n_cross != nodes:
            print(f"n_cross={n_cross:4d} nodes={nodes:4d} icl={icl:4d} e={e:16.8e} elw={elw:16.8e} eup={eup:16.8e} n={n}")
        else:
            print(f"e={e:16.8e}  de={de:16.8e}")
        print(f"solve_sheq not converged after {n_iter} iterations, n={n}")
        # Print c and d_1 if available in the caller context
        frame = inspect.currentframe().f_back
        c = frame.f_locals.get('c', None)
        d_1 = frame.f_locals.get('d_1', None)
        if c is not None:
            print(f"c = {c}")
        if d_1 is not None:
            print(f"d_1 = {d_1}")
        raise ConvergenceError(f"Schroedinger solver did not converge for c={c}, d_1={d_1}")

    # ---- convergence has been achieved -----
    if verbose:
        print(f"convergence achieved at iter # {k:4d}, de = {de:16.8e}")
    
    #compute phase shift at r(x_{i_ph})
    ph_shift = np.arcsin(sqrt_r[i_ph] * y[i_ph]/2) - np.sqrt(np.absolute(e)) * r[i_ph] + (l * np.pi)/2

    return e, y, ph_shift


def result (N, dx, grid, pot, Z, i_ph, n_max, e_arr, chi_arr, ph_shift_arr, name, title = None):
    
    if title is None:
        title = name
    
    #minimal value, l=0 fixed
    n=1

    if not os.path.exists(f"{name}/eigenfunctions"):
        os.makedirs(f"{name}/eigenfunctions")
    if not os.path.exists(f"{name}/eigenvalues"):
        os.makedirs(f"{name}/eigenvalues")
    if not os.path.exists(f"{name}/phase"):
        os.makedirs(f"{name}/phase")

    for i in range(1, n_max + 1):
        e_arr[i-1], chi_arr[i-1], ph_shift_arr[i-1] = solve_schr(N, dx, grid, pot, i, 0, Z, i_ph)
        
        #compute the chi function chi(r) = r^{1/2} y(x(r)) from y returned from solve_schr
        chi_arr[i-1] = chi_arr[i-1] * grid[1]

        #save datasets and plot them
        df_chi = pd.DataFrame({"r": grid[0], "chi": chi_arr[i-1]})
        df_chi.to_parquet(f"{name}/eigenfunctions/eigenfunc_n_{i}.parquet", index=False)

        plt.figure()
        plt.plot(r, chi_arr[i-1], color = 'blue')
        plt.xlabel("r  (Bohr radii)")
        plt.ylabel(r'$\chi(r) $')
        plt.title(f"{title} potential - eigenfunction n = {i}")
        plt.grid(True)
        plt.savefig(f"{name}/eigenfunctions/eigenfunc_n_{i}.png")
        plt.close()

    n_arr = np.arange(1, n_max + 1)
    
    df_e = pd.DataFrame({"n": n_arr, "E": e_arr})
    print("-----------------------------------------------\n")
    print(f"{name} energy eigenvalues\n")
    print(df_e.to_markdown(index=False))
    df_e.to_parquet(f"{name}/eigenvalues/eigenvalues.parquet", index=False)

    df_ph = pd.DataFrame({"n": n_arr, "phase shift": ph_shift_arr, "phase shift mod 2π (π units)": np.mod(ph_shift_arr, 2 * np.pi)/np.pi })
    print("-----------------------------------------------\n")
    print(f"{name} phase shifts at r = {r_ph} \n")
    print(df_ph.to_markdown(index=False))
    df_ph.to_parquet(f"{name}/phase/phase_shift_r_{r_ph}.parquet", index=False)

    return e_arr, chi_arr, ph_shift_arr

    
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the radial Schrödinger equation.")
    parser.add_argument("--r_max", type=float, default=1000, help="Maximum radius (default: %(default)s)")
    parser.add_argument("--x_0", type=float, default=-8.0, help="Minimum x value (default: %(default)s)")
    parser.add_argument("--dx", type=float, default=0.01, help="Grid spacing (default: %(default)s)")
    parser.add_argument("--Z", type=int, default=1, help="Atomic number (default: %(default)s)")
    parser.add_argument("--n_max", type=int, default=20, help="Maximum number of eigenvalues (default: %(default)s)")
    parser.add_argument("--r_ph", type=float, default=800, help="Phase shifts evaluation point (default: %(default)s)")
    parser.add_argument("--b", type=float, default=1, help="Inverse of the range of true potential (default: %(default)s)")
    args = parser.parse_args()

#initialize grid
r_max = args.r_max
x_0 = args.x_0 #x= -8 corresponds to r_min = 3 * 1E-4 Bohr radii
dx = args.dx #grid spacing
Z = args.Z #atomic number
N = int((np.log(Z * r_max) - x_0) / dx)  #number of points on the grid
grid = init_grid(N, x_0, dx, Z)
r = grid[0]

#index of the last energy eigenvalue to be computed
n_max = args.n_max

#r in which the phase shift is evaluated and its index in the grid
r_ph = args.r_ph
i_ph = int(np.floor((np.log(Z * r_ph) - x_0)/dx))

#define Coulomb potential
name_cou = "Coulomb"
pot_cou = -2 * Z / r
save_pot(r, pot_cou, 0, 0.1, name_cou)

#--------------------------Coulomb potential analysis-----------------------------
print("\n" + "-"*50)
print(f"Coulomb theory")
print("-"*50)
print(f"\n Atomic number Z = {Z}")

#compute eigenvalues, eigenfunctions, phase shifts
e_cou = np.zeros(n_max)
chi_cou = np.zeros((n_max, N))
ph_shift_cou = np.zeros(n_max)
e_cou, chi_cou, ph_shift_cou = result(N, dx, grid, pot_cou, Z, i_ph, n_max, e_cou, chi_cou, ph_shift_cou, name_cou)

#analytical results (e_n = -1/n^2)
e_th = -1 / (np.arange(1, n_max + 1) ** 2)
#plot and save the relative errors w.r.t. analytical results 
plt.figure()
plt.plot(
    np.absolute(e_th),
    np.absolute(e_cou - e_th) / np.absolute(e_th),
    marker='o',
    linestyle='-',
)
plt.xlabel(r'|E|')
plt.ylabel(r'$|\Delta E / E|$')
plt.title(f"relative error w.r.t analytical values")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.savefig(f"plot/relative_error_coulomb_analyt.png")
plt.show()
plt.close()

#--------------------------True potential analysis-----------------------------
print("\n" + "-"*50)
print(f"Exact theory")
print("-"*50)

name_true = "true"

#yukawa parameter: the inverse of the characteristic length
b = args.b
print(f"\n Parameter b = {b}") 

#define the true potential: coulomb + yukawa-type potential
pot_true = pot_cou - 2 * Z * (np.exp(- b * r) /r)
save_pot(r, pot_true, 0, 0.1, name_true)

#compute eigenvalues, eigenfunctions, phase shifts
e = np.zeros(n_max)
chi = np.zeros((n_max, N))
ph_shift = np.zeros(n_max)
e, chi, ph_shift = result(N, dx, grid, pot_true, Z, i_ph, n_max, e, chi, ph_shift, name_true)

#----------------(first order perturb. th.) delta potential analysis------------------
print("\n" + "-"*50)
print(f"First order perturbation theory analysis")
print("-"*50)
#fitting the free paramter k to the lowest eigenvalue |E^{true}_{n_max}|: the smallest relative error w.r.t. corresponding coulomb eigenvalues 
k = (e[n_max -1] - e_th[n_max -1]) * np.sqrt(np.pi) * n_max**3
print(f"\n Coefficient delta function: k = {k}")

#define the first order prediction for the eigenvalues
e_delta = np.zeros(n_max) 
n_arr = np.arange(1, n_max + 1)
e_delta = e_th + k / (np.sqrt(np.pi) * n_arr**3)

#--------------------------Effective potential analysis-----------------------------

#define effective potential with three paramters
def pot_eff(a, c, d_1):
    return pot_cou + c * a**2 * (np.exp(- (r**2) / (2 * a**2))) / ((2 * np.pi)**1.5 * a**3) + d_1 * a**4 * (-3 + r**2 / a**2) * (np.exp(- (r**2) / (2 * a**2))) / ((2 * np.pi)**1.5 * a**5)

#define the function to minimize in order to find c and d
def objective(params):
    #handle both cases
    if len(params) == 1:
        c = params[0]
        d_1 = 0
    else:
        c, d_1 = params
    try:
        _, _, ph_shift_eff_n_max = solve_schr(N, dx, grid, pot_eff(a, c, d_1), n_max, 0, Z, i_ph, verbose=False) 
        return (ph_shift_eff_n_max - ph_shift[n_max-1])**2
    except ConvergenceError:
        #ignore the values for which convergence is not achieved: the objective function return infinity for them
        return np.inf


# List of a values and other useful paramterers to iterate over them 
setting_a_values = [
    {
        'a': 1,
        'bounds_a4': [(-60, -50), (-4, -2)], #a^4 theory: bounds of c and d_1 for optimization alghorithm 
        'bounds_a2': [(-60, -40)], #a^2 theory: bounds of c (d_1 = 0) for optimization alghorithm
        'name_a4': 'eff_a_4/a1',
        'name_a2': 'eff_a_2/a1',
        'title_a4': 'Effective a^4 (a = 1)',
        'title_a2': 'Effective a^2 (a = 1)'
    },
    {
        'a': 3,
        'bounds_a4': [(-50, -30), (-12, -4)],
        'bounds_a2': [(-40, -20)],
        'name_a4': 'eff_a_4/a3',
        'name_a2': 'eff_a_2/a3',
        'title_a4': 'Effective a^4 (a = 3)',
        'title_a2': 'Effective a^2 (a = 3)'
       
    },
    {
        'a': 10,
        'bounds_a4': [(-40, -20), (-15, -4)],
        'bounds_a2': [(-20, -10)],
        'name_a4': 'eff_a_4/a10',
        'name_a2': 'eff_a_2/a10',
        'title_a4': 'Effective a^4 (a = 10)',
        'title_a2': 'Effective a^2 (a = 10)'
    },
    {
        'a': 0.1,
        'bounds_a4': [(-160, -140), (-16, -8)],
        'bounds_a2': [(-150, -130)],
        'name_a4': 'eff_a_4/a01',
        'name_a2': 'eff_a_2/a01',
        'title_a4': 'Effective a^4 (a = 0.1)',
        'title_a2': 'Effective a^2 (a = 0.1)'
    }
]

# Prepare lists to collect results for plotting
e_a4_list = []
e_a2_list = []
ph_shift_a4_list = []
ph_shift_a2_list = []


for set in setting_a_values:
    
    print("\n" + "-"*50)
    print(f"Effective theory: a = {set['a']}")
    print("-"*50)

    a = set['a']
    
    #-------a^4 theory-----------
     
    print("\n[ a^4 theory ]")
    
    #find zeros of objective function
    print("\n Optimization algorithm")
    result_a4 = differential_evolution(objective, set['bounds_a4'], maxiter=100, popsize=15)
    c_opt, d_1_opt = result_a4.x
    print(f"\n objective = {objective([c_opt, d_1_opt])}, best parameters for a = {a}: c = {c_opt}, d = {d_1_opt}")
    
    #set potential with optimal values
    pot_a4 = pot_eff(a, c_opt, d_1_opt)
    save_pot(r, pot_a4, 0, 0.01, set['name_a4'], set['title_a4'])

    #compute eigenfunctions, eigenvalues, phase shifts
    print("\n Solving Schroedinger equation")
    e_a4 = np.zeros(n_max)
    chi_a4 = np.zeros((n_max, N))
    ph_shift_a4 = np.zeros(n_max)
    e_a4, chi_a4, ph_shift_a4 = result(N, dx, grid, pot_a4, Z, i_ph, n_max, e_a4, chi_a4, ph_shift_a4, set['name_a4'], set['title_a4'])

    # Store results for plotting
    e_a4_list.append(e_a4)
    ph_shift_a4_list.append(ph_shift_a4)

    #-------a^2 theory (d_1 = 0)-------
    
    print("\n[ a^2 theory (d_1 = 0) ]")

    #find zeros of objective function
    print("\n Optimization algorithm")
    result_a2 = differential_evolution(objective, set['bounds_a2'], maxiter=100, popsize=15)
    c_0_opt = result_a2.x[0]
    print(f"\n objective = {objective([c_0_opt])}, best parameters for a = {a}: c_0 = {c_0_opt}  (d_1 = 0)")

    #set potential with optimal values
    pot_a2 = pot_eff(a, c_0_opt, 0)
    save_pot(r, pot_a2, 0, 0.1, set['name_a2'], set['title_a2'])

    #compute eigenfunctions, eigenvalues, phase shifts
    print("\n Solving Schroedinger equation")
    e_a2 = np.zeros(n_max)
    chi_a2 = np.zeros((n_max, N))
    ph_shift_a2 = np.zeros(n_max)
    e_a2, chi_a2, ph_shift_a2 = result(N, dx, grid, pot_a2, Z, i_ph, n_max, e_a2, chi_a2, ph_shift_a2, set['name_a2'], set['title_a2'])

    # Store results for plotting
    e_a2_list.append(e_a2)
    ph_shift_a2_list.append(ph_shift_a2)

#------------------------------plotting relative errors--------------------------------------

#eigenvalues (true w.r.t. delta, Coulomb, a=1)
plt.figure(figsize=(7, 5))
plt.plot(
    np.absolute(e),
    np.absolute(e_th - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'blue',
    label = "Coulomb"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_delta - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'red',
    label = r'Coulomb + $ c \delta^3(r) \, (1^{st}$ order)'
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a4_list[0] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'green',
    label = "a^4 (a=1)"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a2_list[0] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'yellow',
    label = "a^2 (a=1)"
)

plt.legend(fontsize = 14)
plt.xlabel(r'$|E|$', fontsize=17)
plt.ylabel(r'$\frac{|\Delta E|}{|E|}$', rotation=0, labelpad=20, fontsize=17, va='center')
plt.title(f"Energy - relative difference w.r.t. true eigenvalues")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.subplots_adjust(left=0.18)
plt.savefig(f"plot/eigenvalues_relative_diff_all.png")
plt.show()
plt.close()

#eigenvalues (true w.r.t. a^4, a = 1, 3, 10, 0.1)
plt.figure(figsize=(7, 5))
plt.plot(
    np.absolute(e),
    np.absolute(e_a4_list[0] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'green',
    label = "a=1"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a4_list[1] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'red',
    label = "a=3"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a4_list[2] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'blue',
    label = "a=10"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a4_list[3] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'black',
    label = "a=0.1"
)

plt.legend(fontsize = 14)
plt.xlabel(r'$|E|$', fontsize=17)
plt.ylabel(r'$\frac{|\Delta E|}{|E|}$', rotation=0, labelpad=20, fontsize=17, va='center')
plt.title(f"Energy - relative difference w.r.t. true eigenvalues (a^4)")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.subplots_adjust(left=0.18)
plt.savefig(f"plot/eigenvalues_relative_diff_a_4.png")
plt.show()
plt.close()

#eigenvalues (true w.r.t. a^2, a = 1, 3, 10, 0.1)
plt.figure(figsize=(7, 5))
plt.plot(
    np.absolute(e),
    np.absolute(e_a2_list[0] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'yellow',
    label = "a=1"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a2_list[1]  - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'red',
    label = "a=3"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a2_list[2] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'blue',
    label = "a=10"
)

plt.plot(
    np.absolute(e),
    np.absolute(e_a2_list[3] - e) / np.absolute(e),
    marker='o',
    linestyle='-',
    color = 'black',
    label = "a=0.1"
)

plt.legend(fontsize = 14)
plt.xlabel(r'$|E|$', fontsize=17)
plt.ylabel(r'$\frac{|\Delta E|}{|E|}$', rotation=0, labelpad=20, fontsize=17, va='center')
plt.title(f"Energy - relative difference w.r.t. true eigenvalues (a^2)")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.subplots_adjust(left=0.18)
plt.savefig(f"plot/eigenvalues_relative_diff_a_2.png")
plt.show()
plt.close()


#phase shift (true w.r.t. Coulomb, a=1)
plt.figure(figsize=(7, 5))
plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_cou - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'blue',
    label = "Coulomb"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a4_list[0] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'green',
    label = "a^4 (a=1)"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a2_list[0] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'yellow',
    label = "a^2 (a=1)"
)

plt.legend(fontsize = 14)
plt.xlabel(r'$|E|$', fontsize=17)
plt.ylabel(r'$|\Delta \delta(E)|$', rotation=0, labelpad=30, fontsize=17, va='center')
plt.title(f"Phase shift - relative difference w.r.t. true phase shift")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.subplots_adjust(left=0.20)
plt.savefig(f"plot/phase_relative_diff_r_{r_ph}_all.png")
plt.show()
plt.close()

#phase shift (true w.r.t. a^4, a = 1, 3, 10, 0.1)
plt.figure(figsize=(7, 5))
plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a4_list[0] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'green',
    label = "a=1"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a4_list[1] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'red',
    label = "a=3"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a4_list[2] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'blue',
    label = "a=10"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a4_list[3] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'black',
    label = "a=0.1"
)

plt.legend(fontsize = 14)
plt.xlabel(r'$|E|$', fontsize=17)
plt.ylabel(r'$|\Delta \delta(E)|$', rotation=0, labelpad=30, fontsize=17, va='center')
plt.title(f"Phase shift - relative difference w.r.t. true phase shift (a^4)")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.subplots_adjust(left=0.20)
plt.savefig(f"plot/phase_relative_diff_r_{r_ph}_a_4.png")
plt.show()
plt.close()

#phase shift (true w.r.t. a^2, a = 1, 3, 10, 0.1)
plt.figure(figsize=(7, 5))
plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a2_list[0] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'yellow',
    label = "a=1"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a2_list[1] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'red',
    label = "a=3"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a2_list[2] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'blue',
    label = "a=10"
)

plt.plot(
    np.absolute(e),
    np.absolute(ph_shift_a2_list[3] - ph_shift) / np.absolute(ph_shift),
    marker='o',
    linestyle='-',
    color = 'black',
    label = "a=0.1"
)

plt.legend(fontsize = 14)
plt.xlabel(r'$|E|$', fontsize=17)
plt.ylabel(r'$|\Delta \delta(E)|$', rotation=0, labelpad=30, fontsize=17, va='center')
plt.title(f"Phase shift - relative difference w.r.t. true phase shift (a^2)")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.subplots_adjust(left=0.20)
plt.savefig(f"plot/phase_relative_diff_r_{r_ph}_a_2.png")
plt.show()
plt.close()




