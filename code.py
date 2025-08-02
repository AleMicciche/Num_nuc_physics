#This code solves the radial Schroedinger equation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math

#set display quality 
#plt.rcParams['savefig.dpi'] = 300
#plt.rcParams['figure.dpi'] = 180

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


#save the potential in a .parquet and its plot
def save_pot(r, pot, low_lim_plot, up_lim_plot, name):

    df_pot = pd.DataFrame({"r": r, "V(r)": pot})
    df_pot.to_parquet(f"{name}/potential.parquet", index = False)
    #print(df_pot)

    plt.figure()
    plt.title(f"{name} potential")
    plt.plot(r, pot, 'o')
    plt.xlabel("r     (Bohr radii)")
    plt.ylabel("V(r)     (Ry)")
    plt.xlim(low_lim_plot, up_lim_plot)
    plt.tight_layout()
    plt.savefig(f"{name}/potential.png")

#define the function to solve the readial Schroedinger equation with the potential v_pot
def solve_schr(N, dx, grid, pot, n, l, Z, i_ph):

    #assign these values from the defined grid
    r, sqrt_r, r2= grid

    eps = 1e-10 #tolerance threshlod for convergence of the eigenvalue 
    n_iter = 200 #iterations after the convergence is evaluated

    #initial lower and upper bounds to eigenvalue
    eup = pot[N-1]
    elw = eup 

    for i in range(N):
        elw = min(elw, ((l + 0.5) * (l + 0.5)) / r2[i] + pot[i] )

    if (eup - elw < eps):
        print("solve_schr: lower and upper bound in the energy interval searching are equal:", eup, elw)
        sys.exit(1)
    
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

        #determine the wave function in the first two points (i.e. near r=0)
        nodes = n - l - 1

        """y[0] = (r[0] ** (l + 1)) * (1 - Z * r[0] /(l+1)) / sqrt_r[0]
        y[1] = (r[1] ** (l + 1)) * (1 - Z * r[1] /(l+1)) / sqrt_r[1]"""

        #if the potential is coulomb, use the known asymptotic behaviour in r=0 from analytic solutions
        if np.allclose(pot, -2 * Z / r):
            y[0] = (r[0] ** (l + 1)) * (1 - Z * r[0] /(l+1)) / sqrt_r[0]
            y[1] = (r[1] ** (l + 1)) * (1 - Z * r[1] /(l+1)) / sqrt_r[1]
        #otherwise, use the asymptotic behaviour based simply on the leading r^{-2} dependence of the potential in r=0
        else:
            y[0] = (r[0] ** (l + 1)) / sqrt_r[0]
            # y[0] = 1E-12
            # y[1] = 1E-5
            y[1] = (r[1] ** (l + 1)) / sqrt_r[1]

        #outward integration from x_0 to x_icl
        n_cross = 0
        for i in range(1, icl):
            y[i + 1] = ((12 - 10 *f[i]) * y[i] - f[i - 1] * y[i - 1]) / f[i + 1]
            if y[i] != math.copysign(y[i], y[i + 1]): 
                n_cross += 1
        #Debug: print(f"n_cross = {n_cross}")
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
            #the eigenvalue e is correct: integration from x_{N-1} inward to x_icl
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

            #find the value of the cusp at the matching point
            ycusp = (y[icl - 1] * f[icl - 1] + y[icl + 1] * f[icl + 1]  + 10 * f[icl] * y[icl]) / 12
            dfcusp = f[icl] * (y[icl] / ycusp - 1)

            # Eigenvalue update using perturbation theory
            de = 12 / dx * ycusp * ycusp * dfcusp
            #Debug: print(f"de = {de}")
            if de > 0.0:
                elw = e
            if de < 0.0:
                eup = e

            # Prevent e from going out of bounds
            e = e + de
            e = min(e, eup)
            e = max(e, elw)
        
        k += 1

    # ---- convergence not achieved -----
    if abs(de) > eps:
        if n_cross != nodes:
            print(f"n_cross={n_cross:4d} nodes={nodes:4d} icl={icl:4d} e={e:16.8e} elw={elw:16.8e} eup={eup:16.8e} n={n}")
        else:
            print(f"e={e:16.8e}  de={de:16.8e}")
        
        print(f"solve_sheq not converged after {n_iter} iterations, n={n}")
        sys.exit(1)

    # ---- convergence has been achieved -----
    print(f"convergence achieved at iter # {k:4d}, de = {de:16.8e}")
    
    #compute phase shift at r(x_{i_ph})
    ph_shift = np.arcsin(sqrt_r[i_ph] * y[i_ph]/2) - np.sqrt(np.absolute(e)) * r[i_ph] + (l * np.pi)/2

    return e, y, ph_shift


def result (N, dx, grid, pot, Z, i_ph, n_max, e_arr, chi_arr, ph_shift_arr, name):
    
    #minimal value, l=0 fixed
    n=1

    for i in range(1, n_max + 1):
        e_arr[i-1], chi_arr[i-1], ph_shift_arr[i-1] = solve_schr(N, dx, grid, pot, i, 0, Z, i_ph)
        #compute the chi function chi(r) = r^{1/2} y(x(r)) from y returned from solve_schr
        chi_arr[i-1] = chi_arr[i-1] * grid[1]

        #save datasets and plot them
        df_chi = pd.DataFrame({"r": grid[0], "chi": chi_arr[i-1]})
        df_chi.to_parquet(f"{name}/eigenfunctions/eigenfunc_{name}_n_{i}.parquet", index=False)

        plt.figure()
        plt.plot(r, chi_arr[i-1], color = 'blue')
        plt.xlabel("r  (Bohr radii)")
        plt.ylabel(r'$\chi(r) $')
        plt.title(f"{name} potential eigenfunction n = {i}")
        plt.grid(True)
        plt.savefig(f"{name}/eigenfunctions/eigenfunc_{name}_n_{i}.png")
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

    
#-----------------------------------------------------------------------

#initialize grid
r_max = 1000
x_0 = -8. #corresponds to r_min == 3 * 1E-4 Bohr radii
dx = 0.01 #grid spacing
Z = 1 #atomic number
N = int((np.log(Z * r_max) - x_0) / dx)  #number of points on the grid
grid = init_grid(N, x_0, dx, Z)

#index of the last energy eigenvalue to be computed
n_max = 20

#r in which the phase shift is evaluated and its index in the grid
r_ph = 800
i_ph = int(np.floor((np.log(Z * r_ph) - x_0)/dx))

#define Coulomb potential
name_cou = "Coulomb"
r = grid[0] 
pot_cou = -2 * Z / r
save_pot(r, pot_cou, 0, 0.1, name_cou)
#--------------------------Coulomb potential analysis-----------------------------

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
plt.savefig(f"{name_cou}/eigenvalues/relative_error_analytical_result.png")
plt.show()
plt.close()

#--------------------------True potential analysis-----------------------------

#define the true potential: coulomb + anti-screening yukawa-type potential
name_true = "true"
a = 0.2 #anti-screening parameter
pot_true = pot_cou - 2 * Z * (np.exp(- a * r) /r)
save_pot(r, pot_true, 0, 0.1, name_true)

#compute eigenvalues, eigenfunctions, phase shifts
e = np.zeros(n_max)
chi = np.zeros((n_max, N))
ph_shift = np.zeros(n_max)
e, chi, ph_shift = result(N, dx, grid, pot_true, Z, i_ph, n_max, e, chi, ph_shift, name_true)

#----------------(first order perturb. th.) delta potential analysis------------------
#fitting the free paramter c to the lowest eigenvalues E^{true}_{n_max}: the smallest relative error w.r.t. corresponding coulomb eigenvalues
c = (e[n_max -1] - e_th[n_max -1]) * np.sqrt(np.pi) * n_max**3
#define the first order prediction for the eigenvalues
e_delta = np.zeros(n_max) 
n_arr = np.arange(1, n_max + 1)
e_delta = e_cou + c / (np.sqrt(np.pi) * n_arr**3)


#plot and save the relative difference between true eigenvalues and coulomb ones, and true eigenvalues and those computed from the delta potential
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

plt.text(
    0.95, 0.25,
    f'$c = {c:.2f}$',
    fontsize=14,
    color='black',
    ha='right',
    va='bottom',
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3')
)
plt.legend(fontsize = 14)
plt.xlabel(r'$E_n$', fontsize=17)
plt.ylabel(r'$\frac{|\Delta E_n|}{|E_n|}$', rotation=0, labelpad=20, fontsize=17, va='center')
plt.title(f"Relative difference w.r.t. true eigenvalues")
plt.grid(True)
plt.xscale("log")
plt.yscale("log")
plt.subplots_adjust(left=0.18)
plt.savefig(f"{name_true}/eigenvalues/relative_diff_{name_true}_coulomb_delta.png")
plt.show()
plt.close()




    