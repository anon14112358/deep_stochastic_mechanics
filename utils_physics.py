import numpy as np
import torch
import os
from scipy import sparse
from scipy import integrate
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def V_x_i(x): # it's grad of V
    return (x - 0.1*torch.ones(x.shape).to(device))

def v_0(x, x0_phase=0, h=1e-2, m=1):
    if x0_phase==0:
        return h / m * torch.zeros(x.shape).to(device)
    else:
        return h / m * torch.tensor(x0_phase).repeat(x.shape[0]).reshape(-1, 1)
    
def u_0(x, sig2, h=1e-2, m=1):
    return -h * x / (2 * m * (sig2**2))

def time_transform(t):
    return t


def find_num_solution(sig2, time_splits, h=1e-2, m=1, x0_phase=0.0, dx=np.sqrt(1 / 40000), T=1):
    x     = np.arange(-2, 2, dx)      # spatial grid points
    sigma = sig2                      # width of initial gaussian wave-packet
    x0    = 0.0                       # center of initial gaussian wave-packet

    A = 1.0 / (sigma * np.sqrt(2*np.pi)) # normalization constant

    # Initial Wavefunction
    psi0 = np.sqrt(A) * np.exp(-(x-x0)**2 / (4.0 * (sigma**2))) * np.exp(1j * x0_phase * x )

    # Potential V(x)
    x_Vmin = 0.1         # center of V(x)

    V = 0.5 * (x - x_Vmin)**2

    # Laplace Operator (Finite Difference)
    D2 = sparse.diags([1, -2, 1], [-1, 0, 1], shape=(x.size, x.size)) / dx**2
    # RHS of Schrodinger Equation
    hbar = h
    # hbar = 1.0545718176461565e-34
    def psi_t(t, psi):
        return -1j * (-0.5 * h  / m * D2.dot(psi) + V / hbar * psi)

    dt = T / 1000 # T / 10000 #T / 1000  # time interval for snapshots
    t0 = 0.0    # initial time
    tf = T    # final time
    t_eval = time_splits#np.arange(t0, tf, dt) # np.linspace(t0, tf, len(x)**2) # np.arange(t0, tf, dt)  # recorded time shots

    # Solve the Initial Value Problem
    sol = integrate.solve_ivp(psi_t, t_span = [t0, tf], y0 = psi0, t_eval = t_eval, method="RK23")
    bmeans = []
    bstds = []
    ts = []
    for i, t in enumerate(sol.t):
        ts.append(t)
        bmeans.append(np.dot(x, dx * np.abs(sol.y[:,i])**2)) 
        
        bstds.append(np.dot((x - bmeans[-1]) ** 2, dx * np.abs(sol.y[:,i])**2))
    return sol.y.T, bmeans, bstds


def plot_stats(time_splits, m, v, t, sol_m, sol_v):
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(8, 3)
    x = time_splits.numpy()

    axs[0].plot(x, m.mean(axis=0), color='green', label='preds', linewidth=0.8)
    axs[0].plot(x, m.mean(axis=0) - m.std(axis=0), linestyle='--', color='seagreen', linewidth=0.8)
    axs[0].plot(x, m.mean(axis=0) + m.std(axis=0), linestyle='--', color='seagreen', linewidth=0.8)
    axs[0].fill_between(x, m.mean(axis=0) - m.std(axis=0), 
                        m.mean(axis=0) + m.std(axis=0), color='seagreen', 
                        alpha=0.5, linewidth=0.8)
    axs[0].plot(t, sol_m, color='black', linestyle='--', label='truth', linewidth=0.8)
    axs[0].set_xlabel('time')
    axs[0].legend();
    axs[0].set_title('Harmonic oscillator 1D mean')

    axs[1].plot(x, v.mean(axis=0), color='dodgerblue', label='preds', linewidth=0.8)
    axs[1].plot(x, v.mean(axis=0) - v.std(axis=0), linestyle='--', color='dodgerblue', linewidth=0.8)
    axs[1].plot(x, v.mean(axis=0) + v.std(axis=0), linestyle='--', color='dodgerblue', linewidth=0.8)
    axs[1].fill_between(x, v.mean(axis=0) - v.std(axis=0), 
                        v.mean(axis=0) + v.std(axis=0), color='dodgerblue', 
                        alpha=0.5, linewidth=0.8)
    axs[1].plot(t, sol_v, color='black', linestyle='--', label='truth', linewidth=0.8)
    axs[1].set_title('Harmonic oscillator 1D variance') 
    axs[1].set_xlabel('time')
    axs[1].legend();


def make_our_pinn_stats_plot(x, bmeans, bstds, 
                             mean_trials, var_trials,
                             bmeans_pinn, bstds_pinn, name_to_save='stats_comparison.pdf'):
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(10, 3.9)

    axs[0].plot(x, bmeans, color='black', linestyle='--', label='truth', linewidth=0.8)
    axs[0].plot(x, mean_trials.mean(axis=0), color='dodgerblue', label='DSM', linewidth=0.8)
    axs[0].fill_between(x, mean_trials.mean(axis=0) - mean_trials.std(axis=0), 
                        mean_trials.mean(axis=0) + mean_trials.std(axis=0), color='dodgerblue', 
                        alpha=0.5, linewidth=0.8)
    axs[0].plot(x, bmeans_pinn, color='seagreen', label='PINN', linewidth=0.8)
    axs[0].set_ylabel('value')
    axs[0].set_xlabel('time')
    axs[0].legend();
    axs[0].set_title('$X_t$ mean')

    axs[1].plot(x, bstds, color='black', linestyle='--', label='truth', linewidth=0.8)
    axs[1].plot(x, var_trials.mean(axis=0), color='dodgerblue', label='DSM', linewidth=0.8)
    axs[1].fill_between(x, var_trials.mean(axis=0) - var_trials.std(axis=0), 
                        var_trials.mean(axis=0) + var_trials.std(axis=0), color='dodgerblue', 
                        alpha=0.5, linewidth=0.8)
    axs[1].plot(x, bstds_pinn, color='seagreen', label='PINN', linewidth=0.8)
    axs[1].set_title('$X_t$ variance') 
    axs[1].set_xlabel('time')
    axs[1].legend();

    # plt.savefig('pics/harm_osc_1d_stats1.pdf',bbox_inches='tight')

    plt.show();
    

