import matplotlib.pyplot as plt
from functions import *
import numpy as np


# General parameters
L = 1
Nx = 100
Nt = 1250
dt = 1e-3

# Cell face and center coordinate vectors [m]
x_f = np.linspace(0, L, Nx+1)
x_c = (x_f[1:] + x_f[0:-1]) / 2

# Velocity and diffusion coefficient
v_f = [1*np.ones(Nx+1)]
D_f = [1e-6*np.ones(Nx+1)]
k = 1
n = 2

# Inlet concentration [mol/m3]
C_in = [1]
C = np.zeros(Nx)
C_tot = np.zeros([Nt+1, Nx])
C_tot[0] = C

# Reaction kinetics
def f(u_array):
    rate = np.zeros([Nx, Nx])
    r1 = k*u_array**n * np.identity(Nx)
    rate = r1
    return rate


# Calculate Jacobian
jac, jac_bc = jac_conv_diff_1d(x_f, v_f, D_f, C_in)


# Newton Raphson loop
for i in range(1, Nt+1):
    Jac = jac + jac_react(f, C) + np.identity(Nx)/dt

    [C, it, error_f, error_x] = newton_raphson(C, Jac, jac_bc, dt)
    C_tot[i] = C

    print(f"{round(i/Nt * 100,1)}% completed")



# Plot results
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(x_c, C_tot[0], label='A', color='blue')
ax.set_ylim(0, 1.05)

for i in range(Nt+1):
    line.set_ydata(C_tot[i])

    fig.canvas.flush_events()
    fig.canvas.draw()

    print(f"{i}/{Nt}")

plt.show()
