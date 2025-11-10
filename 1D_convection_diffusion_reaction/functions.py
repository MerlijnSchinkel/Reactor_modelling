from scipy.sparse import diags
import numpy as np


def newton_raphson(u, jac, jac_bc, dt):
    """
    The Newton-Raphson approach to solve linearized partial differential equations.\n
    :param u: State-variable (vector)
    :param jac: Jacobian (matrix)
    :param jac_bc: Constant contribution to Jacobian (vector)
    :param dt: Timestep duration
    :return: u (vector), it, error_f, error_x
    """

    # Define tolerances and maximum number of iterations
    tol_x = 1e-8
    tol_f = 1e-8
    max_it = 20

    # Set errors and number of iterations
    error_x = 2*tol_x
    error_f = 2*tol_f
    it = 0

    # Save old state-variable
    u_old = u

    # Start Newton-Raphson loop
    while (error_x > tol_x) and (error_f > tol_f) and (it < max_it):
        # Function that converges to zero during Newton-Raphson loop
        g = (u - u_old)/dt + (jac - np.identity(len(u))/dt).dot(u) + jac_bc

        # Calculate step
        du = np.linalg.solve(jac, -g)

        # Update state variable
        u = u + du

        # Calculate errors and increase number of iterations
        error_f = max(g)
        error_x = max(du)
        it += 1

    return u, it, error_f, error_x


def jac_react(func, u):
    """
    A function used to numerically calculate the Jacobian of reaction systems.\n
    :param func: A function implementing the reaction kinetics
    :param u: State-variable (vector)
    :return: jac (matrix)
    """

    # Small step used to numerically calculate derivatives
    eps = 1e-8

    # Calculate Jacobian
    jac = (func(u+eps) - func(u)) / eps

    return jac


def jac_conv_diff_1d(x_f, v_f, diff_f, u_in):
    # Determine number of cells
    nx = len(x_f) - 1
    nc = len(u_in)

    # Create lower, main, and upper diagonal arrays
    lower_diagonal = np.zeros(nx*nc-1)
    main_diagonal = np.zeros(nx*nc)
    upper_diagonal = np.zeros(nx*nc-1)

    # Constant contribution
    jac_bc = np.zeros(nc*(len(x_f)-1))

    # Loop over each component and create Jacobian matrix
    for i in range(nc):
        # Internal cells
        lower_diagonal[i*nx:(i+1)*nx-1] = -diff_f[i][1:-1] / (x_f[2:] - x_f[1:-1])**2 - v_f[i][1:-1] / (x_f[2:] - x_f[1:-1])
        main_diagonal[i*nx:(i+1)*nx] = 2*diff_f[i][1:] / (x_f[1:] - x_f[0:-1])**2 + v_f[i][1:] / (x_f[1:] - x_f[0:-1])
        upper_diagonal[i*nx:(i+1)*nx-1] = -diff_f[i][1:-1] / (x_f[2:] - x_f[1:-1])**2

        # Add inlet cell contribution
        main_diagonal[i*nx] = 2*diff_f[i][0]*u_in[i] / (x_f[1] - x_f[0]) + 2*v_f[i][0] / (x_f[1] - x_f[0])
        
        # Add constant contribution to apply boundary conditions
        jac_bc[i*nx] = -diff_f[i][0]*u_in[i] / (x_f[1] - x_f[0]) - 2*v_f[i][0]*u_in[i] / (x_f[1] - x_f[0])

    # Add lower and main diagonal together to create the Jacobian matrix
    jac = diags([lower_diagonal, main_diagonal, upper_diagonal], [-1, 0, 1]).toarray()

    return jac, jac_bc
