import numpy as np

def le_derivatives(xi, Y, n):
    """
    Taking the lane emden equation and reparametrising it to solve it better

    Reparametrisation:
        y_1 = theta(xi)
        y_2 = theta'(xi)
    
    Therefore, the derivatives are:
        y_1' = y_2
        y_2' = -(2/xi)y_2 - y_1^n

    Parameters
    ----------
    xi: float
        the dimensionless parameter xi defined in the L-E equation
    Y: tuple
        contains the values for y_1 and y_2
    n: float
        polytropic index

    Returns
    -------
    (y_1', y_2'): tuple
        the derivatives of the reparametrised functions
    """
    (y_1, y_2) = Y
    dy_1 = y_2
    dy_2 = (-(2/xi) * y_2) - (y_1 ** n)
    return dy_1, dy_2

def find_zero(xi, Y, n):
    """
    Function for the scipy.integrate solve_ivp event handler. Simply returns the first element of Y

    Parameters
    ----------
    xi: float
        the dimensionless parameter xi defined in the L-E equation
    Y: tuple
        contains the values for y_1 and y_2
    n: float
        polytropic index

    Returns
    -------
    float
        the first element of the tuple Y
    """
    return Y[0]

def le_analytical(xi, n):
    """
    Analytical solution to the Lane-Emden equation for n = 0, 1, 5

    Parameters
    ----------
    xi: float
        the dimensionless parameter xi defined in the L-E equation. Must be greater than 0
    n: float
        polytropic index

    Returns
    -------
    float
        the analytical solution to the L-E equation for the given n and xi
    """
    if n == 0:
        return 1 - (xi**2 / 6)
    elif n == 1:
        return np.sin(xi) / xi
    elif n == 5:
        return (1 + (xi**2 / 3))**(-0.5)