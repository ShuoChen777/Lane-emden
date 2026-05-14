import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt

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
    

def getpoly(n,N=int(1e4), G=1, imax=1e4,tolerance = 1.e-19,show=False): 
    ''' Getpoly(n,,N=1e4, G=1, imax=1e4,tolerance = 1.e-19) returns (r,rho,m,P) for a self-consistent polytrope 
    with total mass and radius M=R=1.  Scale according to rho ~ M/R^3 and P ~ GM^2/R^4. ''' 
    r = np.linspace(1.e-10/N,1,N) 
    rho = r.copy()*0.+4*np.pi/3;  
    rhoprev = rho +0.1
    
    keepgoing = 1 
    i = 0 
    while keepgoing: 
        m = cumtrapz(4*np.pi*r**2*rho,r,initial=0) # Could switch to Simpson interator for even greater accuracy

        mmax = np.max(m) ## normalize density and mass to unit mass. 
        rho = rho/mmax
        m = m/mmax

        w = ~np.isnan(rho/rhoprev) # necessary because the last value of rho is zero. 
        error = np.max(np.abs(rho[w]/rhoprev[w]-1))
        keepgoing = (error > tolerance)*(i<imax)

        if show: 
            plt.figure(2)
            if i: plt.plot(r,rhoprev,'r',r,rho,'k')

        g = -G*m/r**2
        Phi = cumtrapz(-g, r,initial=0)
        rhoprev = rho 
        rho = (np.max(Phi)-Phi)**n
        i +=1
    m = cumtrapz(4*np.pi*r**2*rho,r,initial=0) # Could switch to Simpson interator for even greater accuracy
    mmax = np.max(m) ## normalize density and mass to unit mass. 
    rho = rho/mmax
    m = m/mmax

    P = rho**(1.+1./n) ## un-normalized pressure profile. 
    # Normalize the pressure: 
    Egrav = np.trapezoid(G*m/r * rho *4*np.pi*r**2,r)
    TwoEtherm = 3*np.trapezoid(P*4*np.pi*r**2, r)
    P = P*Egrav/TwoEtherm ## enforce virial equilibrium: |E_grav| = 2 E_therm 
    
    return r,rho,m, P