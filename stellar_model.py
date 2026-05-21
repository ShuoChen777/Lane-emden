# Here I'll define a class for the ideal gas and radiation star

from matplotlib.pylab import beta
import numpy as np
from scipy.integrate import solve_ivp, quad, cumulative_trapezoid
from le_fun import *
import astropy.constants as const
import astropy.units as u
import pandas as pd

class ESMStar():
    """
    An Eddington Standard Model Star. This will store the initial parameters of the star.
    The parameters will be dependent on radius in this class, and the user can initialise how the radius
    array will be distributed.

    All units in cgs.
    """

    def __init__(self, n, beta, mu, tot_radius, r_list = -1, t_span=[1e-10, 40]):
        """
        Initialise the star. 

        Parameters
        ----------
        n: float
            Polytrope number. Mostly will remain 3 or 3/2, depending on the object.
        beta: float
            Ratio between gas pressure to total pressure
        mu: float
            Represents mean molecular weight of star
        tot_radius: float
            Radius of star
        r_list: numpy array (optional)
            How the radius of the star is broken up into for the Lane-Emden solver.
        t_span: list (optional)
            The range of the independent variable xi for the Lane-Emden solver. Important for large n.

        Returns
        -------
        None
        """
        # Define astropy constants in cgs
        self._Msun = const.M_sun.cgs
        self._Rsun = const.R_sun.cgs
        self._G = const.G.cgs
        self._a = const.sigma_sb.cgs * (4/const.c.cgs)
        self._kB = const.k_B.cgs

        self.n = n
        self.beta = beta
        self.mu = mu
        self.R = tot_radius
        self._Mch = 50 * self._Msun * ((0.6 * const.m_p.cgs / mu) ** 2)
        self.M = self._Mch * ((1-self.beta)**0.5)/(self.beta**2) # Formula for total star mass.

        # Lane-Emden solution
        find_zero.terminal = True
        self._sol = solve_ivp(fun=le_derivatives, 
                              t_span=t_span, 
                              y0=(1+0j, 0+0j), 
                              dense_output=True, # Allows for interpolation of solution
                              max_step=1e-3, 
                              args=(n,), 
                              events=find_zero)
        self._xi_1 = self._sol.t_events[0][0] # The dimensionless radius where the solution first goes to zero, i.e. the surface of the star.

        # Arrays for radius and values of xi.
        if r_list == -1:
            self.radius = np.linspace(0, self.R.value, 1000)[1:-1] # Avoiding edge cases to prevent singularity in Lane-Emden solution.
        else:
            self.radius = np.array(r_list)[1:-1]
        self._xi = np.array((self.radius * self._xi_1) / self.R.value)
        K3 = (3/self._a) * ((self._kB**4)/(self.mu**4)) * (1-self.beta)/(self.beta**4)
        self._K = (K3**(1/3)).value

        # Central density
        if self.n == 3/2:
            self._rho_c = 5.99071 * (3 * self.M)/(4 * np.pi * self.R**3)
        elif self.n == 3:
            self._rho_c = 54.1825 * (3 * self.M)/(4 * np.pi * self.R**3)

        # Density, pressure, specific energy and temperature profiles
        self.density = np.array(self._get_density(self._xi))
        self.pressure = np.array(self._get_pressure(self._xi))
        self._Pg = np.array(self.pressure * self.beta) # Gas pressure
        self._Pr = np.array(self.pressure * (1 - self.beta)) # Rad pressure
        self._ug = np.array((3/2) * self._Pg / self.density) # Gas internal energy density
        self._ur = np.array(3 * self._Pr / self.density) # Rad internal energy density
        self.specificenergy = self._ug + self._ur
        self.temperature = self._get_temperature(self._xi)

        # Mass and dm profiles
        _dm = [0 for _ in self._xi]
        # dm
        for i in range(len(self._xi) - 1):
            _dm[i] = self._get_dm(self._xi[i], self._xi[i+1]).value

        # mass
        _mass = [sum(_dm[:i+1]) for i in range(len(_dm))]
        self.dm = np.array(_dm) * u.g
        self.mass = np.array(_mass) * u.g


    def _get_density(self, xi):
        """
        Calculate the density of a polytropic gas as a function of the dimensionless radius xi.

        p(xi) = p_c * theta(xi)^n

        Parameters
        ----------
        xi : float
            The dimensionless radius (dimensionless).

        Returns
        -------
        density: float
            The density of the gas in g/cm^3.
        """
        return self._rho_c * (self._sol.sol(xi)[0]**self.n).real
    
    def _get_pressure(self, xi):
        """
        Calculate the total pressure of a polytropic gas given its density for an n=3 polytrope.

        Parameters
        ----------
        xi: float
            The dimensionless radius parameter.

        Returns
        -------
        P: float
            The pressure of the gas in cgs.
        """
        return (self._K * self._get_density(xi)**(4/3)).value * (u.erg / u.cm**3)
    
    def _get_temperature(self, xi):
        """
        Calculate the temperature of a polytropic gas as a function of its density.

        Parameters
        ----------
        rho : float
            The density of the gas in g/cm^3.

        Returns
        -------
        T: float
            The temperature of the gas in K.
        """
        rho = self._get_density(xi).cgs
        T3 = ((1-self.beta)/self.beta) * (3 * rho * self._kB)/(self.mu * self._a)
        return (T3 ** (1/3)).to(u.K)
    
    def _dm_solution_integrand(self, xi):
        return (self.R/self._xi_1) * (4 * np.pi * ((xi * self.R/self._xi_1)**2) * self._get_density(xi)).value

    def _get_dm(self, xi_start, xi_end):
        """
        Calculate the mass of a shell of a polytropic gas between two dimensionless radii xi_start and xi_end.

        Parameters
        ----------
        xi_start : float
            The starting dimensionless radius (dimensionless).
        xi_end : float
            The ending dimensionless radius (dimensionless).

        Returns
        -------
        dm: float
            The mass of the shell in g.
        """
        return cumulative_trapezoid(self._dm_solution_integrand(np.linspace(xi_start, xi_end, 1000)),
                                    np.linspace(xi_start, xi_end, 1000))[-1] * u.g
    
    def make_data(self, output_path='./ESMStar_data.data'):
        """
        Make a pandas dataframe of the star's properties and save it as a csv.

        NOTE: AREPO requires the rows to be in reverse order, and labelled.
        i.e. row 1 would be the column labels, row 2 would be the surface of the star.

        Parameters
        ----------
        output_path: str (optional)
            The path to save the csv file to.

        Returns
        -------
        None
        """
        df = pd.DataFrame({
            'mass': self.mass[::-1],
            'pressure': self.pressure[::-1],
            'temperature': self.temperature[::-1],
            'radius': self.radius[::-1],
            'density': self.density[::-1],
            'specificenergy': self.specificenergy[::-1],
            'dm': self.dm[::-1],
        })
        df.to_csv(output_path, sep='\t', index=False)


