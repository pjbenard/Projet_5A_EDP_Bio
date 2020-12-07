from math import pi

import numpy as np
import numpy.linalg as npl

import scipy as sp
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as sspl

import matplotlib.pyplot as plt

import holoviews as hv
import param
import panel as pn

hv.extension('bokeh')
import bokeh.plotting as bkplt

from tqdm import tqdm


class model():
    
    def __init__(self, N, a, b, dt, T, sigma = 0.5, mu = 5, kappa = 1, r_max = 1,):
        self.N = N
        self.a, self.b = a, b
        self.dt, self.T = dt, T
        self.dz = (b - a) / N
        self.Z = np.linspace(a ,b, N)
        self.sigma = sigma
        
        self.A = self.__get_A__()
        self.nbT = int(T / dt)
        self.mu = mu
        self.kappa = kappa
        self.r_max = r_max
        self.source_types = {'normal': source_normal, 'uniform': source_uniform}
        self.solvers = {"solve_implicit": self.solve_implicit, "solve_explicit": self.solve_explicit, "solve_splitting": self.solve_splitting}

    def solve(self, solver, theta_args, s_args, source_args, discret_args=None, disable_tqdm=False):
        if discret_args is None:
            dt = self.dt
            nbT = self.nbT
            dz = self.dz
            N = self.N
            Z = self.Z
            A = self.A
        else:
            T = discret_args["T"]
            dt = discret_args["dt"]
            a, b, N = discret_args["a"], discret_args["b"], discret_args["N"]
            dz = (b - a) / N
            nbT = int(T / dt)
            A = self.__get_A__(dz, N)
            Z = np.linspace(a, b, N)
        self.s_args = s_args
        self.theta_args = theta_args
        self.source_args = source_args
        U = np.empty(N)
    
        # Initialisation of U_0
        source_type = source_args.get('source_type', 'normal')
        source = self.source_types[source_type]
        solver = self.solvers[solver]
        U[1:-1] = np.copy(source(Z[1:-1], source_args['source_mu'], source_args['source_std']))
        U[0] = U[1]
        U[-1] = U[-2]

        # Arrays that saves the state of the system through time
        Us = np.empty((nbT + 1, N))
        RHO = np.empty(nbT + 1)

        THETA = get_theta(nbT, **theta_args)
        S = get_s(nbT, **s_args)
        
        for i in tqdm(range(0, nbT + 1), leave=True, disable=disable_tqdm):
            rho = self.__get_rho__(U)

            Us[i] = np.copy(U)

            U[1:-1] = solver(A, U[1:-1],  Z[1:-1], rho, dz, dt, THETA[i], S[i])
            U[0] = U[1]
            U[-1] = U[-2]
            
        self.THETA = THETA
        self.S = S
        self.Us = np.array(Us)
        self.RHO = self.__get_rho__(self.Us)
        MEAN = mean(self.Z, self.Us)
        self.MU = MEAN 
        VAR = var(self.Z, self.Us)
        self.VAR = VAR
        
    def plot_2D(self, dynamic=True, log_plot=False, nb_images=50):
        Z = self.Z
        Us = self.Us
        dt = self.dt
        T  = self.T
        time_step = Us.shape[0] // nb_images
        THETA = self.THETA
        if log_plot:
            range_plot = np.unique(np.logspace(start=0, stop=np.log10(Us.shape[0] - 1), num=nb_images, endpoint=True, dtype=int))
            range_plot = np.concatenate(([0], range_plot))
        else: 
            range_plot = np.linspace(start=0, stop=Us.shape[0] - 1, num=nb_images, endpoint=True, dtype=int)

        if THETA is not None and dynamic:
            curve_dict = {round(dt * i, ndigits=2): hv.Curve((Z, Us[i])) * hv.VLine(THETA[i]).opts(title = "Population density",xlabel = "Phenotypic state",color='orange') for i in range_plot}
        else:
            curve_dict = {round(dt * i, ndigits=2): hv.Curve((Z, Us[i])).opts(title =f"Population density", xlabel = "Phenotypic state") for i in range_plot}

        if dynamic:
    #         curve_dict = {i: hv.Curve((Z, Us[i])).opts(title=f'Approximated solution at t = {dt * i:.1f}') for i in range(0, Us.shape[0], time_step)}
            plot = hv.HoloMap(curve_dict, kdims='Time')

        else:
            plot = hv.NdOverlay(curve_dict, kdims='Time').opts(legend_position='right')

        return hv.output(plot.opts(width=600, toolbar=None))
    
    def plot_3D(self):
        time = np.linspace(0, self.T,self.nbT +1)
        Z = self.Z

        options = {'title': 'Population density', 
                   'xlabel': 'Phenotypic state', 'ylabel': 'Time', 
                   'colorbar': True, 'toolbar': None, 'width': 350}
        
        plot = hv.Image((Z, time, self.Us))
        
        return hv.output(plot.opts(cmap='viridis', **options))
        
        
    def plot_params(self):
        RHO = self.RHO
        MEAN = mean(self.Z, self.Us)
        VAR = var(self.Z, self.Us)
        THETA = self.THETA
        self.VAR = VAR
        self.RHO = RHO
        self.MU = MEAN        
        time = np.linspace(0, self.T,self.nbT +1)
        options = {'shared_axes': False, 'toolbar': None}

        curves = []

        curves += [hv.NdOverlay({theta_name: hv.Curve((time, VALUE), 'Iteration', 'Theta').opts(alpha=alpha) for VALUE, theta_name, alpha in zip(
            [MEAN, THETA], ['Numerical', 'Optimal'], [1., .5]
        )}, kdims='Thetas').opts(title='Mean phenotypic state')]

        curves += [hv.Curve((time, VALUE), 'Time', val_name.capitalize()).opts(title=f'{val_name}') for VALUE, val_name in zip(
            [VAR, RHO], ['Phenotypic state variance', 'Population']
        )]

        return hv.output(hv.Layout(curves).opts(width=900, **options))
    
    def plot_moments(self):
        RHO = self.RHO
        MEAN = mean(self.Z, self.Us)
        VAR = var(self.Z, self.Us)
        THETA = self.THETA
        
        dVAR = dvar(VAR, self.S, self.sigma)
        dMU = dmu(MEAN,VAR,THETA,self.S)
        dRHO = drho(RHO, MEAN, VAR, self.S, THETA, self.sigma, self.kappa, self.r_max)
        
        self.dVAR = dVAR
        self.dMU = dMU
        self.dRHO = dRHO
        
        options = {'shared_axes': False, 'toolbar': None}

        curves = []

        curves += [hv.NdOverlay({theta_name: hv.Curve(VALUE, 'Iteration', 'Theta').opts(alpha=alpha) for VALUE, theta_name, alpha in zip(
            [dMU], ['Current'], [1., .5]
        )}, kdims='Thetas').opts(title='Evolution of moment of MU')]

        curves += [hv.Curve(VALUE, 'Iteration', val_name.capitalize()).opts(title=f'Evolution of moment of {val_name}') for VALUE, val_name in zip(
            [dVAR, dRHO], ['variance', 'population']
        )]

        return hv.output(hv.Layout(curves).opts(width=900, **options))
    
    def __get_A__(self):
        N = self.N - 2
        tmp = np.ones((N - 1))
        diag = -2 * np.ones(N)
        diag[0] /= 2
        diag[-1] /= 2
        return sps.diags((tmp, diag, tmp),(-1, 0, 1)) * self.sigma**2 / (self.dz**2)
    
    
    def __get_r__(self,S,THETA,Z):
        return self.r_max - S * (Z - THETA)**2

    def __get_rho__(self, U):
        return self.dz * ((U[..., 0] + U[..., -1]) / 2 + np.sum(U[..., 1:-1], axis=-1))
    
    def F(self, U, r, rho):
        return U * (r - self.kappa * rho)
    
    def solve_explicit(self,A, U, Z, rho, dz, dt, theta, s):
        r = self.__get_r__(s,theta,Z)
        tmp1 = sps.eye(A.shape[0])
        tmp2 = dt * A
        tmp3 = dt * self.F(U, r, rho)
        U_next = (tmp1 + tmp2).dot(U) + tmp3
        return U_next

    def solve_implicit(self,A, U, Z, rho, dz, dt, theta, s):
        r = self.__get_r__(s,theta,Z)

        lhs = (np.eye(A.shape[0]) - dt * A)
        rhs = U + dt * self.F(U, r, rho)
        return sspl.cg(lhs, rhs)[0]

    def solve_splitting(self,A, U, Z, rho, dz, dt, theta, s):
        r = self.__get_r__(s,theta,Z)

        tempU1 = sspl.cg(np.eye(A.shape[0]) - dt / 2 * A, U)[0]
        tempU2 = tempU1 + dt * self.F(tempU1, r, rho)
        return sspl.cg(np.eye(A.shape[0]) - dt / 2 * A, tempU2)[0]
    
    
def source_normal(Z, mu=5, sigma=0.5):    
    return (.5 / (2 * pi * sigma**2)**.5) * np.exp(-0.5 * (Z - mu)**2 / sigma**2)

def source_uniform(Z, mid=5, half_dist=0.5):
    height = 1 / (2 * half_dist)
    U = np.where(np.abs(Z - mid) < half_dist, 0, height / 2)

    return U

def mean(Z, U):
    return np.sum(U * Z, axis=-1) / np.sum(U, axis=-1)

def var(Z, U):
    mu = mean(Z, U)
    return np.sum(U * Z**2, axis=-1) / np.sum(U, axis=-1) - mu**2 

def dvar(VAR, S, sigma):
    return VAR / (2 * S) - (VAR**3) / (2 * sigma)

def dmu(MU,VAR,THETA,S):
    return 2 * S * VAR * (THETA - MU)

def drho(RHO, MU, VAR, S, THETA, sigma, kappa, r):
    f = 1./VAR
    return RHO * ((sigma / f) * (f - 1) + S * MU * (THETA - MU) - S * THETA**2 + r - kappa * RHO)

def get_noise(nbT, noise_type='normal', noise_std=.5, accumulate = False):
    if  noise_type is 'normal':
        NOISE = np.random.normal(loc=0., scale=noise_std, size=nbT + 1)
    elif noise_type is 'uniform':
        NOISE = np.random.uniform(low=-noise_std, high=noise_std, size=nbT + 1)
    elif noise_type is 'discrete':
        noise_values = np.linspace(start=-noise_std, stop=noise_std, num=5)
        NOISE = np.random.choice(noise_values, size=nbT + 1)
        
    if accumulate:
        NOISE = np.cumsum(NOISE)
    
    return NOISE

def get_theta(nbT, **theta_args):
    THETA = np.ones(nbT + 1) * theta_args['theta_value']
    accum = theta_args['accumulative']
    if theta_args['theta_moving']:
        MOVING = np.arange(nbT + 1) * theta_args['theta_speed']
        THETA += MOVING
    
    if theta_args['theta_noise']:
        noise_type = theta_args.get('theta_noise_type', 'normal')
        noise_std = theta_args.get('theta_noise_std', 0.5)
        
        NOISE = get_noise(nbT, noise_type, noise_std, accum)
        THETA += NOISE
        
    return THETA

def get_s(nbT, **s_args):
    S = np.ones(nbT + 1) * s_args['s_value']
    
    if s_args['s_noise']:
        noise_type = s_args.get('s_noise_type', 'normal')
        noise_std = s_args.get('s_noise_std', 0.5)
        
        NOISE = get_noise(nbT, noise_type, noise_std)
        S += NOISE
        
    return np.clip(a=S, a_min=1e-3, a_max=None)    