import numpy as np
from numpy import array, exp, linspace, pi, sqrt, inf, concatenate, zeros, einsum, eye, diag, conjugate, copy, trace
from numpy.linalg import eig, eigvals, inv, norm
from numpy.random import random

from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.special import erf

from itertools import product

import warnings


def F0(t):
    if t == 0:
        return 1
    else:
        return 0.5 * sqrt(pi/t) * erf(t**0.5)


class GaussianBasisFunction:

    def __init__(self, x: float, y: float, z: float, d: float, alpha: float) -> None:
        
        self.x = x
        self.y = y
        self.z = z
        self.Rc = array([x, y, z])

        self.d = d
        self.a = alpha

        self.phi_func = lambda r: self.d * (2 * self.a/pi)**(0.75) * exp(- self.a * norm(r - self.Rc)**2)
    
    def __call__(self, r: array) -> float:
        # assert len(r) == 3, 'wrong length of r'
        return self.phi_func(r=r)

    def __mul__(self, other):

        
        if isinstance(other, GaussianBasisFunction):
            
            alpha = self.a
            beta = other.a
            
            p = alpha + beta
            Rp = (alpha * self.Rc + beta * other.Rc)/(alpha + beta)
            xp, yp, zp = Rp
            rp = self.Rc - other.Rc

            K_AB = (2 * alpha * beta/ ((alpha + beta) * pi))**0.75 * exp(- alpha * beta/(alpha + beta) * norm(rp)**2)
            d_p = K_AB * self.d * other.d

            return GaussianBasisFunction(x = xp, y = yp, z = zp, d = d_p, alpha=p)

        
        elif isinstance(other, float) or isinstance(other, int):
            
            d_p = self.d * other
            
            return GaussianBasisFunction(x = self.x, y = self.y, z = self.z, d = d_p, alpha=self.a)

        else:
            raise Exception()

    def __rmul__(self, other):
        return self.__mul__(other=other)
    
    def __truediv__(self, other):

        if isinstance(other, int) or isinstance(other, float):
            d_p = self.d/other
            return GaussianBasisFunction(x = self.x, y = self.y, z = self.z, d = d_p, alpha=self.a)
        
        else:
            raise Exception()

    
    # The round brackets
    @staticmethod
    def RB_2(GF1, GF2):
        
        assert isinstance(GF1, GaussianBasisFunction) and isinstance(GF2, GaussianBasisFunction), 'input must be GTF'
        
        alpha = GF1.a
        beta = GF2.a
        r_N = norm(GF1.Rc - GF2.Rc)

        N1 = (2 * alpha/pi)**0.75
        N2 = (2 * beta/pi)**0.75

        res = (pi/(alpha + beta))**1.5
        res *= exp(- alpha * beta/(alpha + beta) * r_N**2)
        res = res * GF1.d * GF2.d * N1 * N2
        
        return res

    @staticmethod
    def RB_4(GF1, GF2, GF3, GF4):
        
        assert isinstance(GF1, GaussianBasisFunction) and isinstance(GF2, GaussianBasisFunction) and isinstance(GF3, GaussianBasisFunction) and isinstance(GF4, GaussianBasisFunction), 'input must be GTF'
        
        alpha = GF1.a
        beta = GF2.a
        gamma = GF3.a
        delta = GF4.a

        N1 = (2 * alpha/pi)**0.75
        N2 = (2 * beta/pi)**0.75
        N3 = (2 * gamma/pi)**0.75
        N4 = (2 * delta/pi)**0.75

        A = alpha * beta /(alpha + beta)
        B = gamma * delta /(gamma + delta)
        C = (alpha + beta) * (gamma + delta)/(alpha + beta + gamma + delta)
        
        r_N1 = norm(GF1.Rc - GF2.Rc)
        r_N2 = norm(GF3.Rc - GF4.Rc)

        r_P = (alpha * GF1.Rc + beta * GF2.Rc)/(alpha + beta)
        r_Q = (gamma * GF3.Rc + delta * GF4.Rc)/(gamma + delta)
        r_NPQ = norm(r_P - r_Q)

        res = (2 * pi**2.5)/((alpha + beta)*(gamma + delta)*sqrt(alpha + beta + gamma + delta))
        res *= exp( - A * r_N1**2 - B * r_N2**2)
        res *= F0(C * r_NPQ**2)

        res = res * GF1.d * GF2.d * GF3.d * GF4.d
        res = res * N1 * N2 * N3 * N4

        return res
    
    @staticmethod
    def RB_T(GF1, GF2):
        
        assert isinstance(GF1, GaussianBasisFunction) and isinstance(GF2, GaussianBasisFunction), 'input must be GTF'
        
        alpha = GF1.a
        beta = GF2.a
        r_N = norm(GF1.Rc - GF2.Rc)

        N1 = (2 * alpha/pi)**0.75
        N2 = (2 * beta/pi)**0.75

        res = alpha * beta/(alpha + beta)
        res *= (3 - 2 * alpha * beta/(alpha + beta) * r_N**2)
        res *= (pi/(alpha + beta))**1.5
        res *= exp(- alpha * beta/(alpha + beta) * r_N**2)
        res = res * GF1.d * GF2.d * N1 * N2

        return res
    
    @staticmethod
    def RB_V(GF1, GF2, Zc, rc):

        assert isinstance(GF1, GaussianBasisFunction) and isinstance(GF2, GaussianBasisFunction), 'input must be GTF'

        alpha = GF1.a
        beta = GF2.a

        N1 = (2 * alpha/pi)**0.75
        N2 = (2 * beta/pi)**0.75

        r_N = norm(GF1.Rc - GF2.Rc)
        rp = (alpha * GF1.Rc + beta * GF2.Rc)/(alpha + beta)
        
        res = -2 * pi/(alpha + beta) * Zc
        res *= exp(- alpha * beta/(alpha + beta) * r_N**2)
        res *= F0((alpha + beta)* norm(rc - rp)**2)

        res = res * GF1.d * GF2.d * N1 * N2

        return res


    def __repr__(self) -> str:
        return f'{self.d:.3f} * G({self.a:.3f}; {self.x}, {self.y}, {self.z})'
    
    def __str__(self) -> str:
        return self.__repr__()



class HartreeFock:

    Z_dict = {'H': 1, 'He': 2}

    def __init__(self, molecule_geometry, zeta_dict: dict, SCF_iteration: int = 10, show_iter_info: bool = False) -> None:

        """
        zeta_dict: {atom_name: zeta value}
        molecule_geometry: [atom_name, array([x, y, z])]
        self.zeta_sto3g_param_dict: {atom_name: {d1, d2, d3, a1, a2, a3}}
        """
        
        self.molecule_geometry = molecule_geometry
        
        self.zeta_dict = zeta_dict
        self.zeta_sto3g_param_dict = {name: self.get_sto3g_parameter(zeta=Zc) for name, Zc in self.zeta_dict.items()}
        self.atom_wavefunc_dict = {
                i: {
                    1: GaussianBasisFunction(x = x, y = y, z = z, d = self.zeta_sto3g_param_dict[name]['d1'], alpha = self.zeta_sto3g_param_dict[name]['a1']),
                    2: GaussianBasisFunction(x = x, y = y, z = z, d = self.zeta_sto3g_param_dict[name]['d2'], alpha = self.zeta_sto3g_param_dict[name]['a2']),
                    3: GaussianBasisFunction(x = x, y = y, z = z, d = self.zeta_sto3g_param_dict[name]['d3'], alpha = self.zeta_sto3g_param_dict[name]['a3'])
                    }
                    for i, [name, [x, y, z]] in enumerate(self.molecule_geometry)
            }

        self.S, self.T, self.V = self.compute_STV()
        self.Hcore = self.T + sum(self.V)
        self.RB_4_tensor = self.get_RB4()

        self.eps, self.C, self.F, self.G, self.P, self.E0 = self.SCF_single_iteration()

        self.psi_RB_h = self.get_psi_RB_h()
        self.psi_RB_4_tensor = self.get_psi_RB_4()
        
        self.psi_AB_h = copy(self.psi_RB_h)
        self.psi_AB_4_tensor = self.get_psi_AB_4()

        if SCF_iteration > 0:
            self.SCF_loop(n_iter=SCF_iteration, show_iter_info=show_iter_info)


    @staticmethod
    def get_sto3g_parameter(zeta: float):
        

        def target_function(parameter):
            
            c1, c2, c3, a1, a2, a3 = parameter
            integrand = lambda r: r**2 * (c1 * exp(- a1 * r**2) + c2 * exp(- a2 * r**2) + c3 * exp(- a3 * r**2) - sqrt(zeta**3/pi) * exp(- zeta * r))**2
            
            return quad(integrand, 0, inf)[0]


        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            optimized_param = minimize(fun = target_function, x0 = random(6)).x

        c_vec = optimized_param[:3]
        
        a_vec = optimized_param[3:]
        a1, a2, a3 = a_vec

        d_vec = c_vec * (2 * a_vec/pi)**(-0.75)
        d1, d2, d3 = d_vec

        optimized_param_dict = {'d1': d1, 'd2': d2, 'd3': d3, 'a1': a1, 'a2': a2, 'a3': a3}

        return optimized_param_dict
    
    
    def compute_STV(self) -> array:
        
        n = len(self.atom_wavefunc_dict)
        S = zeros((n, n))
        T = zeros((n, n))
        V = [zeros((n, n)) for _ in range(len(self.molecule_geometry))]
        
        RB = lambda f1, f2: GaussianBasisFunction.RB_2(f1, f2)
        RB_T = lambda f1, f2: GaussianBasisFunction.RB_T(f1, f2)
        RB_V = lambda f1, f2, Zc, Rc: GaussianBasisFunction.RB_V(f1, f2, Zc, Rc)

        for i, phi1 in self.atom_wavefunc_dict.items():
            for j, phi2 in self.atom_wavefunc_dict.items():
                
                S[i, j] +=  RB(phi1[1], phi2[1]) + RB(phi1[1], phi2[2]) + RB(phi1[1], phi2[3])
                S[i, j] +=  RB(phi1[2], phi2[1]) + RB(phi1[2], phi2[2]) + RB(phi1[2], phi2[3])
                S[i, j] +=  RB(phi1[3], phi2[1]) + RB(phi1[3], phi2[2]) + RB(phi1[3], phi2[3])

                T[i, j] +=  RB_T(phi1[1], phi2[1]) + RB_T(phi1[1], phi2[2]) + RB_T(phi1[1], phi2[3])
                T[i, j] +=  RB_T(phi1[2], phi2[1]) + RB_T(phi1[2], phi2[2]) + RB_T(phi1[2], phi2[3])
                T[i, j] +=  RB_T(phi1[3], phi2[1]) + RB_T(phi1[3], phi2[2]) + RB_T(phi1[3], phi2[3])

                for a, atom in enumerate(self.molecule_geometry):

                    name = atom[0]
                    Zc = self.Z_dict[name]
                    Rc = array(atom[1])

                    V[a][i, j] += RB_V(phi1[1], phi2[1], Zc, Rc) + RB_V(phi1[1], phi2[2], Zc, Rc) + RB_V(phi1[1], phi2[3], Zc, Rc)
                    V[a][i, j] += RB_V(phi1[2], phi2[1], Zc, Rc) + RB_V(phi1[2], phi2[2], Zc, Rc) + RB_V(phi1[2], phi2[3], Zc, Rc)
                    V[a][i, j] += RB_V(phi1[3], phi2[1], Zc, Rc) + RB_V(phi1[3], phi2[2], Zc, Rc) + RB_V(phi1[3], phi2[3], Zc, Rc)

 
        return S, T, V

    
    def get_RB4(self) -> array:

        def sto3g_RB4(phi0: dict, phi1: dict, phi2: dict, phi3: dict):
            res = [GaussianBasisFunction.RB_4(g0, g1, g2, g3) for g0, g1, g2, g3 in product(phi0.values(), phi1.values(), phi2.values(), phi3.values())]            
            return sum(res)

        n = len(self.atom_wavefunc_dict)
        RB4_dict = np.zeros((n, n, n, n))
        for (i0, phi0), (i1, phi1), (i2, phi2), (i3, phi3) in product(*([self.atom_wavefunc_dict.items()]*4)):
            RB4_dict[i0, i1, i2, i3] = sto3g_RB4(phi0, phi1, phi2, phi3)

        return RB4_dict


    def SCF_single_iteration(self, P0 = None):
        
        n = len(self.atom_wavefunc_dict)

        if P0 is None:
            P0 = zeros((n, n))
            G = zeros((n, n))
        
        else:
            G = zeros((n, n))
            G += einsum('kl,ijlk->ij', P0, self.RB_4_tensor)
            G -= einsum('kl,iklj->ij', P0, self.RB_4_tensor)/2
        
        F = self.Hcore + G
        S_eig_val, S_U = eig(self.S)
        X = S_U/sqrt(S_eig_val)        
        Fp = X.T @ F @ X 
        _, Cp = eig(Fp)
        C = X @ Cp
        eps = inv(Cp) @ Fp @ Cp

        arg_c = diag(eps).argsort()[:(n//2)]

        P = 2 * C[:, arg_c] @ C[:, arg_c].T

        E0 = 0.5 * trace((F + self.Hcore) @ P)

        return eps.round(10), C, F, G, P, E0
    

    def SCF_loop(self, P0 = None, n_iter = 10, show_iter_info = False):

        if P0 is None:
            P0 = self.P

        cnt = 0
        loop_str = f'cnt = {cnt}, P00 = {self.P[0, 0]:.4f}, P01 = {self.P[0, 1]:.4f}, P11 = {self.P[1, 1]:.4f}, E0 = {self.E0:.4f}'
        
        if show_iter_info:
            print(loop_str)

        while cnt < n_iter:
            self.eps, self.C, self.F, self.G, self.P, self.E0 = self.SCF_single_iteration(P0 = self.P)
            
            self.psi_RB_h = self.get_psi_RB_h()
            self.psi_AB_h = copy(self.psi_RB_h)
            self.psi_RB_4_tensor = self.get_psi_RB_4()
            self.psi_AB_4_tensor = self.get_psi_AB_4()
            cnt += 1
            loop_str = f'cnt = {cnt}, P00 = {self.P[0, 0]:.4f}, P01 = {self.P[0, 1]:.4f}, P11 = {self.P[1, 1]:.4f}, E0 = {self.E0:.4f}'
            
            if show_iter_info:
                print(loop_str)


    def get_psi_RB_h(self):
        RB_h = self.C.T @ self.Hcore @ self.C
        return RB_h


    def get_psi_RB_4(self):
        RB_4 = einsum('ij,kl,mn,op,ikmo->jlnp', self.C, self.C, self.C, self.C, self.RB_4_tensor)
        return RB_4


    def get_psi_AB_4(self):
        
        t_shape = self.RB_4_tensor.shape
        AB_4 = zeros(t_shape)
        
        for a, b, c, d in product(*map(range, t_shape)):
            AB_4[a, b, c, d] = self.psi_RB_4_tensor[a, c, b, d]
        
        return AB_4




if __name__ == '__main__':

    from numpy.random import random, rand
    
    # hf = HartreeFock(molecule_geometry = [['H', [0, 0, 0]], ['H', [0, 0, 1.4]]], zeta_dict={'H': 1.24})
    hf = HartreeFock(molecule_geometry = [['H', [0, 0, 0]], ['He', [0, 0, 1.4632/2]]], zeta_dict={'H': 1.24, 'He': 2.0925}, SCF_iteration=0, show_iter_info=True)
    # print(hf.atom_wavefunc_dict)
    # phi1 = hf.atom_wavefunc_dict[0][1]
    # phi2 = hf.atom_wavefunc_dict[0][1]
    # print(GaussianBasisFunction.RB_2(phi1, phi2))
    
    print('--------------------------------------------------')
    print('S')
    print('--------------------------------------------------')
    print(hf.S)
    
    print('--------------------------------------------------')
    print('H')
    print('--------------------------------------------------')
    print(hf.Hcore)
    
    print('--------------------------------------------------')
    print('RB4')
    print('--------------------------------------------------')
    print(hf.RB_4_tensor[0, 0, 0, 0], hf.RB_4_tensor[1, 1, 1, 1])
    print(hf.RB_4_tensor[0, 0, 1, 1])
    print(hf.RB_4_tensor[1, 0, 0, 0], hf.RB_4_tensor[1, 1, 1, 0])
    print(hf.RB_4_tensor[1, 0, 1, 0])
    
    print('--------------------------------------------------')
    print('Ground State Energy')
    print('--------------------------------------------------')
    print(hf.eps)

    print('--------------------------------------------------')
    print('Combination Coefficient')
    print('--------------------------------------------------')
    print(hf.C)
    
    print('--------------------------------------------------')
    print('Run SCF iterations')
    print('--------------------------------------------------')
    hf.SCF_loop(n_iter=10, show_iter_info=True)
    print('--------------------------------------------------')
    print('Final Combination Coefficient')
    print('--------------------------------------------------')
    print(hf.C)
    print('--------------------------------------------------')
    print('Final F')
    print('--------------------------------------------------')
    print(hf.F)
    print('--------------------------------------------------')
    print('Final eps')
    print('--------------------------------------------------')
    print(hf.eps)
    