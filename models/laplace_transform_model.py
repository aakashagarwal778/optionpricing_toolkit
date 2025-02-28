import cmath
import math
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d

class LaplaceTransformModel:
    def laplace_BS(self, S0, r, sigma, T, K, R, option_type='call'):
        def f_tilde(z):
            if option_type == 'call':
                return cmath.exp((1 - z) * math.log(K)) / (z * (z - 1))
            elif option_type == 'put':
                return cmath.exp((1 - z) * math.log(K)) / (z * (z - 1))

        def chi(u):
            return cmath.exp(complex(0, 1) * u * (math.log(S0) + r * T) - (complex(0, 1) * u + cmath.exp(2 * cmath.log(u))) * math.pow(sigma, 2) / 2 * T)

        def integrand(u):
            return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

        V0 = integrate.quad(integrand, 0, 50)
        return V0

    def laplace_heston(self, S0, r, gam0, kappa, lamb, sig_tild, T, K, R, p):
        def f_tilde(z):
            return p * cmath.exp((1 - z / p) * math.log(K)) / (z * (z - p))

        def chi(u):
            return self.heston_char(u, S0, r, gam0, kappa, lamb, sig_tild, T)

        def integrand(u):
            return math.exp(-r * T) / math.pi * (f_tilde(R + complex(0, 1) * u) * chi(u - complex(0, 1) * R)).real

        V0 = integrate.quad(integrand, 0, 50)
        return V0

    def heston_char(self, u, S0, r, gam0, kappa, lamb, sig_tild, T):
        d = np.sqrt(lamb ** 2 + sig_tild ** 2 * (u * 1j + u ** 2))
        phi = np.cosh(0.5 * d * T)
        psi = np.sinh(0.5 * d * T) / d
        first_factor = (np.exp(lamb * T / 2) / (phi + lamb * psi)) ** (2 * kappa / sig_tild ** 2)
        second_factor = np.exp(-gam0 * (u * 1j + u ** 2) * psi / (phi + lamb * psi))
        return np.exp(u * 1j * (np.log(S0) + r * T)) * first_factor * second_factor

    def Heston_FFT(self, S0, r, gam0, kappa, lamb, sig_tild, T, K, R, N):
        K = np.atleast_1d(K)
        f_tilde_0 = lambda u: 1 / (u * (u - 1))
        chi_0 = lambda u: self.heston_char(u, S0=S0, r=r, gam0=gam0, kappa=kappa, lamb=lamb, sig_tild=sig_tild, T=T)
        g = lambda u: f_tilde_0(R + 1j * u) * chi_0(u - 1j * R)

        kappa_1 = np.log(K[0])
        M = np.minimum(2 * np.pi * (N - 1) / (np.log(K[-1]) - kappa_1), 500)
        Delta = M / N
        n = np.arange(1, N + 1)
        kappa_m = np.linspace(kappa_1, kappa_1 + 2 * np.pi * (N - 1) / M, N)

        x = g((n - 0.5) * Delta) * Delta * np.exp(-1j * (n - 1) * Delta * kappa_1)
        x_hat = np.fft.fft(x)

        V_kappa_m = np.exp(-r * T + (1 - R) * kappa_m) / np.pi * np.real(x_hat * np.exp(-0.5 * Delta * kappa_m * 1j))
        return interp1d(kappa_m, V_kappa_m)(np.log(K))