import numpy as np
import math
from scipy.integrate import solve_ivp

class FiniteDifferenceModel:
    def AmPerpPut_ODE(self, S_max, N, r, sigma, K):
        g = lambda S: np.maximum(K - S, 0)

        S_grid = np.linspace(0, S_max, N + 1)
        v_grid = np.zeros_like(S_grid)

        def fun(x, v):
            return np.array([v[1], 2 * r / (sigma ** 2 * x ** 2) * (v[0] - x * v[1])])

        x_star = 2 * K * r / (2 * r + sigma ** 2)
        v_grid[S_grid <= x_star] = g(S_grid[S_grid <= x_star])

        result = solve_ivp(fun=fun, t_span=(x_star, S_max), y0=[g(x_star), -1], t_eval=S_grid[S_grid > x_star])
        v_grid[S_grid > x_star] = result.y[0]

        return S_grid, v_grid

    def BS_EuOption_FiDi_Explicit(self, S0, r, sigma, T, K, m, nu_max, option_type='call'):
        q = 2 * r / sigma ** 2
        delta_x = (math.log(2 * K / S0)) / m
        delta_t = sigma ** 2 * T / (2 * nu_max)
        fidi_lambda = delta_t / delta_x ** 2
        lambda_tilde = (1 - 2 * fidi_lambda)

        x = np.arange(-m, m + 1) * delta_x
        w = np.zeros((2 * m + 1, nu_max + 1))

        if option_type == 'call':
            w[:, 0] = np.maximum(S0 * np.exp(x) - K, 0)
        elif option_type == 'put':
            w[:, 0] = np.maximum(K - S0 * np.exp(x), 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

        for i in range(1, nu_max + 1):
            for j in range(1, 2 * m):
                w[j, i] = fidi_lambda * w[j - 1, i - 1] + lambda_tilde * w[j, i - 1] + fidi_lambda * w[j + 1, i - 1]

        index_S0 = np.argmin(np.abs(S0 - S0 * np.exp(x)))
        V0 = np.exp(-r * T) * w[index_S0, nu_max]
        return V0