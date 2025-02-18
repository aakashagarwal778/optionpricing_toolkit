import numpy as np
import math

class MonteCarloModel:
    def MonteCarlo_Option(self, S0, r, sigma, T, K, M, option_type='call'):
        X = np.random.normal(0, 1, M)
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * X)

        if option_type == 'call':
            payoff_fun = np.maximum(ST - K, 0)
        elif option_type == 'put':
            payoff_fun = np.maximum(K - ST, 0)
        else:
            raise ValueError("Option type must be 'call' or 'put'.")

        V0 = np.exp(-r * T) * np.mean(payoff_fun)
        se = np.std(payoff_fun) / np.sqrt(M)
        z = 1.96
        c1 = V0 - z * se
        c2 = V0 + z * se

        return V0, c1, c2