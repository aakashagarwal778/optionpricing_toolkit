import numpy as np
import math

class CRRModel:
    def CRR_stock(self, S_0, r, sigma, T, M):
        delta_t = T / M
        beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + sigma**2) * delta_t))
        u = beta + math.sqrt(math.pow(beta, 2) - 1)
        d = 1 / u

        S = np.empty((M + 1, M + 1))
        for i in range(M + 1):
            for j in range(i + 1):
                S[j, i] = S_0 * (u ** j) * (d ** (i - j))
        return S

    def CRR_Option(self, S_0, r, sigma, T, M, K, option_type='call', option_style='European'):
        delta_t = T / M
        beta = 0.5 * (math.exp(-r * delta_t) + math.exp((r + sigma**2) * delta_t))
        u = beta + math.sqrt(math.pow(beta, 2) - 1)
        d = beta - math.sqrt(math.pow(beta, 2) - 1)
        q = (math.exp(r * delta_t) - d) / (u - d)

        S = self.CRR_stock(S_0, r, sigma, T, M)
        if option_type == 'call':
            V = np.maximum(0, S - K)
        elif option_type == 'put':
            V = np.maximum(0, K - S)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        if option_style == 'European':
            for i in range(M-1, -1, -1):
                for j in range(i+1):
                    V[j, i] = math.exp(-r * delta_t) * (q * V[j, i+1] + (1 - q) * V[j+1, i+1])
        elif option_style == 'American':
            for i in range(M-1, -1, -1):
                for j in range(i+1):
                    V[j, i] = max(V[j, i], math.exp(-r * delta_t) * (q * V[j, i+1] + (1 - q) * V[j+1, i+1]))

        return V[0, 0]