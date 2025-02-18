import math
from scipy.stats import norm

class BlackScholesModel:
    def BlackScholes_Option(self, S_0, r, sigma, T, K, option_type='call'):
        d1 = (math.log(S_0 / K) + (r + (sigma ** 2) / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == 'call':
            V_0 = S_0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'put':
            V_0 = K * math.exp(-r * T) * norm.cdf(-d2) - S_0 * norm.cdf(-d1)
        else:
            raise ValueError("Option type must be either 'call' or 'put'.")

        return V_0