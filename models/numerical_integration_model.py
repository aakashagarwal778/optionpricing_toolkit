import math
from scipy import integrate
import numpy as np

class NumericalIntegrationModel:
    def BS_Price_Int(self, S0, r, sigma, T, K, option_type='call'):
        def call_payoff(ST, K):
            return np.maximum(ST - K, 0)

        def put_payoff(ST, K):
            return np.maximum(K - ST, 0)

        def integrand(x):
            norm_const = 1 / math.sqrt(2 * math.pi)
            exponent = (r - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * x
            stock_price_term = S0 * math.exp(exponent)

            if option_type == 'call':
                payoff = call_payoff(stock_price_term, K)
            elif option_type == 'put':
                payoff = put_payoff(stock_price_term, K)
            else:
                raise ValueError("Option type must be 'call' or 'put'.")

            discount_factor = math.exp(-r * T)
            normal_exp_term = math.exp(-0.5 * math.pow(x, 2))
            V = norm_const * payoff * discount_factor * normal_exp_term
            return V

        integral = integrate.quad(integrand, -np.inf, np.inf)
        return integral[0]