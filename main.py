import math
import numpy as np
from pricing_toolkit import OptionPricing

# Initialize the OptionPricing class with parameters
option_pricing = OptionPricing(S_0=100, r=0.05, sigma=0.2, T=1, K=100, M=50)

# Test parameters
S_0 = 100  # Initial stock price
r = 0.05   # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
T = 1      # Time to expiration (in years)
K = 105    # Strike price
M = 100    # Number of time steps for CRR model

# Test commands for different methods and scenarios
option_type = 'call'
option_style = 'European'

# Calculate the option price using the CRR model
option_price_crr_eu_call = option_pricing.crr_model.CRR_Option(S_0, r, sigma, T, M, K, option_type, option_style)
print(f"European Call Option Price (CRR model): {option_price_crr_eu_call}")

# Test Cox-Ross-Rubinstein (CRR) model for an American put option
option_type = 'put'
option_style = 'American'
option_price_crr_am_put = option_pricing.crr_model.CRR_Option(S_0, r, sigma, T, M, K, option_type, option_style)
print(f"American Put Option Price (CRR model): {option_price_crr_am_put}")

#Test Black-Scholes model for a European put option
option_type = 'put'
option_price_bs_eu_put = option_pricing.bs_model.BlackScholes_Option(S_0, r, sigma, T, K, option_type)
print(f"European Put Option Price (Black-Scholes model): {option_price_bs_eu_put}")

# Test Monte Carlo simulation for a European call/put option
M = 10000  # Increase number of simulations for Monte Carlo
option_type = 'call'
option_price_mc_eu_call, _, _ = option_pricing.mc_model.MonteCarlo_Option(S_0, r, sigma, T, K, M, option_type)
print(f"European Call Option Price (Monte Carlo simulation): {option_price_mc_eu_call}")

option_type = 'put'
option_price_mc_eu_put, _, _ = option_pricing.mc_model.MonteCarlo_Option(S_0, r, sigma, T, K, M, option_type)
print(f"European Put Option Price (Monte Carlo simulation): {option_price_mc_eu_put}")


# Test numerical integration for an European call/put option
option_type = 'call'
option_price_int_eu_call = option_pricing.ni_model.BS_Price_Int(S_0, r, sigma, T, K, option_type)
print(f"European Call Option Price (Numerical Integration): {option_price_int_eu_call}")

option_type = 'put'
option_price_int_eu_put = option_pricing.ni_model.BS_Price_Int(S_0, r, sigma, T, K, option_type)
print(f"European Put Option Price (Numerical Integration): {option_price_int_eu_put}")

# Test Laplace transform method in Black-Scholes model for a European call option
R = 1.1
option_type = 'call'
option_price_laplace_call = option_pricing.lt_model.laplace_BS(S_0, r, sigma, T, K, R, option_type)
print(f"European Call Option Price (Laplace transform method): {option_price_laplace_call}")

#Test Heston model using Laplace transform for a European put option
# Heston model parameters
gam0 = math.pow(0.3, 2)
kappa = math.pow(0.3, 2)
lamb = 2.5
sig_tild = 0.2
p = 1
option_price_heston_eu_put = option_pricing.lt_model.laplace_heston(S_0, r, gam0, kappa, lamb, sig_tild, T, K, R=1.5, p=p)
print(f"Heston European Put Option Price (Laplace transform method): {option_price_heston_eu_put[0]}")


# Test for the Fast Fourier Method (FFT) in Heston model
# test fft method
N = 1000
option_price_heston_eu_call_fft = option_pricing.lt_model.Heston_FFT(S_0, r, gam0, kappa, lamb, sig_tild, T, K=np.arange(80, 181), R=1.5, N=N)
print(f"Heston European Call Option Price (FFT method): {option_price_heston_eu_call_fft[0]}")


#Test for European call option using Finite Difference Scheme
m = 100  # Number of space steps
nu_max = 1000  # Number of time steps

# Test for European call option using Finite Difference Scheme
option_type = 'call'
option_price_fd_eu_call = option_pricing.fd_model.BS_EuOption_FiDi_Explicit(S_0, r, sigma, T, K, m, nu_max, option_type)
print(f"European Call Option Price (Finite Difference Scheme): {option_price_fd_eu_call}")

# Test for European put option using Finite Difference Scheme
option_type = 'put'
option_price_fd_eu_put = option_pricing.fd_model.BS_EuOption_FiDi_Explicit(S_0, r, sigma, T, K, m, nu_max, option_type)
print(f"European Put Option Price (Finite Difference Scheme): {option_price_fd_eu_put}")