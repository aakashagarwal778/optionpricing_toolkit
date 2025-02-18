from .models.crr_model import CRRModel
from .models.black_scholes_model import BlackScholesModel
from .models.monte_carlo_model import MonteCarloModel
from .models.numerical_integration_model import NumericalIntegrationModel
from .models.laplace_transform_model import LaplaceTransformModel
from .models.finite_difference_model import FiniteDifferenceModel

class OptionPricing:
    def __init__(self, S_0, r, sigma, T, K, M):
        self.S_0 = S_0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.M = M
        self.crr_model = CRRModel()
        self.bs_model = BlackScholesModel()
        self.mc_model = MonteCarloModel()
        self.ni_model = NumericalIntegrationModel()
        self.lt_model = LaplaceTransformModel()
        self.fd_model = FiniteDifferenceModel()