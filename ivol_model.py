from abc import ABC, abstractmethod


class VolatilityModel(ABC):
    @abstractmethod
    def implied_volatility(self, strike, T):
        """
         Calculate implied volatility for a given strike and time to maturity T

         Parameters:
         - strike: strikes
         - T: time to maturity in years

         Returns:
         - volatility: implied volatility for to the strikes and time T
         """
        pass

    @abstractmethod
    def fit_to_volatility_curve(self, strikes, volatilities, T):
        """
        Fits the model parameters to a given volatility curve (strike, vol)

        Parameters:
        - strikes: strikes
        - volatilities: volatilities at the given strikes
        - T: maturity

        Returns:
        - success: bool, indicating whether the optimization was successful
        - params: dict, fitted model parameters
        """
        pass

