import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares
import ivol_model


class SABRVolatility(ivol_model.VolatilityModel):
    def __init__(self, alpha=0.1, beta=0.1, rho=0.1, nu=0.1, fwd=1.0, strikes=None, volatilities=None, T=None, interv=1):
        """
        Initializes the SABR model with either given SABR parameters or a single volatility curve

        Parameters:
        - alpha: volatility
        - beta: elasticity
        - rho: correlation
        - nu: vol of vol
        - fwd: forward
        - strikes: strikes
        - volatilities at time to expiry T
        - T: time to expiry
        - interv: max number of points for fitting to ensure convergence
        """
        if strikes is not None and volatilities is not None and T is not None:
            self.strikes = strikes
            self.volatilities = volatilities
            self.T = T
            self.fwd = fwd
            self.alpha = 0.1
            self.beta = 0.1
            self.rho = 0.1
            self.nu = 0.1
            self.interv = interv
            self.fit_to_volatility_curve(strikes, volatilities, T, interv)
        else:
            self.alpha = alpha
            self.beta = beta
            self.rho = rho
            self.nu = nu
            self.fwd = fwd

    def implied_volatility(self, strike, T):
        """
        Calculates the SABR implied volatility for strike and time to maturity T

        Parameters:
        - strike: strikes
        - T: time to maturity

        Returns:
        - volatility: SABR implied volatility for the strike and time T
        """
        epsilon = 1e-12  # small number to avoid division by zero
        F = self.fwd
        if isinstance(strike, (int, float)):
            if strike <= 0:
                strike = epsilon

            if F == strike:
                # Special case where F == K to avoid division by zero in log(F/K)
                return self.alpha / (F ** (1 - self.beta))

            z = (self.nu / self.alpha) * (F * strike) ** ((1 - self.beta) / 2) * np.log(F / strike)
            x = np.log((np.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho))
            factor1 = self.alpha / ((F * strike) ** ((1 - self.beta) / 2) * (1 + ((1 - self.beta)**2 / 24 * (np.log(F / strike))**2 + ((1 - self.beta)**4 / 1920 * (np.log(F / strike))**4))))
            volatility = factor1 * (z / x)
        else:
            strike = np.where(strike <= 0, epsilon, strike)

            if F == strike:
                # Special case where F == K to avoid division by zero in log(F/K)
                return self.alpha / (F ** (1 - self.beta))

            z = (self.nu / self.alpha) * (F * strike) ** ((1 - self.beta) / 2) * np.log(F / strike)
            x = np.log((np.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho))
            factor1 = self.alpha / ((F * strike) ** ((1 - self.beta) / 2) * (1 + ((1 - self.beta)**2 / 24 * (np.log(F / strike))**2 + ((1 - self.beta)**4 / 1920 * (np.log(F / strike))**4))))
            volatility = factor1 * (z / x)

        return volatility

    def fit_to_volatility_curve(self, strikes, volatilities, T, interv=1):
        """
        Fits the SABR model parameters to a given volatility curve (strike, vol) using optimization.

        Parameters:
        - strikes: strike prices
        - volatilities: volatilities at the given strikes
        - T: time to expiry
        - interv: interval between 2 points for fitting to ensure convergence

        Returns:
        - success: bool, indicating whether the optimization was successful
        - params: dict, fitted model parameters
        """

        def error_function(param):
            model_vols = np.array(
                [self.implied_volatility(strike, T) for strike in strikes_subset])  # Assuming time T=1 for fitting
            return np.sum((model_vols - volatilities_subset) ** 2)

        # Normalize volatilities
        volatilities_normalized = volatilities / np.max(volatilities)

        # Subset data if necessary
        if len(strikes) > interv:
            indices = np.linspace(0, len(strikes) - 1, interv, dtype=int)
            strikes_subset = strikes[indices]
            volatilities_subset = volatilities_normalized[indices]
        else:
            strikes_subset = strikes
            volatilities_subset = volatilities_normalized

        initial_guess = np.array([self.alpha or 0.1, self.beta or 0.1, self.rho or 0.1, self.nu or 0.1])
        # initial_guess = [[0.1, 0.2, 0.7, 0.6], [0.9, 0.3, -0.4, 0.3], [0.6, 0.8, -0.8, 0.5],
        #                  [0.9, 0.2, -0.9, 0.5], [0.3, 0.8, 0.4, 0.8]]
        bounds = [(0.001, 0.999), (0.001, 1.0), (-0.999, 0.999), (0.001, 0.999)]      # minimize method
        # bounds = ([0.001, 0.001, -0.999, 0.001], [1.0, 1.0, 0.999, 1.0])            # least_squares method
        options ={
            'maxiter': 10000,  # Maximum number of iterations
            'disp': True,      # Display convergence messages
            'xatol': 1e-12,     # Absolute error in xopt between iterations that is acceptable for convergence
            'fatol': 1e-12,     # Absolute error in func(xopt) between iterations that is acceptable for convergence
            'initial_simplex': np.array([[0.9, 0.3, -0.4, 0.3], [0.1, 0.2, 0.7, 0.6], [0.6, 0.8, -0.8, 0.5],
                                [0.9, 0.2, -0.9, 0.5], [0.3, 0.8, 0.4, 0.8]]),
            'adaptive': True
        }

        # Debug: Print initial guess and bounds
        print(f"Initial guess: {initial_guess}")
        print(f"Bounds: {bounds}")

        # Using different optimization methods
        methods = ['Nelder-Mead']                           # minimize method
        for method in methods:
            result = minimize(error_function, initial_guess, tol=1e-12, bounds=bounds, method=method, options=options)
            print(f"Optimization method: {method}")
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Function value: {result.fun}")

            # result = least_squares(error_function, initial_guess, bounds=bounds, method='trf')
            if result.success:
                alpha_fit, beta_fit, rho_fit, nu_fit = result.x
                self.alpha = alpha_fit
                self.beta = beta_fit
                self.rho = rho_fit
                self.nu = nu_fit
                return result.success, {'alpha': alpha_fit, 'beta': beta_fit, 'rho': rho_fit, 'nu': nu_fit}

        return False, {}

    def plot_volatility_surface(self, strikes, maturities):
        """
        Build and plot the volatility surface for given arrays of strikes and maturities.

        Parameters:
        - strikes: array_like, strike prices
        - maturities: array_like, time to maturities in years
        """
        strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
        volatility_surface = np.zeros_like(strike_grid)

        for i in range(strike_grid.shape[0]):
            for j in range(strike_grid.shape[1]):
                volatility_surface[i, j] = self.implied_volatility(strike_grid[i, j], maturity_grid[i, j])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(strike_grid, maturity_grid, volatility_surface, cmap='viridis')

        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        ax.set_title('SABR Volatility Surface')
        # plt.show()

        plt.ion()
        plt.draw()
        plt.pause(1)
        input("Press [enter] to continue.")

# Example usage:
def test():
    strikes = np.linspace(80, 120, 50)
    fwd = 10.0
    alpha = 0.2
    beta = 0.5
    rho = -0.25
    nu = 0.3
    T = 1.0  # Time to maturity
    max_simu = 1

    # Create synthetic volatility data for fitting
    sabr_model = SABRVolatility(alpha=alpha, beta=beta, rho=rho, nu=nu, fwd=fwd)
    true_vols = np.array([sabr_model.implied_volatility(strike, T) for strike in strikes])

    # Add some noise to the synthetic volatilities to simulate market data
    noisy_vols = true_vols + np.random.normal(0, 0.001, size=true_vols.shape)
    print("Noisy Vols built")
    print()

    # Fit the SABR model to the noisy volatility data using the class method
    sabr_model_fit = SABRVolatility(strikes=strikes, volatilities=noisy_vols, T=T, fwd=fwd)
    print(
        f"Fitted parameters: alpha={sabr_model_fit.alpha}, beta={sabr_model_fit.beta}, rho={sabr_model_fit.rho}, nu={sabr_model_fit.nu}")

    # Plot the original and fitted volatility curves
    fitted_vols = np.array([sabr_model_fit.implied_volatility(strike, T) for strike in strikes])

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, true_vols, label='True Volatilities', marker='o')
    plt.plot(strikes, noisy_vols, label='Noisy Volatilities', linestyle='--', marker='x')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('True and Noisy Volatility Curves')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, true_vols, label='True Volatilities', marker='o')
    plt.plot(strikes, fitted_vols, label='Fitted Volatilities', linestyle='-.', marker='s')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('True and Fitted Volatility Curves')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, noisy_vols, label='Noisy Volatilities', linestyle='--', marker='x')
    plt.plot(strikes, fitted_vols, label='Fitted Volatilities', linestyle='-.', marker='s')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('Fitted and Noisy Volatility Curves')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, true_vols, label='True Volatilities', marker='o')
    plt.plot(strikes, noisy_vols, label='Noisy Volatilities', linestyle='--', marker='x')
    plt.plot(strikes, fitted_vols, label='Fitted Volatilities', linestyle='-.', marker='s')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('SABR Model Fit to Volatility Curve')
    plt.legend()
    plt.show()


def main():
    test()


if __name__ == "__main__":
    main()
