import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit, basinhopping
import ivol_model


class SABRVolatility(ivol_model.VolatilityModel):
    def __init__(self, alpha=0.1, beta=0.1, rho=0.1, nu=0.1, fwd=1.0, strikes=None, volatilities=None, T=None, nb=50,
                 optimizer=3, method=None, fitted=False, fun=None, inbound=False):
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
        - nb: sample number of points for fitting to ensure convergence
        - optimizer: 1: Minimize, 2: Curve fit, 3: Basinhopping
        - methods: Minimize methods as list ['','']
        - fitted: indicates whether the curve has been calibrated or not
        - fun: calculates the optimum function at the calibration point
        - inbound: specify whether calibration is accounting for bounds or not
        """
        if strikes is not None and volatilities is not None and T is not None:
            self.strikes = strikes
            self.volatilities = volatilities
            self.T = T
            self.fwd = fwd
            self.nb = nb
            self.optimizer = optimizer
            self.method = 'Nelder-Mead'
            self.fitted = False
            self.fun = None
            self.alpha = 0.1
            self.beta = 0.1
            self.rho = 0.1
            self.nu = 0.1
            self.fit_to_volatility_curve(strikes, volatilities, T, nb, optimizer, method, inbound)           # sample of 50 points for calibration
        else:
            self.alpha = alpha
            self.beta = beta
            self.rho = rho
            self.nu = nu
            self.fwd = fwd


    def implied_volatility(self, strike, t):
        return SABRVolatility.implied_volatility_sabr(self.alpha, self.beta, self.rho, self.nu, self.fwd, strike, t)


    @classmethod
    def implied_volatility_sabr(cls, alpha, beta, rho, nu, f, strike, t):
        """
        Calculates the SABR implied volatility for strike and time to maturity T

        Parameters:
        - strike: strikes
        - T: time to maturity

        Returns:
        - volatility: SABR implied volatility for the strike and time T
        """
        epsilon = 1e-12  # small number to avoid division by zero
        if isinstance(strike, (int, float)):
            if strike <= 0:
                strike = epsilon

            z = (nu / alpha) * ((f * strike) ** ((1 - beta) / 2)) * np.log(f / strike)
            x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            factor1 = alpha / ((f * strike) ** ((1 - beta) / 2) * (1 + ((((1 - beta)**2) / 24) * (np.log(f / strike))**2 + ((1 - beta)**4 / 1920) * (np.log(f / strike))**4)))
            factor2 = 1 + ((((1 - beta)**2) / 24) * ((alpha ** 2) / ((f * strike) ** (1 - beta))) + ((rho * beta * nu * alpha) / (4 * ((f * strike) ** ((1 - beta) / 2)))) + ((2 - 3 * (rho ** 2)) * (nu ** 2)) / 24) * t

            if f == strike:
                # Special case where F == K to avoid division by zero in log(F/K)
                return factor1 * factor2

            volatility = factor1 * factor2 * (z / x)

        else:
            strike = np.where(strike <= 0, epsilon, strike)

            z = (nu / alpha) * ((f * strike) ** ((1 - beta) / 2)) * np.log(f / strike)
            x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))
            factor1 = alpha / ((f * strike) ** ((1 - beta) / 2) * (1 + ((((1 - beta)**2) / 24) * (np.log(f / strike))**2 + ((1 - beta)**4 / 1920) * (np.log(f / strike))**4)))
            factor2 = 1 + ((((1 - beta)**2) / 24) * ((alpha ** 2) / ((f * strike) ** (1 - beta))) + ((rho * beta * nu * alpha) / (4 * ((f * strike) ** ((1 - beta) / 2)))) + ((2 - 3 * (rho ** 2)) * (nu ** 2)) / 24) * t

            if f == strike:
                # Special case where F == K to avoid division by zero in log(F/K)
                return factor1 * factor2

            volatility = factor1 * factor2 * (z / x)

        return volatility

    def fit_to_volatility_curve(self, strikes, volatilities, T, nb=50, optimizer=3, method='Nelder-Mead', inbound=False):
        """
        Fits the SABR model parameters to a given volatility curve (strike, vol) using optimization.

        Parameters:
        - strikes: strike prices
        - volatilities: volatilities at the given strikes
        - T: time to expiry
        - nb: sample number of points for fitting to ensure convergence
        - Optimizer: 1: Minimize, 2: Curve fit, 3: Basinhopping

        Returns:
        - success: bool, indicating whether the optimization was successful
        - params: dict, fitted model parameters
        """

        # Normalize volatilities
        # volatilities_normalized = volatilities / np.max(volatilities)             # not used anymore

        # Subset data if necessary
        if len(strikes) > nb:
            indices = np.linspace(0, len(strikes) - 1, nb, dtype=int)
            strikes_subset = strikes[indices]
            # volatilities_subset = volatilities_normalized[indices]
            volatilities_subset = volatilities[indices]
        else:
            strikes_subset = strikes
            # volatilities_subset = volatilities_normalized
            volatilities_subset = volatilities

        def error_function(x):
            alpha, beta, rho, nu = x
            model_vols = np.array([SABRVolatility.implied_volatility_sabr(alpha, beta, rho, nu, self.fwd, strike,
                                                                          self.T) for strike in strikes_subset])
            return np.sum((model_vols - volatilities_subset) ** 2)

        def impl_vol_function(x, alpha, beta, rho, nu):
            return np.array([SABRVolatility.implied_volatility_sabr(alpha, beta, rho, nu, self.fwd, x, self.T)
                             for x in strikes_subset])

        initial_guess = np.array([1.658, 0.665, 0.044, 2.400])
        bounds = [(0.001, 0.999), (0.001, 0.999), (-0.999, 0.999), (0.001, 0.999)] if inbound else None       # minimize method
        bounds2 = ([0.001, 0.001, -0.999, 0.001], [0.999, 0.999, 0.999, 0.999]) if inbound else None         # least_squares method

        tolerance = 1e-12
        options ={
            'maxiter': 100000,  # Maximum number of iterations
            'disp': False,      # Display convergence messages
            'xatol': tolerance,     # Absolute error in xopt between iterations that is acceptable for convergence
            'fatol': tolerance,     # Absolute error in func(xopt) between iterations that is acceptable for convergence
            #'initial_simplex': np.array([[0.9, 0.3, -0.4, 0.3], [0.1, 0.2, 0.7, 0.6], [0.6, 0.8, -0.8, 0.5],
            #                    [0.9, 0.2, -0.9, 0.5], [0.3, 0.8, 0.4, 0.8]]),
            'adaptive': True
        }

        if optimizer == 1:          # optimizer = mininmize
            # Print initial guess and bounds
            print("Calibration with minimize function ...")
            print(f"Initial guess: {initial_guess}")
            print(f"Bounds: {bounds}")

            # Using optimization method
            result = minimize(error_function, initial_guess, tol=tolerance, bounds=bounds, method=method, options=options)
            print(f"Optimization method: {method}")
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Function value: {result.fun}")

            if result.success:
                alpha_fit, beta_fit, rho_fit, nu_fit = result.x
                self.alpha = alpha_fit
                self.beta = beta_fit
                self.rho = rho_fit
                self.nu = nu_fit
                self.fitted = True
                self.fun = result.fun
                return result.success, {'alpha': alpha_fit, 'beta': beta_fit, 'rho': rho_fit, 'nu': nu_fit}
            else:
                print(f"No Success for: {method}")
                return False, {}
        elif optimizer == 2:
            try:
                popt, pcov, infodict = curve_fit(impl_vol_function, strikes_subset, volatilities_subset, initial_guess,
                                       bounds=bounds2, method='trf', full_output=True)
                self.alpha, self.beta, self.rho, self.nu = popt
                self.fitted = True
                residuals = infodict['fvec']
                self.fun = np.sum(residuals**2)
                return popt
            except RuntimeError as e:
                print(e)
        elif optimizer == 3:
            try:
                rng = np.random.default_rng()
                minimizer_kwargs = {'method': method, 'bounds': bounds}
                result = basinhopping(error_function, x0=initial_guess, niter=50, T=1, stepsize=0.3, seed=rng,
                                      minimizer_kwargs=minimizer_kwargs, disp=True)
                if result.lowest_optimization_result.success:
                    self.alpha, self.beta, self.rho, self.nu = result.x
                    self.fitted = True
                    self.fun = result.fun
                    return result.lowest_optimization_result.success, {'alpha': self.alpha, 'beta': self.beta,
                                                                       'rho': self.rho, 'nu': self.nu}
                else:
                    return False, {}
            except RuntimeError as e:
                print(e)
        else:
            print("Only choices for optimizer: 1, 2 or 3")

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
    strikes = np.linspace(40, 120, 50)
    fwd = 70.0
    alpha = 0.3
    beta = 0.8
    rho = 0.05
    nu = 0.8
    T = 0.5  # Time to maturity

    # Create synthetic volatility data for fitting
    sabr_model = SABRVolatility(alpha=alpha, beta=beta, rho=rho, nu=nu, fwd=fwd)
    true_vols = np.array([sabr_model.implied_volatility(strike, T) for strike in strikes])

    # Add some noise to the synthetic volatilities to simulate market data
    noisy_vols = true_vols + np.random.normal(0, 0.001, size=true_vols.shape)

    # Fit the SABR model to the noisy volatility data using the class method
    sabr_model_fit = SABRVolatility(strikes=strikes, volatilities=noisy_vols, T=T, fwd=fwd)
    print(f"True parameters: alpha={alpha}, beta={beta}, rho={rho}, nu={nu}")
    print(f"Fitted parameters: alpha={sabr_model_fit.alpha}, beta={sabr_model_fit.beta}, rho={sabr_model_fit.rho}, nu={sabr_model_fit.nu}")

    # Plot the original and fitted volatility curves
    fitted_vols = np.array([sabr_model_fit.implied_volatility(strike, T) for strike in strikes])
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
