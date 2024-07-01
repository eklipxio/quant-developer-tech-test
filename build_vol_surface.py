import read_csv_to_dict as rd
import vol_model_sabr as sb
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def year_fract(start, end, conv="actual365"):
    """
    Calculates the year fraction between start and end using Actual/days_ref convention
    Not relying on QuantLib while installation is being sorted

    Parameters:
    - start: starting date, format: YYYY-MM-DD
    - end: ending date, format: YYYY-MM-DD
    - conv: "actual365" for 365 days in a year, "actual360" for 360

    Returns:
    - year fraction between start and end
    """
    days_year = 360 if conv == "actual360" else 365

    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d")
    day_count = (end_date - start_date).days
    year_fraction = day_count / days_year
    return year_fraction


def test_basinhopping_calibration(path, data, start, fwd, end_row, end_col, debut_col, min_val, max_val):
    """
    Calibrates the SABR parameters for volatility curve fitting using basinhopping optimizer and several methods

    Parameters:
    - path: file name
    - data: clean vs. noisy
    - start: starting date, format: YYYY-MM-DD
    - fwd: forward
    - end_row: ending date row in csv file
    - end_col: ending date column in csv file
    - debut_col: starting column for volatilities in csv file
    - min_val: minimum strike value selected for calibration
    - max_val: maximum strike value selected for calibration

    Returns:
    - alpha, beta, rho and nu fitting parameters saved in a file for each optimizer
    - images for true and fitted vols are also saved
    """
    # time to expiry
    end = rd.read_cell(path, end_row, end_col)
    t = year_fract(start, end)

    # strikes and volatilities
    data_dict = rd.csv_to_dicts(path, debut_col)
    strikes = np.array([float(key) for key in data_dict[0]])
    volatilities = np.array([float(value)/100 for value in data_dict[0].values()])
    nb = len(strikes)

    # calibration for a specific strike range: [min_val, max_val]
    strikes_reduced = np.array([strike for strike in strikes if min_val <= strike <= max_val])
    first_elt = strikes_reduced[0]
    res = np.where(strikes == first_elt)
    init_idx = res[0][0]
    last_idx = init_idx + len(strikes_reduced)
    volatilities_reduced = volatilities[init_idx:last_idx]

    # Test basinhopping calibration for different methods
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'COBYLA', 'SLSQP', 'trust-constr']
    with open(f"SABR {data} fitted parameters - Basinhopping.txt", "a") as f:
        f.truncate(0)
        for method in methods:
            print(f"Running Basinhopping optimization for {data} with {method}")
            sabr_model = sb.SABRVolatility(fwd=fwd, strikes=strikes, volatilities=volatilities, T=t, nb=nb, optimizer=3,
                                           method=method)
            if sabr_model.fitted:
                print(f"Success! Fitted parameters: alpha={sabr_model.alpha}, beta={sabr_model.beta}, rho={sabr_model.rho},"
                      f"nu={sabr_model.nu}")
                # Save the fitted parameters to a text file
                f.write(f"Method = {method}\n")
                f.write(f"F(solution) = {sabr_model.fun}\n")
                f.write(f"Fitted parameters:\n")
                f.write(f"alpha = {sabr_model.alpha}\n")
                f.write(f"beta = {sabr_model.beta}\n")
                f.write(f"rho = {sabr_model.rho}\n")
                f.write(f"nu = {sabr_model.nu}\n\n")

                # Plot the original and fitted volatility curves
                fitted_vols = np.array([sabr_model.implied_volatility(strike, t) for strike in strikes])

                # Plot the calibrated SABR model for whole strike range
                plt.figure(figsize=(10, 6))
                plt.plot(strikes, volatilities, label='True Volatilities', marker='o')
                plt.plot(strikes, fitted_vols, label='Fitted Volatilities', linestyle='-.', marker='s')
                plt.xlabel('Strike')
                plt.ylabel('Volatility')
                plt.title(f'SABR Model Fit to {data} - Basinhopping - {method}')
                plt.legend()
                plt.savefig(f'images/basinhopping/SABR Vols - {data} - Basinhopping - {method}.png')
                # plt.show()
            else:
                print("Optimization did not converge")
                f.write(f"Method = {method}\n")
                f.write("Optimization did not converge\n\n")


def test_minimize_calibration(path, data, start, fwd, end_row, end_col, debut_col, min_val, max_val):
    """
    Calibrates the SABR parameters for volatility curve fitting using minimize optimizer and several methods

    Parameters:
    - path: file name
    - data: clean vs. noisy
    - start: starting date, format: YYYY-MM-DD
    - fwd: forward
    - end_row: ending date row in csv file
    - end_col: ending date column in csv file
    - debut_col: starting column for volatilities in csv file
    - min_val: minimum strike value selected for calibration
    - max_val: maximum strike value selected for calibration

    Returns:
    - alpha, beta, rho and nu fitting parameters saved in a file for each optimizer
    - images for true and fitted vols are also saved
    """
    # time to expiry
    end = rd.read_cell(path, end_row, end_col)
    t = year_fract(start, end)

    # strikes and volatilities
    data_dict = rd.csv_to_dicts(path, debut_col)
    strikes = np.array([float(key) for key in data_dict[0]])
    volatilities = np.array([float(value)/100 for value in data_dict[0].values()])
    nb = len(strikes)

    # calibration for a specific strike range: [min_val, max_val]
    strikes_reduced = np.array([strike for strike in strikes if min_val <= strike <= max_val])
    first_elt = strikes_reduced[0]
    res = np.where(strikes == first_elt)
    init_idx = res[0][0]
    last_idx = init_idx + len(strikes_reduced)
    volatilities_reduced = volatilities[init_idx:last_idx]

    # Test basinhopping calibration for different methods
    methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'COBYLA', 'SLSQP', 'trust-constr']
    with open(f"SABR {data} fitted parameters - Minimize.txt", "a") as f:
        f.truncate(0)
        for method in methods:
            print(f"Running Minimize optimization for {data} with {method}")
            sabr_model = sb.SABRVolatility(fwd=fwd, strikes=strikes, volatilities=volatilities, T=t, nb=nb, optimizer=1,
                                           method=method, inbound=False)
            if sabr_model.fitted:
                print(f"Success! Fitted parameters: alpha={sabr_model.alpha}, beta={sabr_model.beta}, rho={sabr_model.rho},"
                      f"nu={sabr_model.nu}")
                # Save the fitted parameters to a text file
                f.write(f"Method = {method}\n")
                f.write(f"F(solution) = {sabr_model.fun}\n")
                f.write(f"Fitted parameters:\n")
                f.write(f"alpha = {sabr_model.alpha}\n")
                f.write(f"beta = {sabr_model.beta}\n")
                f.write(f"rho = {sabr_model.rho}\n")
                f.write(f"nu = {sabr_model.nu}\n\n")

                # Plot the original and fitted volatility curves
                fitted_vols = np.array([sabr_model.implied_volatility(strike, t) for strike in strikes])

                # Plot the calibrated SABR model for whole strike range
                plt.figure(figsize=(10, 6))
                plt.plot(strikes, volatilities, label='True Volatilities', marker='o')
                plt.plot(strikes, fitted_vols, label='Fitted Volatilities', linestyle='-.', marker='s')
                plt.xlabel('Strike')
                plt.ylabel('Volatility')
                plt.title(f'SABR Model Fit to {data} - Minimize - {method}')
                plt.legend()
                plt.savefig(f'images/minimize/SABR Vols - {data} - Minimize - {method}.png')
                # plt.show()
            else:
                print("Optimization did not converge")
                f.write(f"Method = {method}\n")
                f.write("Optimization did not converge\n\n")


def test_specific_calibration(path, start, fwd, end_row, end_col, debut_col, min_val, max_val):
    """
    Calibrates the SABR parameters for volatility curve fitting using a specific optimizer and several methods
    Tests the calibration around a specified range [min_val, max_val]

    Parameters:
    - path: file name
    - start: starting date, format: YYYY-MM-DD
    - fwd: forward
    - end_row: ending date row in csv file
    - end_col: ending date column in csv file
    - debut_col: starting column for volatilities in csv file
    - min_val: minimum strike value selected for calibration
    - max_val: maximum strike value selected for calibration

    Returns:
    - alpha, beta, rho and nu fitting parameters saved in a file for each optimizer
    - images for true and fitted vols are also saved
    """
    # time to expiry
    end = rd.read_cell(path, end_row, end_col)
    t = year_fract(start, end)

    # strikes and volatilities
    data_dict = rd.csv_to_dicts(path, debut_col)
    strikes = np.array([float(key) for key in data_dict[0]])
    volatilities = np.array([float(value)/100 for value in data_dict[0].values()])
    nb = len(strikes)

    # calibration for a specific strike range: [min_val, max_val]
    strikes_reduced = np.array([strike for strike in strikes if min_val <= strike <= max_val])
    first_elt = strikes_reduced[0]
    res = np.where(strikes == first_elt)
    init_idx = res[0][0]
    last_idx = init_idx + len(strikes_reduced)
    volatilities_reduced = volatilities[init_idx:last_idx]

    # Fit the SABR model for whole strike range
    sabr_model = sb.SABRVolatility(fwd=fwd, strikes=strikes, volatilities=volatilities, T=t, nb=nb)
    print(f"Fitted parameters: alpha={sabr_model.alpha}, beta={sabr_model.beta}, rho={sabr_model.rho}, nu={sabr_model.nu}")

    # Fit the SABR model for specific strike range
    sabr_model2 = sb.SABRVolatility(fwd=fwd, strikes=strikes_reduced, volatilities=volatilities_reduced, T=t, nb=nb)
    print(f"Fitted parameters 2: alpha={sabr_model2.alpha}, beta={sabr_model2.beta}, rho={sabr_model2.rho}, nu={sabr_model2.nu}")

    # SABR model for specific strike range extended whole range
    sabr_model2_ext = sb.SABRVolatility(alpha=sabr_model2.alpha, beta=sabr_model2.beta, rho=sabr_model2.rho, nu=sabr_model2.nu, fwd=sabr_model2.fwd)

    # Plot the original and fitted volatility curves
    fitted_vols = np.array([sabr_model.implied_volatility(strike, t) for strike in strikes])
    fitted_vols2 = np.array([sabr_model2.implied_volatility(strike, t) for strike in strikes_reduced])
    fitted_vols2_ext = np.array([sabr_model2_ext.implied_volatility(strike, t) for strike in strikes])

    # Plot the calibrated SABR model for whole strike range
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, volatilities, label='True Volatilities', marker='o')
    plt.plot(strikes, fitted_vols, label='Fitted Vols', linestyle='-.', marker='s')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('True and Fitted Volatility Curves')
    plt.legend()
    plt.show()

    # Plot the calibrated SABR model for specific strike range
    plt.figure(figsize=(10, 6))
    plt.plot(strikes_reduced, volatilities_reduced, label='True Volatilities', marker='o')
    plt.plot(strikes_reduced, fitted_vols2, label='Fitted Vols', linestyle='-.', marker='s')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('True and Fitted Volatility Curves')
    plt.legend()
    plt.show()

    # Plot the calibrated SABR model for specific strike range on the whole range
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, volatilities, label='True Volatilities', marker='o')
    plt.plot(strikes, fitted_vols2_ext, label='Fitted Vols', linestyle='-.', marker='s')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('True and Fitted Volatility Curves')
    plt.legend()
    plt.show()


def main():
    # read data and get inputs
    clean_path = 'clean.csv'
    noisy_path = 'noisy_updated.csv'
    data_clean = 'Clean Data'
    data_noisy = 'Noisy Data'
    start = "2024-06-17"
    fwd = 1174.75
    end_row = 1
    end_col = 1
    debut_col = 2
    min_val_clean = 950
    max_val_clean = 1500
    min_val_noisy = 50
    max_val_noisy = 120

    # Clean data calibration
    # test_basinhopping_calibration(clean_path, data_clean, start, fwd, end_row, end_col, debut_col, min_val, max_val)
    # test_minimize_calibration(clean_path, data_clean, start, fwd, end_row, end_col, debut_col, min_val, max_val)

    # Noisy data calibration
    test_basinhopping_calibration(noisy_path, data_noisy, start, fwd, end_row, end_col, debut_col, min_val_noisy, max_val_noisy)

if __name__ == "__main__":
    main()
