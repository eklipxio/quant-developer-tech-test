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


def build_vol_surface():
    clean_path = 'clean.csv'
    start = "2024-06-17"
    F0 = 1174.75
    interv=500

    # time to expiry
    end = rd.read_cell(clean_path, 1, 1)
    t = year_fract(start, end)

    # strikes and volatilities
    clean = rd.csv_to_dicts(clean_path, 2)
    strikes = np.array([float(key) for key in clean[0]])
    volatilities = np.array([float(value) for value in clean[0].values()])

    # Fit the SABR model using the class method
    sabr_model = sb.SABRVolatility(fwd=F0, strikes=strikes, volatilities=volatilities, T=t, interv=interv)
    print(
        f"Fitted parameters: alpha={sabr_model.alpha}, beta={sabr_model.beta}, rho={sabr_model.rho}, nu={sabr_model.nu}")

    # Plot the original and fitted volatility curves
    fitted_vols = np.array([sabr_model.implied_volatility(strike, t) for strike in strikes])


    plt.figure(figsize=(10, 6))
    plt.plot(strikes, volatilities, label='True Volatilities', marker='o')
    plt.plot(strikes, fitted_vols, label='Fitted Volatilities', linestyle='-.', marker='s')
    plt.xlabel('Strike')
    plt.ylabel('Volatility')
    plt.title('True and Fitted Volatility Curves')
    plt.legend()
    plt.show()


def main():
    build_vol_surface()


if __name__ == "__main__":
    main()
