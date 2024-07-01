import matplotlib.pyplot as plt


# Create the plot
def plot_series(x, y, title="Plot", xl="X-axis", yl="Y-axis", fx=True):
    # Enable interactive mode

    plt.clf()
    plt.plot(x, y)

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xl)
    plt.ylabel(yl)

    # Display the plot
    # plt.grid(True)

    # Keep the plot open while the script continues running
    if fx:
        plt.ion()
        plt.draw()
        plt.pause(1)
        input("Press [enter] to continue.")
    else:
        plt.show()


def Test():
    # Example data
    x = [1, 2.3, 3.4, 4.9, 5.2]
    y = [2, 3, 5, 7, 11]

    plot_series(x, y, "Graph", "X","Y", False)


def main():
    # Example data
    Test()


if __name__ == "__main__":
    main()
