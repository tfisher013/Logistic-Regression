import matplotlib.pyplot as plt
import csv

def plot_convergence_by_step_size():
    """ Plots 3 rates of convergence for different step sizes on the same graph
    """

    # read in loss data
    loss_eta_point_one = []
    with open('plots/gradient_descent_loss-eta-0.1.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            loss_eta_point_one.append(float(row[0]))

    loss_eta_one = []
    with open('plots/gradient_descent_loss-eta-1.0.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            loss_eta_one.append(float(row[0]))

    loss_eta_two = []
    with open('plots/gradient_descent_loss-eta-2.0.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            loss_eta_two.append(float(row[0]))

    # plot convergence rates
    plt.plot(range(len(loss_eta_point_one)), loss_eta_point_one, label='η = 0.1', color='blue')
    plt.plot(range(len(loss_eta_one)), loss_eta_one, label='η = 1.0', color='red')
    plt.plot(range(len(loss_eta_two)), loss_eta_two, label='η = 2.0', color='green')

    # format and show plot
    plt.title('Change in Convergence Rate with Step Size')
    plt.xlabel('# Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_convergence_by_step_size()
