from typing import List
import csv

def write_loss_to_file(loss_list: list[float]):
    """ Writes the loss over gradient descent iterations to a csv file

        Parameters:
            loss_list: a list of floats representing the loss
                at each iteration of gradient descent
    """

    with open('gradient_descent_loss.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write data rows
        for value in loss_list:
            writer.writerow([value])