import numpy as np
import matplotlib.pyplot as plt
    
class ExperimentResult:
    """
    Class to store and manage the results of experiments.
    """
    def __init__(self, concI0_arr, kobs_arr, fit_output):
        raise NotImplementedError()
        self.concI0_arr = concI0_arr
        self.kobs_arr = kobs_arr
        self.fit_output = fit_output

    def plot_results(self):
        fig = plt.figure(figsize=(8, 4))
        plt.plot(self.concI0_arr, self.kobs_arr, label="data")
        plt.plot(self.fit_output.userdata["x"], self.fit_output.best_fit, label="fit", ls='--', color="black")
        plt.legend()
        plt.show()