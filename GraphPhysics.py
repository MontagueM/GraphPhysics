# import PIL
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import os

class Graph:
    """

    """
    def __init__(self, x_data, y_data, xaxis_title, yaxis_title, plot_title, plot_label, x_errors, y_errors):
        self.x_data = x_data
        self.y_data = y_data
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title
        self.plot_title = plot_title
        self.plot_label = plot_label
        self.x_errors = x_errors
        self.y_errors = y_errors

        self.poly1d = None
        self.gradient = None
        self.fit = None
        self.cov = None
        self.slope_uncertainty = None

    params = {
        'axes.labelsize': 20,
        'font.size': 20,
        'legend.fontsize': 10,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'figure.figsize': [10, 6],
        'font.family': "Arial"
    }
    plt.rcParams.update(params)

    def draw_plot(self):
        # print(self.x_data)
        # print(self.y_data)
        plt.plot(self.x_data, self.y_data, 'rx', label=self.plot_label, mew=2, ms=10)
        plt.errorbar(self.x_data, self.y_data, yerr=self.y_errors, xerr=self.x_errors,  fmt='o', mew=2, ms=3, capsize=4)
        # ax.plot(frame_number, find_peaks(x_mid, prominence=0.01))
        plt.xlabel(self.xaxis_title)
        plt.ylabel(self.yaxis_title)
        plt.title(self.plot_title)
        if self.plot_label is not None:
            plt.legend()
        plt.grid(True)
        # plt.xticks(sp.arange(0, int(frames/video.get(cv2.CAP_PROP_FPS)), 1))
        try:
            os.mkdir("images")
        except FileExistsError:
            print("")
        plt.savefig('images/' + self.plot_title)
        plt.show()

    def get_gradient(self):
        # w=[1/x for x in self.y_errors]
        self.fit, self.cov = sp.polyfit(self.x_data, self.y_data, 1, cov=True)  # fit, cov = sp.polyfit(self.x_data, self.y_data, 1, w=[1 / x for x in y_errors, cov=True) ''
        self.poly1d = sp.poly1d(self.fit)
        self.gradient = self.fit[0]
        self.slope_uncertainty = sp.sqrt(self.cov[0, 0])
        # wav_error = slope_uncertainty * d
        # print(wav_error)
        # wav_percent_err = wav_error / wavelength
        return self.poly1d, self.gradient, self.slope_uncertainty

    def draw_line_of_best_fit(self):
        if self.poly1d is None:
            self.get_gradient()
        plt.plot(self.x_data, self.poly1d(self.x_data))

    def get_intercept(self):
        self.fit, self.cov = sp.polyfit(self.x_data, self.y_data, 1, cov=True)  # fit, cov = sp.polyfit(self.x_data, self.y_data, 1, w=[1 / x for x in y_errors, cov=True) ''
        self.poly1d = sp.poly1d(self.fit)
        self.int_uncertainty = sp.sqrt(self.cov[1, 1])
        return self.fit[1], self.int_uncertainty


def process_data(csv_name):
    df = pd.read_csv(csv_name)
    data = []
    for col in range(0, len(df.columns), 2):
        data.append([df.iloc[:, [col]].values.T.tolist()[0], df.iloc[:, [col + 1]].values.T.tolist()[0]])
    return data[0][0], data[0][1]
