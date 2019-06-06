import pandas as pd

import filter_data as fd


class Apop(object):

    def __init__(self, data_path):

        self.df = pd.read_pickle(str(data_path))
        self.sensors = pd.DataFrame(columns=['fluorophore', 'b', 'color'])

    def add_window_fit(self, xdata_column='time', ydata_column='anisotropy'):

        for fluo in self.sensors.fluorophore:
            anis_col = '_'.join([fluo, ydata_column])

            self.df['_'.join([fluo, 'sigmoid_popts'])] = self.df.apply(lambda row: fd.window_fit(fd.sigmoid,
                                                                                                 row[anis_col],
                                                                                                 row[xdata_column],
                                                                                                 windowsize=50,
                                                                                                 windowstep=30),
                                                                       axis=1)
