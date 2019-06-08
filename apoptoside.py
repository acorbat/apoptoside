import pandas as pd
import numpy as np

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

    def add_sigmoid_mask(self):

        self.df['sigmoid_mask'] = self.df.apply(lambda row: fd.is_apoptotic_region([np.nan] * len(row['time']),
                                                                                   row['Cit_sigmoid_popts']),
                                                axis=1)

    def filter_non_apoptotic(self):

        for fluo in self.sensors.fluorophore:
            self.df[fluo + '_apoptotic'] = self.df['_'.join([fluo, 'sigmoid_popts'])].apply(
                lambda x: any([fd.is_apoptotic(*popt) for popt in x]))
