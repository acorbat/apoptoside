import pandas as pd
import numpy as np

import filter_data as fd
import viewer as vw
import transformation as tf
import anisotropy_functions as af


class Apop(object):

    def __init__(self, data_path, save_dir=None):

        self.df = pd.read_pickle(str(data_path))
        self.sensors = pd.DataFrame(columns=['fluorophore', 'b', 'color'])
        self.save_dir = save_dir

    def add_window_fit(self, xdata_column='time', ydata_column='anisotropy'):

        for fluo in self.sensors.fluorophore:
            anis_col = name_col(fluo, ydata_column)

            self.df[name_col(fluo, 'sigmoid_popts')] = self.df.apply(
                lambda row: fd.window_fit(fd.sigmoid,
                                          row[anis_col],
                                          x=row[xdata_column],
                                          windowsize=50,
                                          windowstep=30),
                axis=1)

    def add_sigmoid_mask(self):

        self.df['sigmoid_mask'] = self.df.apply(
            lambda row: fd.is_apoptotic_region([np.nan] * len(row['time']),
                                               row['Cit_sigmoid_popts']),
            axis=1)

    def filter_non_apoptotic(self):

        for fluo in self.sensors.fluorophore:
            self.df[fluo + '_apoptotic'] = self.df[
                name_col(fluo, 'sigmoid_popts')].apply(
                lambda x: any([fd.is_apoptotic(*popt) for popt in x]))

    def view(self, skip_non_apoptotic=False):
        if skip_non_apoptotic:
            apoptotic_columns = [fluo + '_apoptotic'
                                 for fluo in self.sensors.fluorophore]
            if not all([col in self.df.columns for col in apoptotic_columns]):
                print('filter_non_apoptotic method has not been run on '
                      'DataFrame.')
                return

            condition = ' and '.join([fluo + '_apoptotic'
                                      for fluo in self.sensors.fluorophore])
            vw.df_viewer(self.df.query(condition), self.sensors, self.save_dir)

        else:
            vw.df_viewer(self.df, self.sensors, self.save_dir)

    def add_is_apoptotic(self):
        self.df['is_apoptotic'] = self.df.sigmoid_mask.apply(lambda x: any(x))

    def estimate_pre_and_pos(self, col, length=5):
        moments = ['pre', 'pos']

        for moment in moments:
            self.df[name_col(col, moment)] = self.df.apply(
                lambda x: np.nanmean(tf.get_region(
                    moment, x[col], x['sigmoid_mask'], length=length)),
                axis=1)

    def add_pre_pos_intensity(self, estimator='mean'):
        for fluo in self.sensors.fluorophore:
            for orientation in ['parallel', 'perpendicular']:
                self.estimate_pre_and_pos(name_col(fluo, orientation, estimator))

    def add_fluo_int(self, col_end):
        for fluo in self.sensors.fluorophore:
            self.df[name_col(fluo, 'fluo', col_end)] = self.df.apply(
                lambda x: af.Fluos_FromInt(
                    x[name_col(fluo, 'parallel', col_end)],
                    x[name_col(fluo, 'perpendicular', col_end)]),
                axis=1)

    def add_delta_b(self, estimator='mean'):
        for fluo in self.sensors.fluorophore:
            self.df[name_col(fluo, 'b')] = self.df.apply(
                lambda x: tf.estimate_delta_b(
                    x[name_col(fluo, 'fluo', estimator, 'pre')],
                    x[name_col(fluo, 'fluo', estimator, 'pos')]),
                axis=1)


def name_col(*args):
    return '_'.join(list(args))
