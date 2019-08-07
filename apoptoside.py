import pandas as pd
import numpy as np

from itertools import combinations

import filter_data as fd
import viewer as vw
import transformation as tf
import anisotropy_functions as af


class Apop(object):
    """Apop class is the scaffold containing and organizing the experimental
    dataframe.

    Attributes
    ----------
    df : pandas.DataFrame
        pandas DataFrame containing all the data from the experiments and
        analysis performed
    sensors : pandas.DataFrame
        pandas DataFrame containing the information regarding the sensors
        present in the experiment
    save_dir : pathlib.Path
        Path object where the modified DataFrame should be saved

    Methods
    -------
    add_window_fit : Performs windowed fit over the rows of each fluorophore
        in the DataFrame and adds a column with the parameters for each window.
    add_sigmoid_mask : Adds column with the automatic sigmoid mask for each
        section according to the parameters from the window fit.
    filter_non_apoptotic : Adds a columnof bools specifying whther the row has
        any apoptotic region
    view : Calls the viewer to manually curate the data and select the
        apoptotic regions.
    add_is_apoptotic : Adds a column of bools showing whether there is an
        apoptotic region in the curve.
    estimate_pre_and_pos : Adds a column with the pre and post estimates of col
        using the sigmoid_mask column.
    add_pre_pos_intensity : Adds the pre and pos intensity estimation for each
        fluorophore for parallel and perpendicular intensities.
    add_fluo_int : Adds a column for the total fluorescence intensity from the
        columns ending in col_end, starting with fluo and having orientation in
        the middle.
    add_delta_b : Adds an estimation of delta b parameter column.
    add_activities : Adds the activity column for each fluorophore using the
        found b parameter for each row.
    add_times : Generate a new time vector with another time_step.
    add_interpolation : Add an interpolation column.
    """

    def __init__(self, data_path, save_dir=None):

        self.df = pd.read_pickle(str(data_path))
        self.sensors = pd.DataFrame(columns=['fluorophore', 'b', 'color'])
        self.save_dir = save_dir

    def add_window_fit(self, xdata_column='time', ydata_column='anisotropy'):
        """Performs windowed fit over the rows of each fluorophore in the
        DataFrame and adds a column with the parameters for each window.

        Parameters
        ----------
        xdata_column : string
            Name of the xdata column
        ydata_column : string
            Name of the ydata column
        """

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
        """Adds column with the automatic sigmoid mask for each section
        according to the parameters from the window fit."""
        # TODO: This is not general enough
        self.df['sigmoid_mask'] = self.df.apply(
            lambda row: fd.is_apoptotic_region([np.nan] * len(row['time']),
                                               row['Cit_sigmoid_popts']),
            axis=1)

    def filter_non_apoptotic(self):
        """Adds a columnof bools specifying whther the row has any apoptotic
        region"""
        for fluo in self.sensors.fluorophore:
            self.df[fluo + '_apoptotic'] = self.df[
                name_col(fluo, 'sigmoid_popts')].apply(
                lambda x: any([fd.is_apoptotic(*popt) for popt in x]))

    def view(self, skip_non_apoptotic=False):
        """Calls the viewer to manually curate the data and select the
        apoptotic regions.

        Parameters
        ----------
        skip_non_apoptotic : Bool (optional, default=False)
            if True, only curves with apoptotic regions are shown.
        """
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
        """Adds a column of bools showing whether there is an apoptotic region
        in the curve."""
        self.df['is_apoptotic'] = self.df.sigmoid_mask.apply(lambda x: any(x))

    def estimate_pre_and_pos(self, col, length=5):
        """Adds a column with the pre and post estimates of col using the
        sigmoid_mask column.

        Parameters
        ----------
        col : string
            Name of the column to get the values for the estimation
        length : int (optional, default=5)
            Number of points to use for the estimation. Should be odd.
        """
        moments = ['pre', 'pos']

        for moment in moments:
            self.df[name_col(col, moment)] = self.df.apply(
                lambda x: np.nanmean(tf.get_region(
                    moment, x[col], x['sigmoid_mask'], length=length)),
                axis=1)

    def add_pre_pos_intensity(self, estimator='mean'):
        """Adds the pre and pos intensity estimation for each fluorophore for
         parallel and perpendicular intensities.

        Parameters
        ----------
        estimator : string (optional, default='mean')
            Which statistical estimator to use
        """
        for fluo in self.sensors.fluorophore:
            for orientation in ['parallel', 'perpendicular']:
                self.estimate_pre_and_pos(name_col(fluo, orientation, estimator))

    def add_fluo_int(self, col_end):
        """Adds a column for the total fluorescence intensity from the columns
        ending in col_end, starting with fluo and having orientation in the
        middle.

        Parameters
        ----------
        col_end : string
            ending of the column name to use
        """
        for fluo in self.sensors.fluorophore:
            self.df[name_col(fluo, 'fluo', col_end)] = self.df.apply(
                lambda x: af.Fluos_FromInt(
                    x[name_col(fluo, 'parallel', col_end)],
                    x[name_col(fluo, 'perpendicular', col_end)]),
                axis=1)

    def add_delta_b(self, estimator='mean'):
        """Adds an estimation of delta b parameter column.

        Parameters
        ----------
        estimator : string
            Which statistic estimator to use
        """
        for fluo in self.sensors.fluorophore:
            self.df[name_col(fluo, 'b')] = self.df.apply(
                lambda x: tf.estimate_delta_b(
                    x[name_col(fluo, 'fluo', estimator, 'pre')],
                    x[name_col(fluo, 'fluo', estimator, 'pos')]),
                axis=1)

    def add_activities(self):
        """Adds the activity column for each fluorophore using the found b
        parameter for each row."""
        for fluo in self.sensors.fluorophore:
            self.df[name_col(fluo, 'time_activity')], self.df[name_col(fluo, 'activity')] = zip(*self.df.apply(
                lambda x: self._calculate_activity(
                    x['time'],
                    x[name_col(fluo, 'anisotropy')],
                    x['sigmoid_mask'],
                    x[name_col(fluo, 'b')]
                ),
                axis=1
            ))

    def add_interpolation(self, new_time_col, time_col, curve_col):
        """Add an interpolation column.

        Parameters
        ----------
        new_time_col : string
            name of the column with the new time vector to use
        time_col : string
            name of the column with the old time vector
        curve_col : string
            name of the column with the old data to be interpolated
        """
        for fluo in self.sensors.fluorophore:
            self.df[name_col(fluo, curve_col, 'interpolate')] = self.df.apply(
                lambda x: tf.interpolate(
                    x[new_time_col],
                    x[time_col],
                    x[curve_col]
                ),
                axis=1
            )

    def add_times(self, time_col, time_step):
        """Generate a new time vector with another time_step."""
        self.df[name_col('new', time_col)] = self.df.apply(
            lambda x: self._generate_time_vector(
                x[time_col],
                time_step
            ),
            axis=1
        )

    def add_max_times(self):
        pass

    def add_time_differences(self):
        for fluo1, fluo2 in combinations(self.sensors.fluorophore.values):
            pass


    def _generate_time_vector(self, time, time_step):
        """Generates a new time vector with same start as time, until almost
        end with time_step separation."""
        start = np.nanmin(time)
        end = np.nanmax(time)
        return np.arange(start, end, time_step)

    def _mask(self, curve, mask):
        """Returns a masked array or nans if there is no True  in mask, or mask
         is discontinous"""
        if not any(mask):
            return [np.nan] * len(curve)

        inds = np.where(mask)[0]

        if (np.diff(inds) != 1).any():
            return np.asarray([np.nan] * len(curve))

        return curve[mask]

    def _calculate_activity(self, time, anisotropy, mask, delta_b):
        """Masks the anisotropy curve and calculates the corresponding
        enzimatic activity"""
        time = self._mask(time, mask)
        anisotropy = self._mask(anisotropy, mask)

        if not any(np.isfinite(time)):
            return np.asarray(time), np.asarray(anisotropy)

        return tf.calculate_activity(time, anisotropy, delta_b)


def name_col(*args):
    """Joins strings with underscore to create a column name"""
    return '_'.join(list(args))
