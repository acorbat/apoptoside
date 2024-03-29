from itertools import combinations

import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.interpolate import interp1d

from . import anisotropy_functions as af
from . import experimental_tests as et
from . import filter_data as fd
from . import transformation as tf
from .sensors import Sensors
from . import viewer as vw


class Apop(object):
    """Apop class is the scaffold containing and organizing the experimental
    dataframe.

    Attributes
    ----------
    df : pandas.DataFrame
        pandas DataFrame containing all the data from the experiments and
        analysis performed
    sensors : apoptoside.sensors.Sensors
        Sensors class containing different Sensor for the experiment
    save_dir : pathlib.Path
        Path object where the modified DataFrame should be saved

    Methods
    -------
    load_df : Pandas.DataFrame or str or pathlib.Path
        Loads DataFrame into Apoptoside.df. If it's a Pandas.DataFrame it is
        loaded directly, else it is considered a string and loaded from pickle.
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

    def __init__(self, data=None, sim_data=None, save_dir=None, sensors_path=None):

        if data is not None:
            self.load_df(data)
        if sensors_path is None:
            self.sensors = pd.DataFrame(columns=['fluorophore', 'delta_b', 'color',
                                             'caspase'])
        else:
            self.load_sensors(sensors_path)
        if sim_data is not None:
            self.load_sim(sim_data)
        self.refer_to = 'Cas3'
        self.time_diff_cols = []
        self.save_dir = save_dir

    def load_df(self, data):
        """Loads DataFrame into Apoptoside.df. If it's a Pandas.DataFrame it is
        loaded directly, else it is considered a string and loaded from pickle."""
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            self.df = pd.read_pickle(str(data))

    def load_sim(self, results):
        self.df = results
        try:
            self.df['stimuli'] = self.df.apply(
                lambda x: 'intrinsic' if x['IntrinsicStimuli_0'] != 0 else 'extrinsic',
                axis=1)
        except KeyError:
            self.df['stimuli'] = 'extrinsic'
        self.df['sigmoid_mask'] = self.df.apply(
            lambda row: [True] * len(row['BFP_anisotropy']),
            axis=1)
        self.df['name'] = 'simulation'
        self.add_is_apoptotic()
        self.df['time'] = self.df.apply(
            lambda row: np.arange(0, len(row['BFP_anisotropy']) / 60, 1 / 60),
            axis=1)
        self.add_delta_b(fix_b=True)
        self.add_activities()
        self.add_max_times('time', 'activity', single_time_col=True)
        self.add_time_differences(refer_to='Cas3')
        self.add_activity_widths('time_activity', 'activity', single_time_col=False)

    def load_sensors(self, path):
        """Load sensors from file."""
        self.sensors = Sensors.parse_file(path)

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

        for sensor in self.sensors.sensors:
            fluo = sensor.name
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
        for sensor in self.sensors.sensors:
            fluo = sensor.name
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
            apoptotic_columns = [sensor.name + '_apoptotic'
                                 for sensor in self.sensors.sensors]
            if not all([col in self.df.columns for col in apoptotic_columns]):
                print('filter_non_apoptotic method has not been run on '
                      'DataFrame.')
                return

            condition = ' and '.join([sensor.name + '_apoptotic'
                                      for sensor in self.sensors.sensors])
            vw.df_viewer(self.df.query(condition), self.sensors, self.save_dir)

        else:
            vw.df_viewer(self.df, self.sensors, self.save_dir)

    def add_is_apoptotic(self):
        """Adds a column of bools showing whether there is an apoptotic region
        in the curve."""
        self.df['is_apoptotic'] = self.df.sigmoid_mask.apply(lambda x: any(x))

        self.df['is_single_apoptotic'] = self.df.sigmoid_mask.apply(
            lambda x: tf.single_apoptotic(x)
        )

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
        for sensor in self.sensors.sensors:
            fluo = sensor.name
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
        for sensor in self.sensors.sensors:
            fluo = sensor.name
            self.df[name_col(fluo, 'fluo', col_end)] = self.df.apply(
                lambda x: af.Fluos_FromInt(
                    x[name_col(fluo, 'parallel', col_end)],
                    x[name_col(fluo, 'perpendicular', col_end)]),
                axis=1)

    def add_delta_b(self, estimator='mean', fix_b=False):
        """Adds an estimation of delta b parameter column.

        Parameters
        ----------
        estimator : string
            Which statistic estimator to use
        fix_b : boolean (default: False)
            If True, b is set from sensors dictionary
        """
        if fix_b:
            for sensor in self.sensors.sensors:
                fluo = sensor.name
                self.df['_'.join([fluo, 'delta_b'])] = sensor.delta_b

        else:
            for sensor in self.sensors.sensors:
                fluo = sensor.name
                self.df[name_col(fluo, 'delta_b')] = self.df.apply(
                    lambda x: tf.estimate_delta_b(
                        x[name_col(fluo, 'fluo', estimator, 'pre')],
                        x[name_col(fluo, 'fluo', estimator, 'pos')]),
                    axis=1)

    def add_activities(self):
        """Adds the activity column for each fluorophore using the found b
        parameter for each row."""
        for sensor in self.sensors.sensors:
            fluo = sensor.name
            casp = sensor.enzyme
            self.df = self.df.join(self.df.apply(
                lambda x: pd.Series(dict(zip([name_col(casp, 'time_activity'), name_col(casp, 'activity')],
                                             self._calculate_activity(
                                                 x['time'],
                                                 x[name_col(fluo, 'anisotropy')],
                                                 x['sigmoid_mask'],
                                                 x[name_col(fluo, 'delta_b')]
                                             )))),
                axis=1))

    def add_monomer_curves(self):
        """Adds the monomer_curve column for each fluorophore using the found b
        parameter for each row."""
        for sensor in self.sensors.sensors:
            fluo = sensor.name
            casp = sensor.enzyme
            self.df = self.df.join(self.df.apply(
                lambda x: pd.Series(dict(zip([name_col(casp, 'time_monomer'),
                                              name_col(casp,
                                                       'monomer_fraction')],
                                             self._calculate_monomer(
                                                 x['time'],
                                                 x[name_col(fluo, 'anisotropy')],
                                                 x['sigmoid_mask'],
                                                 x[name_col(fluo, 'delta_b')]
                                             )))),
                axis=1))

    def add_interpolation(self, new_time_col, time_col, curve_col,
                          all_fluo=False, all_casp=False):
        """Add an interpolation column.

        Parameters
        ----------
        new_time_col : string
            name of the column with the new time vector to use
        time_col : string
            name of the column with the old time vector
        curve_col : string
            name of the column with the old data to be interpolated
        all_fluo : bool (optional, default=False)
            If True, adds the fluorophore at the beginning of every column
        all_casp : bool (optional, default=False)
            If True, adds the caspase at the beginning of every column
        """
        if all_fluo or all_casp:
            names = [sensor.name if all_fluo else sensor.enzyme
                     for sensor in self.sensors.sensors]

            for fluo in names:
                fluo_new_time_col = name_col(fluo, new_time_col)
                fluo_time_col = name_col(fluo, time_col)
                fluo_curve_col = name_col(fluo, curve_col)
                self.add_interpolation(fluo_new_time_col,
                                       fluo_time_col,
                                       fluo_curve_col,
                                       all_fluo=False)
        else:
            self.df[name_col(curve_col, 'interpolate')] = self.df.apply(
                lambda x: tf.interpolate(
                    x[new_time_col],
                    x[time_col],
                    x[curve_col]
                ),
                axis=1
            )

    def add_times(self, time_col, time_step, all_fluo=False, all_casp=False):
        """Generate a new time vector with another time_step.

        Parameters
        ----------
        time_col : string
            Name of the column from which to generate the new time vector.
        time_step : float
            Time step to be used for the new vector.
        all_fluo : bool (optional, default=False)
            If True, adds the fluorophore at the beginning of every column
        all_casp : bool (optional, default=False)
            If True, adds the caspase at the beginning of every column
        """
        if all_fluo or all_casp:
            names = [sensor.name if all_fluo else sensor.enzyme
                     for sensor in self.sensors.sensors]

            for fluo in names:
                fluo_time_col = name_col(fluo, time_col)
                self.add_times(fluo_time_col, time_step, all_fluo=False)

        else:
            self.df[name_col(time_col, 'new')] = self.df.apply(
                lambda x: self._generate_time_vector(
                    x[time_col],
                    time_step
                ),
                axis=1
            )

    def add_max_times(self, time_col, curve_col, single_time_col=False):
        """Add the time of the maximum of curve"""
        for sensor in self.sensors.sensors:
            fluo = sensor.name
            casp = sensor.enzyme
            if single_time_col:
                this_time_col = time_col
            else:
                this_time_col = name_col(fluo, time_col)
                if this_time_col not in self.df.columns:
                    this_time_col = name_col(casp, time_col)
            self.df[name_col(casp, 'max_time')] = self.df.apply(
                lambda x: x[this_time_col][np.argmax(x[name_col(casp, curve_col)])],
                axis=1
            )

    def add_activity_widths(self, time_col, curve_col, single_time_col=False):
        """Add the time of the maximum of curve"""
        for sensor in self.sensors.sensors:
            fluo = sensor.name
            casp = sensor.enzyme
            if single_time_col:
                this_time_col = time_col
            else:
                this_time_col = name_col(fluo, time_col)
                if this_time_col not in self.df.columns:
                    this_time_col = name_col(casp, time_col)
            self.df[name_col(casp, 'act_width')] = self.df.apply(
                lambda x: self._get_activity_width(x[this_time_col],
                                                   x[name_col(casp, curve_col)],
                                                   x[name_col(casp, 'max_time')]
                                                   ),
                axis=1
            )

    def add_time_differences(self, refer_to=None):
        """Adds a column with the time differences between max activities.
        refer_to parameter can be used to refer every time to one specific
        sensor"""
        caspases = [sensor.enzyme for sensor in self.sensors.sensors]

        if refer_to:
            ind = list(caspases).index(refer_to)
            get = caspases[ind], caspases[-1]
            caspases[-1], caspases[ind] = get

        self.time_diff_cols = []
        for casp1, casp2 in combinations(caspases, 2):
                self.time_diff_cols.append(name_col(casp1, 'to', casp2))
                self.df[name_col(casp1, 'to', casp2)] = self.df.apply(
                    lambda x: x[name_col(casp1, 'max_time')] - x[name_col(casp2, 'max_time')],
                    axis=1
                )

    def make_reports(self, base_pdf_dir, groupby, hue=None):
        """Makes the pdf reports grouping by every var at groupby and adding
        their case at the name."""
        for this_var, this_df in self.df.groupby(groupby):
            this_var = list([this_var]) if isinstance(this_var, str) else list(this_var)
            this_var.append(base_pdf_dir.name)
            pdf_dir = base_pdf_dir.with_name('_'.join(this_var))

            pdf_dir.parent.mkdir(parents=True, exist_ok=True)

            vw.make_report(pdf_dir, this_df, self.sensors, hue=hue,
                           cols=self.time_diff_cols[-2:])

    def add_test_dynamics(self):
        self.df = self.df.join((self.df.apply(et.test_dynamics, axis=1)))

    def _get_activity_width(self, time, activity, max_time):
        """Calculates full width at half maximum of activity.

        Parameters
        ----------
        time: vector
            Time vector of data
        activity: vector
            Activity vector
        max_time: float
            Time of maximum activity

        Returns
        -------
        width: float
            Full width at half maximum of activity
        """
        max_ind = np.where(time == max_time)[0][0]

        width, width_height, left_ind, right_ind = sg.peak_widths(
            [0] + list(activity), [max_ind + 1])

        f = interp1d(np.arange(len(time)), time, kind='linear')

        if left_ind < 1:
            left_ind = 1

        left_time = f(left_ind - 1)
        right_time = f(right_ind - 1)

        width = right_time - left_time

        return width[0]

    def _generate_time_vector(self, time, time_step):
        """Generates a new time vector with same start as time, until almost
        end with time_step separation."""
        if np.isfinite(time).all():
            start = np.nanmin(time)
            end = np.nanmax(time)
            return np.arange(start, end, time_step)
        else:
            return np.array([np.nan])

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

    def _calculate_monomer(self, time, anisotropy, mask, delta_b):
        """Masks the anisotropy curve and calculates the corresponding
        enzimatic activity"""
        time = self._mask(time, mask)
        anisotropy = self._mask(anisotropy, mask)

        if not any(np.isfinite(time)):
            return np.asarray(time), np.asarray(anisotropy)

        return np.asarray(time), tf.calculate_monomer_curve(anisotropy, delta_b)


def name_col(*args):
    """Joins strings with underscore to create a column name"""
    return '_'.join(list(args))
