import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from matplotlib.widgets import RectangleSelector, Button
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as pe

import transformation as tf

def df_viewer(df, sensors, save_dir):
    """DataFrame viewer lets you see the results in a DataFrame and reselect
    the sigmoid regions for posterior analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing all the results to be shown
    sensors : pandas.DataFrame
        DataFrame containing the information pertaining the sensors present and
         colors to be used for representation
    save_dir : pathlib.Path
        path where to save the modified DataFrame
    """

    fig, axs = plt.subplots(5, 1,
                            figsize=(8, 10),
                            gridspec_kw={'height_ratios': [5, 5, 5, 5, 1]},
                            sharex=True)

    class SubPlot(object):
        """This subplots class keeps record of axes and data to enable fast
        replotting. More than one column is plotted per subplot.

        Attributes
        ----------
        axs : matplotlib.Axes
            Axes where to plot
        df_row : DataFrame.Series
            row of DataFrame to be ploted
        sensors : pandas.DataFrame
            DataFrame with the information of sensors suchs as names and colors
        x_col : string
            name of the column to use as x axis
        y_col_suffix : string
            suffix to be appended to the fluorophore name to get the y axis col

        Methods
        -------
        update(df_row): Replots the subplot with the given pandas.Series data.
        set_title(df_row): Retitles the plot with the given information.
        """

        def __init__(self, axs, df_row, sensors,
                     x_col='time', y_col_suffix='anisotropy'):
            self.sensors = sensors
            self.axs = axs
            self.x_col = x_col
            self.y_col_suffix = y_col_suffix

            time = df_row[self.x_col]
            mask = df_row.sigmoid_mask

            plt.sca(self.axs)

            self.lines = []
            for _, sensor_row in sensors.iterrows():
                this_y = df_row['_'.join([sensor_row.fluorophore, self.y_col_suffix])]
                l, = self.axs.plot(time, this_y, color=sensor_row.color,
                                   label=sensor_row.fluorophore)
                self.lines.append(l)

            plt.legend()
            plt.sca(self.axs)
            if len(time) == len(mask):
                self.set_title(df_row)
                y_lims = self.axs.get_ylim()
                self.fill = self.axs.fill_between(time, y_lims[0], y_lims[1],
                                                  where=mask, color='g', alpha=0.1)
            plt.xlabel('Time (min.)')
            plt.ylabel(y_col_suffix)
            plt.draw()

        def update(self, df_row):
            """Replots the subplot with the given pandas.Series data."""
            plt.sca(self.axs)
            time = df_row[self.x_col]
            mask = df_row.sigmoid_mask
            for line, (_, sensor_row) in zip(self.lines,
                                             self.sensors.iterrows()):
                this_y = df_row['_'.join([sensor_row.fluorophore, self.y_col_suffix])]
                line.set_xdata(time)
                line.set_ydata(this_y)

            self.axs.relim(visible_only=True)
            self.axs.autoscale_view()
            if len(time) == len(mask):
                y_lims = self.axs.get_ylim()
                self.fill.remove()
                self.fill = self.axs.fill_between(time, y_lims[0], y_lims[1],
                                                  where=mask, color='g', alpha=0.1)
                self.set_title(df_row)
            plt.draw()

        def set_title(self, df_row):
            """Retitles the plot with the given information."""
            ind = df_row.name
            date = df_row.date
            drug = df_row.drug
            plasmid = df_row.plasmid
            self.axs.set_title('index: %s; date: %s; drug: %s; plasmid: %s' %
                               (ind, date, drug, plasmid))

    class SimpleSubPlot(object):
        """This subplots class keeps record of axes and data to enable fast
        replotting. Only one column is plotted per subplot.

        Attributes
        ----------
        axs : matplotlib.Axes
            Axes where to plot
        df_row : DataFrame.Series
            row of DataFrame to be ploted
        x_col : string
            name of the column to use as x axis
        y_col : string
            name of the column to use as y axis

        Methods
        -------
        update(df_row): Replots the subplot with the given pandas.Series data.
        """

        def __init__(self, axs, df_row, x_col, y_col):
            self.axs = axs
            self.x_col = x_col
            self.y_col = y_col
            x = df_row[self.x_col]
            y = df_row[self.y_col]

            plt.sca(self.axs)

            self.line, = self.axs.plot(x, y, color='k')

            plt.sca(self.axs)

            plt.xlabel('Time (min.)')
            plt.ylabel(y_col)
            plt.draw()

        def update(self, df_row):
            """Replots the subplot with the given pandas.Series data."""
            plt.sca(self.axs)
            x = df_row[self.x_col]
            y = df_row[self.y_col]

            self.line.set_xdata(x)
            self.line.set_ydata(y)

            self.axs.relim(visible_only=True)
            self.axs.autoscale_view()
            plt.draw()

    subplot_anisotropy = SubPlot(axs[0], df.iloc[0], sensors)
    subplot_activity = SubPlot(axs[1], df.iloc[0], sensors,
                      x_col='Cit_time_activity_new',
                      y_col_suffix='activity_interpolate')

    subplot_area = SimpleSubPlot(axs[2], df.iloc[0], x_col='time', y_col='area')
    subplot_solidity = SimpleSubPlot(axs[3], df.iloc[0], x_col='time',
                                 y_col='solidity')

    subplots = [subplot_anisotropy,
                subplot_activity,
                subplot_area,
                subplot_solidity]

    class Index(object):
        """Index class keeps track of the index of the DataFrame that is being
        worked on and calls for the replotting of every subplot, as well as
        modifying the DataFrame.

        Attributes
        ----------
        df : pandas.DataFrame
            DataFrame to be worked on.
        subplots : list of Subplots
            list of all the subplots to be updated on each index change.

        Methods
        -------
        next(event): Next index.
        prev(event): Previous index.
        get_mask(): Generates a boolean mask from RectangleSelector.
        load(event_press, event_release): Modifies the actual mask attribute.
        add_region(event): Adds the actual_mask to the DataFrame.
        remove_region(event): Sets the boolean mask in DataFrame to all False.
        save_all(event): Saves the DataFrame to the selected path.
        """

        def __init__(self, df, subplots):
            self.df = df
            self.cur_ind = 0
            self.max_ind = len(self.df.index)
            self.this_ind = self.df.index[self.cur_ind]

            self.subplots = subplots

            self.rectangle = []
            self.extents = []
            self.actual_mask = []

        def next(self, event):
            self.cur_ind = min(self.cur_ind + 1, self.max_ind)
            self.this_ind = df.index[self.cur_ind]
            for subplot in self.subplots:
                subplot.update(df.loc[self.this_ind])

        def prev(self, event):
            self.cur_ind = max(self.cur_ind - 1, 0)
            self.this_ind = df.index[self.cur_ind]
            for subplot in self.subplots:
                subplot.update(df.loc[self.this_ind])

        def get_mask(self):
            t_ini = self.extents[0]
            t_end = self.extents[1]

            times = df.loc[self.this_ind].time

            mask = [True if t_ini <= time <= t_end else
                    False for time in times]

            return mask

        def load(self, event_press, event_release):
            self.extents = self.rectangle.extents
            self.actual_mask = self.get_mask()

        def add_region(self, event):
            ind = self.this_ind
            mask = self.actual_mask

            df.at[ind, 'sigmoid_mask'] = np.logical_or(mask,
                                                       self.df.at[ind, 'sigmoid_mask'])
            print('added')

        def remove_region(self, event):
            ind = self.this_ind
            self.df.at[ind, 'sigmoid_mask'] = np.array([False] *
                                                       len(self.df.at[ind, 'sigmoid_mask']))
            print('removed')

        def save_all(self, event):
            if save_dir is None:
                print("No save directory was specified.")
                return

            self.df.to_pickle(str(save_dir))

            print('saved')

    callback = Index(df, subplots)
    plt.sca(axs[1])

    axs[-1].axis('off')

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    rect = RectangleSelector(axs[0], callback.load)
    callback.rectangle = rect

    axadd = plt.axes([0.50, 0.05, 0.15, 0.075])
    badd = Button(axadd, 'Add Region')
    badd.on_clicked(callback.add_region)

    axremove = plt.axes([0.30, 0.05, 0.15, 0.075])
    bremove = Button(axremove, 'Remove all')
    bremove.on_clicked(callback.remove_region)

    axsave = plt.axes([0.1, 0.05, 0.1, 0.075])
    bsave = Button(axsave, 'Save All')
    bsave.on_clicked(callback.save_all)

    plt.show()


def view_curves(axs, df_row, sensors, lines=None, fill=None):
    """If there si no use for this, it is going to be deleted."""
    time = df_row.time
    mask = df_row.sigmoid_mask

    ind = df_row.index
    date = df_row.date
    drug = df_row.drug
    plasmid = df_row.plasmid
    axs.set_title('ind: %s; date: %s; drug: %s; plasmid: %s' %
                  (ind, date, drug, plasmid))

    if lines is None:
        plt.sca(axs)

        lines = []
        for _, sensor_row in sensors.iterrows():
            anis = df_row[sensor_row.fluorophore + '_anisotropy']
            l, = axs.plot(time, anis, color=sensor_row.color,
                          label=sensor_row.fluorophore)
            lines.append(l)

        plt.legend()
        plt.sca(axs)
        y_lims = axs.get_ylim()
        fill = axs.fill_between(time, y_lims[0], y_lims[1], where=mask,
                                color='g', alpha=0.1)
        plt.xlabel('Time (min.)')
        plt.ylabel('Anisotropy')
        plt.draw()

        return lines, fill

    else:
        for line, (_, sensor_row) in zip(lines, sensors.iterrows()):
            anis = df_row[sensor_row.fluorophore + '_anisotropy']
            line.set_xdata(time)
            line.set_ydata(anis)

        fill.remove()
        y_lims = axs.get_ylim()
        fill = axs.fill_between(time, y_lims[0], y_lims[1], where=mask,
                                color='g', alpha=0.1)
        plt.draw()

        return lines, fill


def plot_delta_b_histogram(df, sensors):
    """Plots an histogram for the delta b in the df. Must have a 'b' column"""
    for _, sensor_row in sensors.iterrows():
        fluo = sensor_row.fluorophore
        color = sensor_row.color
        b_values = df[fluo + '_b'].values
        b_values = b_values[np.isfinite(b_values)]

        plt.hist(b_values, bins=50, edgecolor='k',
                 color=color, alpha=0.6, label=fluo)
    plt.xlabel('$\Delta$b')
    plt.ylabel('Frequency')
    plt.legend()


def plot_histogram_2d(df, sensors, kind='histogram_2d', hue=None,
                      cols=["BFP_to_Cit", "BFP_to_Kate"]):
    """Generates a 2D hexbin histogram or scatetr of the given DataFrame and
    sensor dictionary."""
    x_data_col, y_data_col = cols

    df_fil = df.query('is_single_apoptotic')
    df_var = df_fil[[x_data_col, y_data_col]]
    df_fil = df_fil[(np.abs(stats.zscore(df_var)) < 3).all(axis=1)]
    g = sns.JointGrid(x_data_col, y_data_col, df_fil, height=6.6)

    sns.distplot(df_fil[x_data_col], kde=False, bins=30, ax=g.ax_marg_x)
    sns.distplot(df_fil[y_data_col], kde=False, bins=30, ax=g.ax_marg_y,
                 vertical=True)
    if kind == 'histogram_2d':
        g.ax_joint.hexbin(df_fil[x_data_col], df_fil[y_data_col], gridsize=30,
                          mincnt=1, cmap='Greys')
    elif kind == 'scatter':
        if hue:
            for this_var, this_df in df_fil.groupby(hue):
                g.ax_joint.scatter(this_df[x_data_col],
                                   this_df[y_data_col],
                                   label=this_var)
            plt.sca(g.ax_joint)
            plt.legend()

        else:
            g.ax_joint.scatter(df_fil[x_data_col], df_fil[y_data_col])

    else:
        raise ValueError('%s is not a permitted kind. Only "histogram_2d" or '
                         '"scatter" allowed')

    g.ax_joint.axhline(y=0, ls='--', color='k', alpha=0.3)
    g.ax_joint.axvline(x=0, ls='--', color='k', alpha=0.3)


def make_report(pdf_dir, df, sensors, hue=None,
                cols=["BFP_to_Cit", "BFP_to_Kate"]):
    """Generates a pdf with all the important results of the analysis."""
    with PdfPages(str(pdf_dir)) as pp:
        plot_delta_b_histogram(df, sensors)
        pp.savefig()
        plt.close()

        plot_histogram_2d(df, sensors, cols=cols)
        pp.savefig()
        plt.close()

        plot_histogram_2d(df, sensors, cols=cols, kind='scatter', hue=hue)
        pp.savefig()
        plt.close()


# Make KDE plots
def get_level_for_kde(x, y, level):
    """Returns Levels to be used for kde contour plotting."""
    if isinstance(level, (list, tuple)):
        vals = []
        for level in level:
            val = get_level_for_kde(x, y, level)
            vals.append(val)

    else:
        kde = stats.gaussian_kde([x, y])
        probs = kde.pdf([x, y])
        arr = np.asarray([x, y, probs]).T
        arr = arr[arr[:, 2].argsort()]
        ind = np.ceil(len(x) * level).astype(int)
        vals = arr[ind, 2]

    return vals


def plot_joint_kde(g, df, cols=["BFP_to_Cit", "BFP_to_Kate"], label=None,
                   my_lvls=[0.34, 0.68]):
    """Plots a specific df in the kdeplot.

    Parameters
    ----------
    g : seaborn.JointGrid
        Grid where to plot the kde.
    df : pandas.DataFrame
        DataFrame containing the data.
    cols : list of strings
        Names of the columns of the Dataframe to use as x and y axis.
    label : string
        label to name the plot.
    my_lvls : float or list of floats
        levels of contour curves to surround the data.
    """
    # Filter dataset
    my_df = df.dropna(subset=cols)

    # Plot marginals
    sns.kdeplot(my_df[cols[0]], shade=True, ax=g.ax_marg_x)
    sns.kdeplot(my_df[cols[1]], shade=True, ax=g.ax_marg_y, vertical=True)

    # Plot joint
    this_levels = get_level_for_kde(my_df[cols[0]],
                                    my_df[cols[1]],
                                    my_lvls)
    cs = sns.kdeplot(my_df[cols[0]], my_df[cols[1]], alpha=1,
                     levels=this_levels,
                     ax=g.ax_joint, shade_lowest=False,
                     vmin=2*this_levels[0]-this_levels[1])
    cs.collections[-1].set_label(label)

    return g


def plot_distributions(df, cols=["BFP_to_Cit", "BFP_to_Kate"], groupby=None,
                  my_lvls=[0.34, 0.68]):
    """Plots the distributions in a kde plot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data.
    cols : list of string
        names of columns in df that contain the x and y axis.
    groupby : string or list of strings
        list of columns to group the DataFrame.
    my_lvls : float or list of floats
        Contours surrounding data.
    """
    g = sns.JointGrid(cols[0], cols[1], df)

    if groupby:
        for this_var, this_df in df.groupby(groupby):
            plot_joint_kde(g, this_df, cols=cols, label=this_var,
                           my_lvls=my_lvls)
    else:
        plot_joint_kde(g, df, cols=cols, my_lvls=my_lvls)

    g.ax_joint.axvline(x=0, color='k', lw=1, ls='--', alpha=0.5)
    g.ax_joint.axhline(y=0, color='k', lw=1, ls='--', alpha=0.5)
    plt.sca(g.ax_joint)
    plt.legend(loc=3, framealpha=0)

    g.ax_marg_x.set_xlabel('')
    g.ax_marg_y.set_ylabel('')
    g.ax_marg_x.legend_.remove()
    g.ax_marg_y.legend_.remove()
    plt.tight_layout()

    return g


def plot_curves(df, sensors, normalize=False):
    """Centers the curves of the three sensors to the first one that reaches
    maximum activity. It then calculates the average curves and plots the mean
    and 25 and 75 percentiles."""
    fine_time = np.arange(-60, 60, 1)
    curves = {fluo: [] for fluo in sensors.fluorophore.values}
    for ind, row in df.iterrows():
        max_times = [row[fluo + '_max_time'] for fluo in sensors.fluorophore.values]
        max_time = np.min(max_times)
        if np.isnan(max_time):
            continue

        for fluo_ind, this_fluo in sensors.iterrows():
            fluo = this_fluo.fluorophore
            time = row.time.copy()
            anisotropy = row[fluo + '_anisotropy'].copy()

            time -= max_time
            mask = (-60 < time) & (time < 60)
            anisotropy = anisotropy[mask]
            if normalize:
                anisotropy = tf.normalize(anisotropy[mask])

            curves[fluo].append(interpolate(fine_time, time[mask], anisotropy))

            plt.plot(time[mask], anisotropy, color=this_fluo.color, alpha=0.1)

    for fluo_ind, this_fluo in sensors.iterrows():
        fluo = this_fluo.fluorophore
        anis = np.asarray(curves[fluo])
        anis_mean = np.nanmean(anis, axis=0)
        anis_perc = np.nanpercentile(anis, [25, 75], axis=0)

        plt.plot(fine_time, anis_mean, color=this_fluo.color, lw=3,
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'),
                               pe.Normal()])
        plt.fill_between(fine_time, anis_perc[0], anis_perc[1],
                         edgecolor='k', facecolor=this_fluo.color, alpha=0.7)

    plt.title('N = %s' % anis.shape[0])
    plt.xlabel('Time (min.)')
    plt.ylabel('Anisotropy')
    plt.xlim((-50, 50))
    plt.show()


def interpolate(new_time, time, curve):
    """Interpolate curve using new_time as xdata"""
    if not np.isfinite(time).all():
        return np.array([np.nan])

    curve = savgol_filter(curve, window_length=3, polyorder=1, mode='nearest')
    f = interp1d(time, curve, kind='linear', bounds_error=False)
    return f(new_time)
