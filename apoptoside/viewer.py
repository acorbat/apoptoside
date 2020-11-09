import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import seaborn as sns
from scipy import stats


def df_viewer(df, sensors, save_dir):
    print('You can view %d curves' % len(df))

    if 'solidity' in df.columns:
        fig, axs = plt.subplots(5, 1,
                                figsize=(8, 10),
                                gridspec_kw={'height_ratios': [5, 5, 5, 5, 1]},
                                sharex=True)
    else:
        fig, axs = plt.subplots(4, 1,
                                figsize=(8, 10),
                                gridspec_kw={'height_ratios': [5, 5, 5, 1]},
                                sharex=True)
    plt.subplots_adjust(hspace=0)

    class SubPlot(object):

        def __init__(self, axs, df_row, sensors,
                     x_col='time', y_col_suffix='anisotropy',
                     y_col_prefix='name'):
            self.sensors = sensors
            self.axs = axs
            self.x_col = x_col
            self.y_col_suffix = y_col_suffix
            self.y_col_prefix = y_col_prefix

            time = df_row[self.x_col]
            mask = df_row.sigmoid_mask

            plt.sca(self.axs)

            self.lines = []
            for sensor in sensors.sensors:
                this_y = df_row['_'.join([getattr(sensor, self.y_col_prefix),
                                          self.y_col_suffix])]
                l, = self.axs.plot(time, this_y, color=sensor.color,
                                   label=sensor.name)
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
            plt.sca(self.axs)
            time = df_row[self.x_col]
            mask = df_row.sigmoid_mask
            for line, sensor in zip(self.lines,
                                             self.sensors.sensors):
                this_y = df_row['_'.join([getattr(sensor, self.y_col_prefix),
                                          self.y_col_suffix])]
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
            ind = df_row.name
            date = df_row.date
            pos = df_row.position
            label = df_row.label
            drug = df_row.drug
            plasmid = df_row.plasmid
            self.axs.set_title('index: %s; date: %s; drug: %s; \n '
                               'position: %d; label: %d; plasmid: %s' %
                               (ind, date, drug, pos, label, plasmid))

    class SimpleSubPlot(object):

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
                      x_col='Cas3_time_activity_new',
                      y_col_suffix='activity_interpolate',
                               y_col_prefix='enzyme')
    subplot_area = SimpleSubPlot(axs[2], df.iloc[0], x_col='time', y_col='area')

    subplots = [subplot_anisotropy,
                subplot_activity,
                subplot_area]

    if 'solidity' in df.columns:
        subplot_solidity = SimpleSubPlot(axs[3], df.iloc[0], x_col='time',
                                     y_col='solidity')

        subplots.append(subplot_solidity)

    class Index(object):

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
        for sensor in sensors.sensors:
            anis = df_row[sensor.name + '_anisotropy']
            l, = axs.plot(time, anis, color=sensor.color,
                          label=sensor.name)
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
        for line, sensor in zip(lines, sensors.sensors):
            anis = df_row[sensor.name + '_anisotropy']
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
    for sensor in sensors.sensors:
        fluo = sensor.name
        color = sensor.color
        b_values = df[fluo + '_delta_b'].values
        b_values = b_values[np.isfinite(b_values)]

        plt.hist(b_values, bins=50, edgecolor='k',
                 color=color, alpha=0.6, label=fluo)
    plt.xlabel('$\Delta$b')
    plt.ylabel('Frequency')
    plt.legend()


def plot_histogram_2d(df, sensors, kind='histogram_2d', hue=None,
                      cols=["BFP_to_Cit", "BFP_to_Kate"]):
    """Generates a 2D histogram of the given DataFrame and sensor dictionary."""
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


def make_report(pdf_dir, df, sensors, hue=None, cols=["BFP_to_Cit", "BFP_to_Kate"]):
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


def get_level_for_kde(x, y, level):
    """Estimates the level that includes the percentage of data points
    wanted."""
    kde = stats.gaussian_kde([x, y])
    probs = kde.pdf([x, y])
    arr = np.asarray([x, y, probs]).T
    arr = arr[arr[:, 2].argsort()]
    ind = np.ceil(len(x) * level).astype(int)
    return arr[ind, 2]


def get_levels_for_kde(x, y, levels):
    """Estimates the levels that include the percentages of data points
    wanted."""
    vals = []
    for level in levels:
        val = get_level_for_kde(x, y, level)
        vals.append(val)
    return vals


def plot_kde(x, y, g, cmap='Blues_d', label=None, gridsize=60,
             levels=(0.34, 0.68)):
    """Plots a single kde plot with marginal dist plots on JointGrig g."""
    sns.kdeplot(x, y, cmap=cmap, alpha=0.6,
                # levels from seaborn now uses probability mass
                levels=levels,  # get_levels_for_kde(x, y, levels),
                ax=g.ax_joint, label=label)

    sns.distplot(x, kde=False, bins=gridsize, ax=g.ax_marg_x)
    sns.distplot(y, kde=False, bins=gridsize, ax=g.ax_marg_y,
                 vertical=True)


def plot_kdes(df, x_col, y_col, hue=None, xlim=(-20, 40), ylim=(-20, 40),
              cmaps=None, labels=None, gridsizes=None, levels=(0.34, 0.68)):
    """Generates a JointGrid plot and makes de kde plots."""
    g = sns.JointGrid(data=df, x=x_col, y=y_col,
                      xlim=xlim, ylim=ylim)

    if hue is None:
        plot_kde(df[x_col].values, df[y_col].values, g, cmap=cmaps, label=labels,
                 gridsize=gridsizes)
    else:
        for (this_var, this_df), this_cmap,  gridsize in zip(
                df.groupby(hue),
                cmaps,
                gridsizes):
            plot_kde(this_df[x_col].values, this_df[y_col].values, g,
                     cmap=this_cmap, label=this_var, gridsize=gridsize,
                     levels=levels)

    return g


def add_experimental_noise(times, cov=None):
    """Add correlated experimental noise to time differences."""
    if cov is None:
        cov = np.array([[27.4241789, 2.48916841], [2.48916841, 21.80573026]])
        cov *= 0.75

    times += np.random.multivariate_normal([0, 0], cov, times.shape[0])
    return times
