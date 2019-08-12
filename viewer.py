import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from matplotlib.widgets import RectangleSelector, Button
from matplotlib.backends.backend_pdf import PdfPages


def df_viewer(df, sensors, save_dir):

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

    class SubPlot(object):

        def __init__(self, axs, df_row, sensors):
            self.sensors = sensors
            self.axs = axs
            time = df_row.time
            mask = df_row.sigmoid_mask

            plt.sca(self.axs)

            self.lines = []
            for _, sensor_row in sensors.iterrows():
                anis = df_row[sensor_row.fluorophore + '_anisotropy']
                l, = self.axs.plot(time, anis, color=sensor_row.color,
                                   label=sensor_row.fluorophore)
                self.lines.append(l)

            plt.legend()
            plt.sca(self.axs)
            self.set_title(df_row)
            y_lims = self.axs.get_ylim()
            self.fill = self.axs.fill_between(time, y_lims[0], y_lims[1],
                                              where=mask, color='g', alpha=0.1)
            plt.xlabel('Time (min.)')
            plt.ylabel('Anisotropy')
            plt.draw()

        def update(self, df_row):
            plt.sca(self.axs)
            time = df_row.time
            mask = df_row.sigmoid_mask
            for line, (_, sensor_row) in zip(self.lines,
                                             self.sensors.iterrows()):
                anis = df_row[sensor_row.fluorophore + '_anisotropy']
                line.set_xdata(time)
                line.set_ydata(anis)

            self.axs.relim(visible_only=True)
            self.axs.autoscale_view()
            y_lims = self.axs.get_ylim()
            self.fill.remove()
            self.fill = self.axs.fill_between(time, y_lims[0], y_lims[1],
                                              where=mask, color='g', alpha=0.1)
            self.set_title(df_row)
            plt.draw()

        def set_title(self, df_row):
            ind = df_row.name
            date = df_row.date
            drug = df_row.drug
            plasmid = df_row.plasmid
            self.axs.set_title('index: %s; date: %s; drug: %s; plasmid: %s' %
                               (ind, date, drug, plasmid))

    subplot = SubPlot(axs[0], df.iloc[0], sensors)

    class Index(object):

        def __init__(self, df):
            self.df = df
            self.cur_ind = 0
            self.max_ind = len(self.df.index)
            self.this_ind = self.df.index[self.cur_ind]

            self.rectangle = []
            self.extents = []
            self.actual_mask = []

        def next(self, event):
            self.cur_ind = min(self.cur_ind + 1, self.max_ind)
            self.this_ind = df.index[self.cur_ind]
            subplot.update(df.loc[self.this_ind])

        def prev(self, event):
            self.cur_ind = max(self.cur_ind - 1, 0)
            self.this_ind = df.index[self.cur_ind]
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

    callback = Index(df)
    plt.sca(axs[1])

    axs[1].axis('off')

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


def plot_histogram_2d(df, sensors):
    """Generates a 2D histogram of the given DataFrame and sensor dictionary."""
    x_data_col = "BFP_to_Cit"
    y_data_col = "BFP_to_Kate"

    df_fil = df.query('is_single_apoptotic')
    df_fil = df_fil[[x_data_col, y_data_col]]
    df_fil = df_fil[(np.abs(stats.zscore(df_fil)) < 3).all(axis=1)]
    g = sns.JointGrid(x_data_col, y_data_col, df_fil, height=6.6)

    sns.distplot(df_fil[x_data_col], kde=False, bins=30, ax=g.ax_marg_x)
    sns.distplot(df_fil[y_data_col], kde=False, bins=30, ax=g.ax_marg_y,
                 vertical=True)
    g.ax_joint.hexbin(df_fil[x_data_col], df_fil[y_data_col], gridsize=30,
                      mincnt=1, cmap='Greys')
    g.ax_joint.axhline(y=0, ls='--', color='k', alpha=0.3)
    g.ax_joint.axvline(x=0, ls='--', color='k', alpha=0.3)


def make_report(pdf_dir, df, sensors):
    with PdfPages(str(pdf_dir)) as pp:
        plot_delta_b_histogram(df, sensors)
        pp.savefig()
        plt.close()

        plot_histogram_2d(df, sensors)
        pp.savefig()
        plt.close()
