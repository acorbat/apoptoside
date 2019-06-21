import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button


def df_viewer(df, sensors):

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
                l, = self.axs.plot(time, anis, color=sensor_row.color, label=sensor_row.fluorophore)
                self.lines.append(l)

            plt.legend()
            plt.sca(self.axs)
            self.set_title(df_row)
            y_lims = self.axs.get_ylim()
            self.fill = self.axs.fill_between(time, y_lims[0], y_lims[1], where=mask, color='g', alpha=0.1)
            plt.xlabel('Time (min.)')
            plt.ylabel('Anisotropy')
            plt.draw()

        def update(self, df_row):
            plt.sca(self.axs)
            time = df_row.time
            mask = df_row.sigmoid_mask
            for line, (_, sensor_row) in zip(self.lines, self.sensors.iterrows()):
                anis = df_row[sensor_row.fluorophore + '_anisotropy']
                line.set_xdata(time)
                line.set_ydata(anis)

            self.fill.remove()
            self.axs.relim()
            self.axs.autoscale_view()
            y_lims = self.axs.get_ylim()
            self.fill = self.axs.fill_between(time, y_lims[0], y_lims[1], where=mask, color='g', alpha=0.1)
            self.set_title(df_row)
            plt.draw()

        def set_title(self, df_row):
            date = df_row.date
            drug = df_row.drug
            plasmid = df_row.plasmid
            self.axs.set_title('date: %s; drug: %s; plasmid: %s' % (date, drug, plasmid))

    subplot = SubPlot(axs[0], df.iloc[0], sensors)

    class Index(object):
        index = df.index
        cur_ind = 0
        max_ind = len(df.index)
        this_ind = df.index[cur_ind]

        def next(self, event):
            self.cur_ind = min(self.cur_ind + 1, self.max_ind)
            self.this_ind = df.index[self.cur_ind]
            subplot.update(df.loc[self.this_ind])

        def prev(self, event):
            self.cur_ind = max(self.cur_ind - 1, 0)
            self.this_ind = df.index[self.cur_ind]
            subplot.update(df.loc[self.this_ind])

    callback = Index()
    plt.sca(axs[1])

    axs[1].axis('off')

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    class RectangleData(object):
        def __init__(self, indexer):
            self.indexer = indexer
            self.rectangle = []
            self.actual_index = indexer.this_ind
            self.extents = []
            self.actual_mask = []

        def get_mask(self):
            t_ini = self.extents[0]
            t_end = self.extents[1]

            times = df.loc[self.actual_index].time

            mask = [True if t_ini <= time <= t_end else False for time in times]

            return mask

        def load(self, event_press, event_release):
            self.extents = self.rectangle.extents
            self.actual_index = self.indexer.this_ind
            self.actual_mask = self.get_mask()

    rectangle = RectangleData(callback)
    rect = RectangleSelector(axs[0], rectangle.load)
    rectangle.rectangle = rect

    def save_region(event):
        ind = rectangle.actual_index
        mask = rectangle.actual_mask

        df.at[ind, 'sigmoid_mask'] = np.logical_or(mask, df.at[ind, 'sigmoid_mask'])
        print('saved')

    axsave = plt.axes([0.59, 0.05, 0.1, 0.075])
    bsave = Button(axsave, 'Save')
    bsave.on_clicked(save_region)

    def remove_region(event):
        ind = callback.this_ind
        df.at[ind, 'sigmoid_mask'] = np.array([False] * len(df.at[ind, 'sigmoid_mask']))
        print('removed')

    axremove = plt.axes([0.42, 0.05, 0.15, 0.075])
    bremove = Button(axremove, 'Remove all')
    bremove.on_clicked(remove_region)

    plt.show()


def view_curves(axs, df_row, sensors, lines=None, fill=None):
    time = df_row.time
    mask = df_row.sigmoid_mask

    ind = df_row.index
    date = df_row.date
    drug = df_row.drug
    plasmid = df_row.plasmid
    axs.set_title('ind: %s; date: %s; drug: %s; plasmid: %s' % (ind, date, drug, plasmid))

    if lines is None:
        plt.sca(axs)

        lines = []
        for _, sensor_row in sensors.iterrows():
            anis = df_row[sensor_row.fluorophore + '_anisotropy']
            l, = axs.plot(time, anis, color=sensor_row.color, label=sensor_row.fluorophore)
            lines.append(l)

        plt.legend()
        plt.sca(axs)
        y_lims = axs.get_ylim()
        fill = axs.fill_between(time, y_lims[0], y_lims[1], where=mask, color='g', alpha=0.1)
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
        fill = axs.fill_between(time, y_lims[0], y_lims[1], where=mask, color='g', alpha=0.1)
        plt.draw()

        return lines, fill
