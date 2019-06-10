import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button


def df_viewer(df, sensors):

    fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})

    view_curves(axs[0], df.iloc[0], sensors)

    class Index(object):
        index = df.index
        cur_ind = 0
        max_ind = len(df.index)
        this_ind = df.index[cur_ind]

        def next(self, event):
            self.cur_ind = min(self.cur_ind + 1, self.max_ind)
            self.this_ind = df.index[self.cur_ind]
            view_curves(axs[0], df.loc[self.this_ind], sensors)

        def prev(self, event):
            self.cur_ind = max(self.cur_ind - 1, 0)
            self.this_ind = df.index[self.cur_ind]
            view_curves(axs[0], df.loc[self.this_ind], sensors)

    callback = Index()
    plt.sca(axs[1])
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()


def view_curves(axs, df_row, sensors):

    plt.sca(axs)
    plt.cla()

    time = df_row.time
    mask = df_row.sigmoid_mask
    for _, sensor_row in sensors.iterrows():
        anis = df_row[sensor_row.fluorophore + '_anisotropy']
        axs.plot(time, anis, color=sensor_row.color, label=sensor_row.fluorophore)

    axs.fill_between(time, 0, 1, where=mask)
    plt.sca(axs)
    plt.legend()
    plt.xlabel('Time (min.)')
    plt.ylabel('Anisotropy')
    plt.draw()

    def onselect(eclick, erelease):
        "eclick and erelease are matplotlib events at press and release."
        xs = []
        ys = []
        xs.append(eclick.xdata)
        ys.append(eclick.ydata)
        xs.append(erelease.xdata)
        ys.append(erelease.ydata)

        return [np.min(xs), np.min(ys), np.max(xs), np.max(ys)]

