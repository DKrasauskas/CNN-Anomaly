import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class Plotter:
    def __init__(self, data, title1, title2):
        #primary
        self.fig1, self.ax1 =  plt.subplots(2, 1, figsize=(8, 6))
        self.fig1.canvas.manager.set_window_title('title1')
        #secondary controls window
        self.fig2 = plt.figure(figsize=(6, 2))
        self.fig2.canvas.manager.set_window_title('Slider Controls')
        axcolor = 'lightgoldenrodyellow'

        ax_xmin = plt.axes([0.1, 0.6, 0.8, 0.15], facecolor=axcolor)
        ax_xmax = plt.axes([0.1, 0.4, 0.8, 0.15], facecolor=axcolor)
        ax_ymin = plt.axes([0.1, 0.2, 0.8, 0.15], facecolor=axcolor)
        ax_ymax = plt.axes([0.1, 0.0, 0.8, 0.15], facecolor=axcolor)

        self.s_xmin = Slider(ax_xmin, 'X Min', 0, 10, valinit=0)
        self.s_xmax = Slider(ax_xmax, 'X Max', 0, 10, valinit=10)
        self.s_ymin = Slider(ax_ymin, 'Y Min', -2, 0, valinit=-1)
        self.s_ymax = Slider(ax_ymax, 'Y Max', 0, 2, valinit=1)

        self.s_xmin.on_changed(self.update)
        self.s_xmax.on_changed(self.update)
        self.s_ymin.on_changed(self.update)
        self.s_ymax.on_changed(self.update)


    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.data)
        plt.title('Data Plot')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.show()