import sys, matplotlib
import numpy as np
from matplotlib import pyplot as plt
import time
matplotlib.use('TkAgg')

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib import gridspec
#from matplotlib.backend_bases import key_press_handler
#from matplotlib.figure import Figure

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


####################### Tkinter Root ##################################
root = Tk.Tk()
root.wm_title("Dynamic TestData Plot")

# read testNames.txt and return the name list
def readTestName(inputFile):
    testName = open(inputFile, 'r')
    n = testName.readline()
    n = n.rstrip('\n')
    nn = n.split(',')
    ns = []
    name = nn[0]
    for l in nn[1:]:
        if l.isupper():
            ns.append(name)
            name = l
        else:
            name += l
    return ns


# read KSSvec.txt and return a list of data
def readKSSvec(inputFile):
    KSSVec = open(inputFile, 'r')
    k = KSSVec.readline()
    k = k.rstrip('\n')
    kk = k.split(',')
    data = [int(l) for l in kk]
    return data


# read testData and return a dictionary containing all the data lists
def readTestData(inputFile):
    testData = open(inputFile, 'r')
    tDict = {}
    token = 't'
    lines = testData.readlines()
    # for the first line
    data = lines[0].rstrip('\n').split(',')
    names = []
    for i in range(len(data)):
        names.append(token + str(i))
    for n in names:
        tDict[n] = []
    for t in lines:
        if t == '\n':
            break
        t = t.rstrip('\n')
        tt = t.split(',')
        for i in range(len(tt)):
            tDict['t'+str(i)].append(float(tt[i]))
    return tDict


class TestDataPlot:
    def __init__(self, testNamesFile, testDataFile):
        testDataDict = readTestData(testDataFile)
        self.testNames = readTestName(testNamesFile)
        self.num = len(self.testNames)
        self.data = []
        for i in range(self.num):
            self.data.append(testDataDict['t'+str(i)])
        self.START = False
        self.PAUSE = False
        self.SPEED = 1
        self.speedChange = False
        self.ax = []

    def _start(self):
        self.START = True

    def _reset(self):
        if self.SPEED != 1:
            self.speedChange = True
            self.SPEED = 1
        self.START = False
        self.PAUSE = False

    def _quit(self):
        root.quit()
        root.destroy()

    def _pause(self):
        self.PAUSE = not self.PAUSE

    def _speedUp(self):
        self.SPEED += 1
        self.speedChange = True

    def _speedDown(self):
        if self.SPEED == 1:
            return
        else:
            self.SPEED -= 1
            self.speedChange = True

    def tkPlot(self):

        fig = plt.figure()
        # create grid layout
        gs = gridspec.GridSpec(self.num, 5)

        # every signal is rendered for line plot
        for i in range(self.num):
            self.ax.append(fig.add_subplot(gs[i,:-1]))
        # assign the last signal for bar plot
        self.ax.append(fig.add_subplot(gs[:,-1]))

        # initialize the y_lim and y_label
        for i in range(self.num):
            self.ax[i].set_ylim(min(self.data[i])-1, max(self.data[i])+1)
            self.ax[i].set_ylabel(self.testNames[i], rotation=0, labelpad=40)
            if i == 0:
                self.ax[i].spines['bottom'].set_visible(False)
            elif i == self.num-1:
                self.ax[i].spines['top'].set_visible(False)
            else:
                self.ax[i].spines['top'].set_visible(False)
                self.ax[i].spines['bottom'].set_visible(False)
        self.ax[self.num].get_xaxis().set_visible(False)
        fig.patch.set_visible(False)

        # initialize bar chart
        bar_index = self.num - 1
        bar_data = self.data[bar_index]
        rect = self.ax[self.num].bar([1], bar_data[0], width=0.35, align='center')
        self.ax[self.num].set_ylim(min(bar_data)-1, max(bar_data)+1)

        lines = []
        for i in range(self.num):
            lines.append(self.ax[i].plot([], [], lw=2)[0])

        xdata = np.asarray(list(range(len(self.data[0]))))
        ydata = []
        for i in range(self.num):
            ydata.append(np.asarray(self.data[i]))
            lines[i].set_data(xdata, ydata[i])

        ############## set the tkinter features ####################
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg(canvas, window=root)
        toolbar.update()
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        button0 = Tk.Button(master=root, text='Start', command=self._start)
        button0.pack(side=Tk.LEFT, expand=1)

        button1 = Tk.Button(master=root, text='Reset', command=self._reset)
        button1.pack(side=Tk.LEFT, expand=1)

        button2 = Tk.Button(master=root, text='Quit', command=self._quit)
        button2.pack(side=Tk.LEFT, expand=1)

        button3 = Tk.Button(master=root, text='Pause/Continue', command=self._pause)
        button3.pack(side=Tk.LEFT, expand=1)

        button4 = Tk.Button(master=root, text='Speed Down', command=self._speedDown)
        button4.pack(side=Tk.LEFT)

        v_speed = Tk.StringVar()
        speedLabel = Tk.Label(master=root, textvariable=v_speed, font='Verdana 10 bold')
        speedLabel.pack(side=Tk.LEFT)
        v_speed.set('Speed: '+str(self.SPEED))

        button5 = Tk.Button(master=root, text='Speed Up', command=self._speedUp)
        button5.pack(side=Tk.LEFT)

        v_time = Tk.StringVar()
        timeLabel = Tk.Label(master=root, textvariable=v_time, font='Verdana 10 bold')
        timeLabel.pack(side=Tk.LEFT, expand=1)

        ############# initialization ################
        cnt = 0
        n_min = 2
        sampling_rate = 60
        windowLen = n_min * sampling_rate * 60
        n = int(n_min * 20)

        totalLen = len(self.data[0])
        frameNum = totalLen

        tStart = 0
        tEnd = windowLen

        x_ticks = list(range(0, len(self.data[0]), sampling_rate * 30))
        t_ticks = []
        for i in range(len(x_ticks)):
            min_tick = str(i/2)
            if not i%2:
                sec_tick = '00'
            else:
                sec_tick = '30'
            t_ticks.append(min_tick + ':' + sec_tick)

        self.ax[self.num-1].set_xticks(x_ticks)
        self.ax[self.num-1].set_xticklabels(t_ticks)

        for i in range(self.num-1):
            self.ax[i].set_xticks([0])

        while True:
            ######### if not started, looping inside here #################
            if not self.START:
                tStart = 0
                tEnd = windowLen
                cnt = 0
                for i in range(self.num):
                    self.ax[i].set_xlim(tStart, tEnd)
                rect.patches[0].set_height(bar_data[tEnd])

                if self.speedChange:
                    v_speed.set('Speed: '+str(self.SPEED))
                    self.speedChange = False

                time_now = tEnd / sampling_rate
                hour_now = time_now / 3600
                min_now = (time_now - hour_now * 3600) / 60
                sec_now = time_now - hour_now * 3600 - min_now * 60
                v_time.set('Time: '+str(hour_now).zfill(2)+':'+str(min_now).zfill(2)
                           + ':' + str(sec_now).zfill(2))
                fig.canvas.draw()
                fig.canvas.flush_events()
                fig.canvas.get_tk_widget().update()
            ########## else the program is started #######################
            else:
                if not self.PAUSE:
                    time.sleep(0.2)

                    tStart += n*self.SPEED
                    tEnd += n*self.SPEED

                    # update the data or axis
                    for i in range(self.num):
                        self.ax[i].set_xlim(tStart, tEnd)
                    rect.patches[0].set_height(bar_data[tEnd])

                    time_now = tEnd / sampling_rate
                    hour_now = time_now / 3600
                    min_now = (time_now - hour_now * 3600) / 60
                    sec_now = time_now - hour_now * 3600 - min_now * 60
                    v_time.set('Time: '+str(hour_now).zfill(2)+':'+str(min_now).zfill(2)
                               + ':' + str(sec_now).zfill(2))

                    if self.speedChange:
                        v_speed.set('Speed: '+str(self.SPEED))
                        self.speedChange = False

                    cnt += 1
                    if cnt == frameNum:
                        break
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                fig.canvas.get_tk_widget().update()

        Tk.mainloop()



if __name__ == '__main__':
    dataPlot = TestDataPlot('testNames.txt', 'testData.txt')
    dataPlot.tkPlot()


