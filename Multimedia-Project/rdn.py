import random
import numpy as np
from matplotlib import pyplot as plt
import time, sys, matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib import gridspec


if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

matplotlib.use('TkAgg')

root = Tk.Tk()
root.wm_title('Dynamic Random Plot in TK')

START = False
n_min = 2

def _start():
    global START
    START = True

def _reset():
    global START
    START = False

def _quit():
    root.quit()
    root.destroy()

PAUSE = False

def _pause():
    global PAUSE
    PAUSE = not PAUSE

SPEED = 1

def _speedUp():
    global SPEED
    SPEED += 1

def _speedDown():
    global SPEED
    if SPEED == 1:
        return
    else:
        SPEED -= 1

def _getText(e):
    global n_min
    s = e.get()
    try:
        n_min = int(s)
        print n_min
        e.delete(0, 'end')
    except:
        e.delete(0, 'end')
    return


def dynamicPlot():
    numOfSamples = 100000
    data = [random.randint(1, 50) for _ in range(numOfSamples)]
    #plt.ion()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.set_ylim(0, 60)
    line, = ax0.plot([],[],lw=2)

    ax0.set_ylim(min(data)-1, max(data)+1)
    ax0.set_ylabel('random data', rotation=0, labelpad=30)

    rect = ax1.bar([1], data[0], width=0.5, align='center')
    ax1.set_ylim(min(data)-1, max(data)+1)
    ax1.get_xaxis().set_visible(False)
    #ax.patch.set_visible(False)
    #fig.patch.set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    toolbar = NavigationToolbar2TkAgg(canvas, window=root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    v = Tk.StringVar()
    timeLabel = Tk.Label(master=root, textvariable=v, font='Verdana 10 bold')
    timeLabel.pack(side=Tk.LEFT, expand=1)

    button0 = Tk.Button(master=root, text='Start', command=_start)
    button0.pack(side=Tk.LEFT, expand=1)

    button1 = Tk.Button(master=root, text='Reset', command=_reset)
    button1.pack(side=Tk.LEFT, expand=1)

    button2 = Tk.Button(master=root, text='Quit', command=_quit)
    button2.pack(side=Tk.LEFT, expand=1)

    button3 = Tk.Button(master=root, text='Pause/Continue', command=_pause)
    button3.pack(side=Tk.LEFT, expand=1)

    button4 = Tk.Button(master=root, text='SpeedUp', command=_speedUp)
    button4.pack(side=Tk.LEFT, expand=1)

    button5 = Tk.Button(master=root, text='SpeedDown', command=_speedDown)
    button5.pack(side=Tk.LEFT, expand=1)

    entry0 = Tk.Entry(master=root)
    entry0.pack(side=Tk.LEFT)

    button6 = Tk.Button(master=root, text='okay', command=lambda : _getText(entry0))
    button6.pack(side=Tk.LEFT)

    xdata = np.asarray(list(range(numOfSamples)))
    ydata = np.asarray(data)
    line.set_data(xdata, ydata)

    sampling_rate = 10
    windowLen = n_min * 60 * sampling_rate
    n_move = int(n_min * 30)

    x_ticks = list(range(0, numOfSamples, windowLen/n_min/2))

    t_ticks = []
    for i in range(len(x_ticks)):
        min_tick = str(i/2)
        if not i%2:
            sec_tick = '00'
        else:
            sec_tick = '30'
        t_ticks.append(min_tick + ':' + sec_tick)

    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(t_ticks)

    xStart = 0
    xEnd = windowLen

    cnt = 0
    while True:
        if not START:
            xStart = 0
            xEnd = windowLen
            cnt = 0
            ax0.set_xlim(xStart, xEnd)
            rect.patches[0].set_height(data[xEnd])
            time_now = xEnd / sampling_rate
            v.set('Time: '+str(time_now)+' [sec]')
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.canvas.get_tk_widget().update()
        else:
            if not PAUSE:
                time.sleep(0.1)

                xStart += SPEED * n_move
                xEnd += SPEED * n_move
                ax0.set_xlim(xStart, xEnd)

                rect.patches[0].set_height(data[xEnd])

                time_now = xEnd / sampling_rate

                v.set('Time: '+str(time_now)+' [sec]')

                cnt += 1
                if cnt == numOfSamples:
                    break
                fig.canvas.draw()
                fig.canvas.flush_events()

            fig.canvas.get_tk_widget().update()

    Tk.mainloop()


dynamicPlot()