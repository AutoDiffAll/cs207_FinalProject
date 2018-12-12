import sys
from .optimizer import minimize
from .autodiff.variables import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_path_2D(val_arr, x_grid, y_grid, fn):
    f_grid = fn(x_grid.reshape(1, -1),
                y_grid.reshape(-1, 1))
    # f_grid=f_grid-f_grid.min()+1

    plt.contourf(x_grid, y_grid, f_grid, cmap='Blues',
                 norm=colors.LogNorm(vmin=f_grid.min(), vmax=f_grid.max())
                 )
    plt.colorbar(orientation='horizontal')

    v0 = val_arr[0]
    vends = val_arr[-1]
    plt.plot(val_arr[:, 0], val_arr[:, 1], label='path', color='black',
             linewidth=3.0)

    # n = int(len(val_arr)/2)

    # try:
    #     x, y = val_arr[n]
    #     dx,dy=val_arr[n+1]-val_arr[n]
    # except:
    #     x, y = val_arr[n-1]
    #     dx,dy=val_arr[n]-val_arr[n-1]
    # plt.arrow(x,y, dx, dy, shape='full',
    #           lw=0, length_includes_head=True, head_width=0.05, color='black')

    plt.scatter(v0[0], v0[1], s=100, color='red', label='start')
    plt.scatter(vends[0], vends[1], s=100, color='black',
                label='minimum [{0:.2f},{1:.2f}]'.format(vends[0], vends[1]))

    plt.legend(fontsize=14)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Start = [{0}, {1}]'.format(v0[0], v0[1]), fontsize=16)

    plt.xlim([x_grid.min(), x_grid.max()])
    plt.ylim([y_grid.min(), y_grid.max()])


def plot_path_1D(val_arr, x_grid, fn):
    f_grid = fn(x_grid)
    plt.plot(x_grid, f_grid, label='F function', color='black', linewidth=3.0)
    v0 = val_arr[0]
    vends = val_arr[-1]
    plt.scatter(v0, fn(v0), s=100, color='red', label='start')
    plt.scatter(vends, fn(vends), s=100, color='black',
                label='minimum {0:.2f}'.format(vends[0]))

    plt.plot(val_arr, fn(val_arr), 'r--', label='path',
             linewidth=3.0)
    # n=int(len(val_arr)/2)

    # plt.arrow(val_arr[n], fn(*val_arr[n]), 0.1, fn(*(val_arr[n]+0.1))-fn(*val_arr[n]), shape='full',
    #                    lw=0, length_includes_head=True, head_width=0.05,color='red')

    plt.title('Start = {0}'.format(v0), fontsize=16)

    plt.xlim([x_grid.min(), x_grid.max()])
    plt.ylim([f_grid.min(), f_grid.max()])
    plt.legend(fontsize=14)

    plt.xlabel('x')
    plt.ylabel('f(x)')


def plot_path(fn, val_lists, title, dim=2, **kwargs):
    n = len(val_lists)
    if n >= 3:
        plt.figure(figsize=(7*3, 8*np.ceil(n/3)))
    else:
        plt.figure(figsize=(7*n, 8))
    for i in range(n):
        plt.subplot(np.ceil(n/3), min(3, n), i+1)
        if dim == 1:
            # plot 1d path
            plot_path_1D(val_arr=val_lists[i].val_rec, fn=fn, **kwargs)
        else:
            # plot 2D path
            plot_path_2D(val_arr=val_lists[i].val_rec, fn=fn, **kwargs)
    plt.suptitle(title, fontsize=25, y=1.02)
    plt.show()


def plot_conv(time_lists, label_lists, precision=1e-5):
    n = len(time_lists)
    for i in range(n):
        plt.plot(time_lists[i], label=label_lists[i])
    plt.title(
        'Convergency time for different model (Precesion_threshold={})'.format(precision))
    plt.xlabel('iterations')
    plt.ylabel('time(s)')
    plt.legend()
    plt.grid()
    plt.show()


def plot_convergency(res, i=0, **kwargs):
    conv = {}
    conv_labels = []
    not_conv = {}
    notc_labels = []
    for k, v in res.items():
        if v[-1].converge:
            conv[k] = v
            conv_labels.append(k)
        else:
            not_conv[k] = v
            notc_labels.append(k)
    if len(conv_labels) > 0:
        plot_conv(
            [v[i].time_rec for k, v in conv.items()], conv_labels)
    if len(notc_labels) > 0:
        plot_conv(
            [v[i].time_rec for k, v in not_conv.items()], notc_labels)


def plot_accuracy(res, true, i=0, **kwargs):
    conv = {}
    conv_labels = []
    not_conv = {}
    notc_labels = []
    for k, v in res.items():
        if v[-1].converge:
            conv[k] = v
            conv_labels.append(k)
        else:
            not_conv[k] = v
            notc_labels.append(k)
    if len(conv_labels) > 0:
        plot_acc(
            [v[i].val_rec for k, v in conv.items()], true, conv_labels, **kwargs)
    if len(notc_labels) > 0:
        plot_acc(
            [v[i].val_rec for k, v in not_conv.items()], true, notc_labels, **kwargs)


def plot_acc(val_lists, true, label_lists, norm='L2'):
    n = len(val_lists)
    for i in range(n):
        if norm == 'L1':
            err = np.linalg.norm(val_lists[i]-[1, 1], 1, axis=-1)
        elif norm == 'L_inf':
            err = np.linalg.norm(val_lists[i]-[1, 1], np.inf, axis=-1)
        else:
            err = np.linalg.norm(val_lists[i]-[1, 1], 2, axis=-1)
        plt.plot(err, label=label_lists[i])
    plt.title(norm+' Error per iterations')
    plt.xlabel('iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.grid()
    plt.show()


def show_acc(val_lists, true, label_lists, norm='L2'):
    n = len(label_lists)

    if norm == 'L1':
        err = np.linalg.norm(np.array(val_lists)-np.array([1, 1]), 1, axis=-1)
    elif norm == 'L_inf':
        err = np.linalg.norm(np.array(val_lists) -
                             np.array([1, 1]), np.inf, axis=-1)
    else:
        err = np.linalg.norm(np.array(val_lists)-np.array([1, 1]), 2, axis=-1)
    for i, e in enumerate(err):
        print(norm+' error for {0:<20} is: {1:.2E}'.format(label_lists[i], e))


if __name__ == "__main__":

    def f1(x, y): return 100*(y-x**2)**2 + (1-x)**2

    v0_list = [[-1, 1], [0, 1], [2, 1]]
    val_lists = [minimize(f1, vo, method="Steepest Descent") for vo in v0_list]
    x_grid = np.linspace(-3, 3, 150)
    y_grid = np.linspace(-3, 4, 200)
    plot_path(f1, val_lists, "Steepest Descent", x_grid=x_grid, y_grid=y_grid)
    val_arr = [val.val_rec for val in val_lists]
    plot_acc(val_arr, [1, 1], ['start at '+str(v0)
                               for v0 in v0_list], norm='L2')

    plot_convergency([val.time_rec for val in val_lists], [
        'start at '+str(v0) for v0 in v0_list])

    def f0(x): return x**2+(x-3)**2
    v0_list = [[-1], [0], [2]]
    val_lists = [minimize(f0, vo, method="Steepest Descent") for vo in v0_list]
    x_grid = np.linspace(-2, 4, 150)
    plot_path(f0, val_lists, "Steepest Descent", dim=1, x_grid=x_grid)
