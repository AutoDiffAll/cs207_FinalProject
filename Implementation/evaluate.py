import sys
try:
    from Optimizer import minimize
except:
    from Implementation.Optimizer import minimize

try:
    sys.path.append('../AutoDiff')
    from variables import Variable
except:
    from AutoDiff.variables import Variable


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def plot_path_2D(val_arr, x_grid, y_grid, fn):
    f_grid = fn(x_grid.reshape(1, -1),
                y_grid.reshape(-1, 1))
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
    plt.scatter(vends[0], vends[1], s=100, color='black', label='minimum')

    plt.legend(fontsize=14)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Start = [{0}, {1}]'.format(v0[0], v0[1]), fontsize=16)

    plt.xlim([x_grid.min(), x_grid.max()])
    plt.ylim([y_grid.min(), y_grid.max()])

def plot_path_1D(val_arr,x_grid,fn):
    f_grid = fn(x_grid)
    plt.plot(x_grid,f_grid,label='F function',color='black',linewidth=3.0)
    v0 = val_arr[0]
    vends = val_arr[-1]
    plt.scatter(v0, fn(v0), s=100, color='red', label='start')
    plt.scatter(vends, fn(vends), s=100, color='black', label='minimum')

    plt.plot(val_arr, fn(val_arr), 'r--',label='path', 
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


def f1(x, y): return 100*(y-x**2)**2 + (1-x)**2
res = minimize(f1, [-1, 1], method="Steepest Descend")
x_grid = np.linspace(-3, 3, 150)
y_grid = np.linspace(-3, 4, 200)


plt.figure(figsize=(6, 8))
plot_path_2D(res.val_rec, x_grid, y_grid, f1)
plt.show()

f0=lambda x: (x-2)**2+1
res=minimize(f0,[0],method="Steepest Descend")
x_grid = np.linspace(-2, 3, 150)

plt.figure(figsize=(6, 8))
plot_path_1D(res.val_rec, x_grid,  f0)
plt.show()

