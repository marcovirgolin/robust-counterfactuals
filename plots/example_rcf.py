import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns


def f(x):
    #y=0.8+0.1*np.log(1-1.2*x)
    y=-.02*np.log(x-.605)+.635
    y[np.isnan(y)] = 1
    y[np.isinf(y)] = 1
    return y


x_1 = np.arange(0.0, 0.851, step=0.001)
x_2 = f(x_1)

xlim = (0.6, 0.735)
ylim = (0.665, 0.8)
width = 45*(xlim[1] - xlim[0])
height = 40*(ylim[1] - ylim[0])


# PLOT
import matplotlib
font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(width,height))
#ax.set_ylim(0.0, 0.7501)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
#ax.set_ylim((-2,2))

#sns.set()




# decision boundary
#sns.scatterplot(x_1,x_2)
ax.fill_between(x_1,x_2,ylim[1], color='indianred')
ax.fill_between(x_1,x_2,ylim[0], color='darkseagreen')

# x
ax.scatter(0.7, 0.75, marker='x', color='k', s=120)
# non-robust z
ax.errorbar(0.7, 0.68, xerr=np.array([[0.02],[0.02]]).reshape((2,1)), marker='s', color='k', markersize=8, capsize=1)
# robust z
ax.errorbar(0.605, 0.75, yerr=np.array([[0.03],[0.04]]).reshape((2,1)), marker='d', color='k', markersize=10, capsize=1)


# x-to-non-robust-z
ax.arrow(0.7, 0.75-0.002, 0.0, 0.68-0.75+0.004, length_includes_head=True, linestyle='-', width=0, head_width=0.002, color='white')
# x-to-robust-z
ax.arrow(0.7-0.002, 0.75, 0.605-0.7+0.004, 0.0, length_includes_head=True, linestyle='-', width=0, head_width=0.002, color='white')

ax.set_xlabel('Blood pressure')
ax.set_ylabel('Vitamin deficiency')



fig.tight_layout()
plt.savefig('example_bloodpressure.pdf')
plt.show()