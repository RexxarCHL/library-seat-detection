"""
Visualize the labels based on frame count
Code adapted from https://stackoverflow.com/a/8782324/10420149
"""

import numpy as np
import matplotlib.pyplot as plt

labels_detection = np.genfromtxt("detection_labels.csv", dtype=int, delimiter=",")
ground_truth = np.genfromtxt("ground_truth_labels.csv", dtype=int, delimiter=",")
labels_e2e = np.genfromtxt("e2e_labels.csv", dtype=float)
labels_e2e = np.array([labels_e2e[:8564], labels_e2e[8564:8564*2],
                       labels_e2e[8564*2:8564*3], labels_e2e[8564*3:]]).T.astype(int)

# SEE_SEAT = 0
# data = [ground_truth[:, SEE_SEAT],
#         labels_detection[:, SEE_SEAT]]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.axes.get_yaxis().set_visible(False)
ax.set_aspect(1)


def avg(a, b):
    return (a + b) / 2.0


m = 250
num_rows = 2

for seat in range(4):
    data = [ground_truth[:, seat],
            # labels_detection[:, seat],
            labels_e2e[:, seat]]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_aspect(1)
    for y, row in enumerate(data):
        for x, col in enumerate(row):
            x1 = [x, x+1]
            y1 = np.array([y*m+y*30, y*m+y*30])
            y2 = y1+m
            if col == 0:  # EMPTY
                plt.fill_between(x1, y1, y2=y2, color='green')

            if col == 1:  # ON_HOLD
                plt.fill_between(x1, y1, y2=y2, color='yellow')

            if col == 2:  # OCCUPIED
                plt.fill_between(x1, y1, y2=y2, color='red')

    plt.ylim(num_rows*m+30, 0)
    plt.xlim(0, len(ground_truth))
    plt.savefig("bar_graph_seat{}".format(seat), bbox_inches="tight", dpi=1000)
    plt.show()
    plt.gcf().clear()