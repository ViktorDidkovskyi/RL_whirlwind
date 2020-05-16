import matplotlib.pyplot as plt
import itertools
import numpy as np


def plot_the_whirlwind(points_memory, path_to_save: str = ""):
    """

    :param z_memory:
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))  # (25,20)
    first = points_memory[0]

    color_list = get_colours()
    np.random.shuffle(color_list)
    for c, first_ in enumerate(first):
        ax.scatter(*first_, s=150, label="o", alpha=1, color=color_list[c], marker='o')

    for points in points_memory[1:]:

        for point in points:
            ax.scatter(*point, label="x", alpha=1, marker='x')

        colorst = itertools.cycle(color_list[:c + 1])
        for j1, color in zip(ax.collections, colorst):
            j1.set_color(color)

    # ax.legend( prop={'size': 10})

    plt.grid()
    plt.savefig(path_to_save + ".jpg")
    plt.show(block=True)
    plt.interactive(False)



def plot_metrics(list_data, title: str = "Title", path_to_save: str = ""):
    """

    :param list_data:
    :param title:
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))  # (25,20)

    ax.scatter(range(len(list_data)), list_data, label="x", alpha=1, s=25, color="blue", marker='x')

    # ax.legend(prop={'size': 10})
    plt.title(title.title())
    plt.grid()
    plt.savefig(path_to_save + ".jpg")
    plt.show(block=True)
    plt.interactive(False)


def plot_points(points):
    """

    :param points:
    :return:
    """
    # point = np.unique(points)
    fig, ax = plt.subplots(1, 1, figsize=(25, 15))  # (25,20)

    ax.scatter(*zip(*points), label="x", color="blue", marker='x')

    ax.legend(prop={'size': 10})
    plt.grid()
    plt.show(block=True)
    plt.interactive(False)


def get_colours():
    """

    :return:
    """

    deep = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
            "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

    muted = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4",
             "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]

    pastel = ["#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B", "#D0BBFF",
              "#DEBB9B", "#FAB0E4", "#CFCFCF", "#FFFEA3", "#B9F2F0"]

    bright = ["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
              "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"]

    dark = ["#001C7F", "#B1400D", "#12711C", "#8C0800", "#591E71",
            "#592F0D", "#A23582", "#3C3C3C", "#B8850A", "#006374"]

    colorblind = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC",
                  "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"]

    colours = deep + muted  + dark + colorblind + bright


    return colours

