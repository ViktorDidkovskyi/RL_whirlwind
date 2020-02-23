import math
import numpy as np

def circulation_center(points, k=None):
    """

    :param points:
    :param k:
    :return:
    """
    if k is None: k = np.array([[1] * len(points)])
    return np.sum(points * np.append(k, k, axis=0).T, axis=0) / np.sum(k)


def circulation_moment(points):
    """

    :param points:
    :return:
    """
    return np.sum([point ** 2 for point in points])


def circulation_dispersion(points):
    """

    :param points:
    :return:
    """
    calculate_circulation_center = circulation_center(points)
    return np.sum([(point - calculate_circulation_center) ** 2 for point in points])


def calculate_the_measure(zi, zj):
    """
    calculate the distance between 2 points
    :param zi:
    :param zj:
    :return:
    """
    return math.hypot(zj[0] - zi[0], zj[1] - zi[1])


def caculate_the_interaction(points, i):
    """

    :param z_input:
    :param i:
    :return:
    """
    u = np.array([[0, 1],
                  [-1, 0]])
    n_whirlwind = len(points)

    sum_of_interaction = []
    for j in range(n_whirlwind):
        if i != j:
            interaction = np.dot(u, (points[i] - points[j]))
            interaction = interaction / calculate_the_measure(points[i], points[j])

            sum_of_interaction.append(interaction)

    return sum(sum_of_interaction)


def next_z_generate_point(points, ksi, betta, alfa, use_second_interaction=False):
    """
    :param points:
    :param ksi:
    :param betta:
    :param alfa:
    :param use_second_interaction:
    :return:
    """
    n_whirlwind = len(points)
    new_points = []
    for i in range(n_whirlwind):

        weight_interaction = betta * caculate_the_interaction(points, i)

        second_weight_interaction = 0
        if use_second_interaction:
            second_sum_of_interaction_calculation = []
            for k in range(n_whirlwind):
                if k != i:
                    result_second_intersaction = betta ** 2 / 2

                    intersection_i_m = caculate_the_interaction(points, i)
                    intersection_k_m = caculate_the_interaction(points, k)

                    second_sum_of_interaction_calculation.append(
                        result_second_intersaction * (intersection_i_m - intersection_k_m)
                    )

            second_weight_interaction = np.sum(second_sum_of_interaction_calculation)

        new_points.append(points[i] + weight_interaction + second_weight_interaction + ksi[i] - alfa * (
                points[i] - circulation_center(points)))

    return np.stack(new_points, axis=0)


def loss_center_alfa_betta(alfa_betta, args):
    """

    :param alfa_betta:
    :param args:
    :return:
    """
    betta, alfa = alfa_betta
    z_null, z, ksi, alfa, use_second_interaction, betta_list, alfa_list = args

    z_new = next_z_generate_point(z, ksi, betta, alfa, use_second_interaction)

    II_null = circulation_moment(z_null)
    # II_prev = calculate_dist_to_center_complex(z_complex, betta)
    II_next = circulation_moment(z_new)

    # return II_next - II_null + np.sum(np.divide(1 ,betta_list))
    return II_next + betta ** 2


def loss_center_alfa(betta, args):
    """

    :param betta:
    :param args:
    :return:
    """
    z_null, z, ksi, alfa, use_second_interaction, betta_list = args

    z_new = next_z_generate_point(z, ksi, betta, alfa, use_second_interaction)

    II_null = circulation_moment(z_null)
    II_next = circulation_moment(z_new)

    return II_next - II_null + np.sum(np.divide(1- betta_list, betta_list))

def generate_random_dots(min_value: float, max_value: float, n_whirlwind: int) -> np.array:
    """
    :param min_value:
    :param max_value:
    :param n_whirlwind:
    :return:
    """
    return np.random.uniform(min_value, max_value, (n_whirlwind, 2))


def generate_noise_dots(n_whirlwind: int, distribution_params: (float, float) = (0, 1)) -> np.array:
    """

    :param n_whirlwind:
    :param distribution_params:
    :return:
    """
    return np.array([
        np.random.normal(distribution_params[0], distribution_params[1], n_whirlwind) for i in range(2)
                      ]).T


def save_div(x,y):
    """

    :param x:
    :param y:
    :return:
    """
    try:
        return x/y
    except ZeroDivisionError:
        return 0