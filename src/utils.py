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

    :param zi:
    :param zj:
    :return:
    """
    return math.hypot(zj[0] - zi[0], zj[1] - zi[1])


def caculate_the_interaction(z_input, i):
    """

    :param z_input:
    :param i:
    :return:
    """
    u = np.array([[0, 1],
                  [-1, 0]])
    n_whirlwind = len(z_input)

    sum_of_interaction = []
    for j in range(n_whirlwind):
        if i != j:
            interaction = np.dot(u, (z_input[i] - z_input[j]))
            interaction = interaction / calculate_the_measure(z_input[i], z_input[j])

            sum_of_interaction.append(interaction)

    return sum(sum_of_interaction)


def next_z_generate_point(z_input, ksi, betta, alfa, use_second_interaction=False):
    """
    :param z_input:
    :param ksi:
    :param betta:
    :param alfa:
    :param use_second_interaction:
    :return:
    """
    n_whirlwind = len(z_input)
    z_new = []
    for i in range(n_whirlwind):

        weight_interaction = betta * caculate_the_interaction(z_input, i)

        second_weight_interaction = 0
        if use_second_interaction:
            second_sum_of_interaction_calculation = []
            for k in range(n_whirlwind):
                if k != i:
                    result_second_intersaction = betta ** 2 / 2

                    intersection_i_m = caculate_the_interaction(z_input, i)
                    intersection_k_m = caculate_the_interaction(z_input, k)

                    second_sum_of_interaction_calculation.append(
                        result_second_intersaction * (intersection_i_m - intersection_k_m)
                    )

            second_weight_interaction = np.sum(second_sum_of_interaction_calculation)

        z_new.append(z_input[i] + weight_interaction + second_weight_interaction + ksi[i] - alfa * (
                z_input[i] - circulation_center(z_input)))

    return z_new


def generate_random_dots(min_value: float, max_value: float, n_whirlwind: int) -> np.array:
    """
    :param min_value:
    :param c:
    :param n_whirlwind:
    :return:
    """
    return np.random.randint(min_value, max_value, (2, n_whirlwind))


def generate_noise_dots(n_whirlwind: int, distribution_params: (float, float) = (0, 1)) -> np.array:
    """

    :param n_whirlwind:
    :param distribution_params:
    :return:
    """
    return np.array([np.random.normal(distribution_params[0], distribution_params[1], n_whirlwind),
                     np.random.normal(distribution_params[0], distribution_params[1], n_whirlwind)]).T