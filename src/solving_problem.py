import numpy as np
import scipy

def calculate_beta(r_dst_level, gama):
    current_beta = r_dst_level / 2 * np.sqrt(1 - gama ** 2)
    return current_beta

def statistic_for_dispresion(Q, param):
    n_obs = len(Q)
    return np.sum([(Q[i + 1] - param * Q[i]) ** 2 for i in range(n_obs - 1)]) / (2 * n_obs)


def find_parameter_estimates(center: np.array, points: np.array):
    """

    :param center:
    :param points:
    :return:
    """

    distance = []
    for point in points:
        distance.append(
            np.linalg.norm(point[0] - point[1])
        )
    r_dst_pred = distance[-1]

    param1 = 0
    param2 = 1
    count = 0
    while True:
        count += 1
        dispre_1 = statistic_for_dispresion(center, param1)
        dispre_2 = statistic_for_dispresion(center, param2)
        if dispre_1 > dispre_2:

            param1 = (param1 + param2) / 2
            param2 = param2

        elif dispre_1 < dispre_2:

            param2 = (param1 + param2) / 2
            param1 = param1
        else:
            print("param1 == param2")
            break

        if np.abs(param1 - param2) < 1e-10:
            break

    pred_gama = (param1 + param2) / 2
    pred_beta = calculate_beta(r_dst_pred, pred_gama)

    pred_disp = statistic_for_dispresion(center, pred_gama)
    print(f"pred_beta: {pred_beta}, "
          f"pred_gama: {pred_gama}, "
          f"r_dst_pred: {r_dst_pred},"
          f"pred_disp: {pred_disp}")

    return pred_beta, pred_gama, r_dst_pred, pred_disp


def find_best_beta_gama(param_0, func_to_optimized, distance_level):
    bnds = ((0, np.inf), (0, 1))
    result = scipy.optimize.minimize(func_to_optimized, x0=param_0,
                                     constraints={
                                         'type': 'eq',
                                         'fun': lambda x: 2 * x[0] / np.sqrt(1 - x[1] ** 2) - distance_level,
                                     },
                                     bounds=bnds)
    return result.x