import pandas as pd
import numpy as np
import math
import os

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)
np.seterr(divide='ignore', invalid='ignore')


def generate_random_dots(min_value,max_value,number_of_dots):
    """

    :param min_value:
    :param max_value:
    :param number_of_dots:
    :return:
    """
    return np.random.randint(min_value,max_value,(2,number_of_dots)).T



def generate_ksi_similar_coordinate(number_of_data):
    """

    :param number_of_data:
    :return:
    """
    random_x_y = list(np.random.normal(0,1, 2))
    return np.array([random_x_y for i in range(number_of_data)])


def circulation_center(points, k=None):
    """
    k - circulation
    
    """
    if k is None: k = np.array([[1]*len(points)])
    return np.sum(points* np.append(k, k, axis=0).T ,axis= 0)/ np.sum(k)

def circulation_moment(points):
    """

    :param points:
    :return:
    """
    return np.sum([point**2 for point in points])

def circulation_dispersion(points):
    """

    :param points:
    :return:
    """
    calculate_circulation_center = circulation_center(points)
    return np.sum([(point - calculate_circulation_center)**2 for point in points])


def calculate_the_measure(zi, zj):
    """

    :param zi:
    :param zj:
    :return:
    """
    return math.hypot(zj[0] - zi[0] , zj[1]  - zi[1])



def caculate_the_interaction(z_input, i):
    """

    :param z_input:
    :param i:
    :return:
    """
    
    U = np.array([[0, 1],
                 [-1, 0]])
    n_whirlwind = len(z_input)
    
    sum_of_interaction = []
    for j in range(n_whirlwind):
        if i!=j:
            interaction = np.dot(U, (z_input[i] - z_input[j]))
            measure_between_points = calculate_the_measure(z_input[i], z_input[j])
            measure_between_points = np.nan_to_num(measure_between_points)
            interaction = interaction / measure_between_points
            interaction = np.nan_to_num(interaction)
            sum_of_interaction.append(interaction)

    return sum(sum_of_interaction)                          

def next_z_generate_point(z_input, ksi, betta, gamma, use_second_interaction=False):
    """

    :param z_input:
    :param ksi:
    :param betta:
    :param gamma:
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
                if k!=i:

                    result_second_intersaction = betta**2 / 2

                    intersection_i_m = caculate_the_interaction(z_input, i)
                    intersection_k_m = caculate_the_interaction(z_input, k)

                    second_sum_of_interaction_calculation.append(
                                result_second_intersaction * (intersection_i_m - intersection_k_m)
                        )

            second_weight_interaction = np.sum(second_sum_of_interaction_calculation)

        
        z_new.append(gamma* z_input[i] + weight_interaction + second_weight_interaction + ksi[i] ) # - alfa * (z_input[i] - circulation_center(z_input)))
    
    return np.stack(z_new)



def save_div(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return np.zeros(len(x))





def generate_the_whirlwind(iteration: int, z: np.array, beta_0: int, gama_0: int, config_dict: dict):
    """

    :param iteration:
    :param z:
    :param beta_0:
    :param gama_0:
    :param config_dict:
    :return:
    """
    
    Q_center_list, L_moment_list, D_dispersion_list = [], [], []

    betta = beta_0 
    gamma = gama_0 
    z_new = z.copy()

    betta_list, gamma_list = [], []
    z_memory = []
    R_distance = []
    

    for i in range(iteration):
        
        betta_list.append(betta)
        gamma_list.append(gamma)
        
        z_memory.append(z_new)

        Q_center_list.append(circulation_center(z_new))
        L_moment_list.append(circulation_moment(z_new))
        D_dispersion_list.append(circulation_dispersion(z_new))

        R_distance.append(np.linalg.norm(z_new[0]-z_new[1]))


    
        if config_dict['use_the_random_noise']:
            ksi = generate_ksi_similar_coordinate(len(z))
        else:
            ksi = np.zeros(len(z))

        z_new = next_z_generate_point(z_new, ksi, betta, gamma, config_dict['use_second_interaction'])
        

    
    main_measure_dict = {
        'z': z_memory,
        'r_distance': R_distance,
        'center': Q_center_list,
        'moment': L_moment_list,
        'dispersion': D_dispersion_list,
        'betta': betta_list,
        'gamma': gamma_list
    }
    
    return main_measure_dict



def create_the_folder(dir_name):
    """
    Parameter:
        dir_name:
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory " , dir_name ,  " Created ")
    else:
        print("Directory " , dir_name ,  " already exists")



