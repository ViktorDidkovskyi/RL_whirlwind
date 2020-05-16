import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.getcwd() + '/src' )

from utils import (generate_the_whirlwind, create_the_folder,
                 generate_random_dots)
from utils_plot import plot_the_whirlwind, plot_metrics

from solving_problem import (calculate_beta, find_best_beta_gama, find_parameter_estimates)
from clustering import make_clustering, train_model_to_predict_clusters
from sklearn.cluster import DBSCAN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# def calculate_dispersion(x_change_list):
    # return np.sum([np.sum(x**2) for x in x_change_list])/(2*len(x_change_list))


min_value,max_value = -50, 50
number_of_dots = 6
beta_0, gamma_0 = 0.9, 0.54 #0.45

config_dict = {
    'experiment_type': "best_beta_gamma",
    'use_second_interaction': False,
    'use_the_random_noise': False,
    'generete_on_circule': False

}


dir_name = os.getcwd() + "/../output/beta_random"
create_the_folder(dir_name)




### generate z or z circule
if config_dict['generete_on_circule']:
    R = 1
    z = np.array([(R * np.cos(2 * np.pi / number_of_dots * j), R * np.sin(2 * np.pi / number_of_dots * j)) for j in
                  range(number_of_dots)])
else:
    z = generate_random_dots(min_value,max_value,number_of_dots=number_of_dots)



#################### evolution ####################
main_measure_dict = generate_the_whirlwind(500, z, beta_0, gamma_0, config_dict)
filter_z_memory = [main_measure_dict['z'][i] for i in range(0,500,5)]
plot_the_whirlwind(filter_z_memory, path_to_save=dir_name + "/base_graph")# os.path.join(output_path, "default_evolution"))


for type_graph in list(main_measure_dict.keys())[2:]:
    try:
        plot_graph = main_measure_dict[type_graph]
        if plot_graph != []:
            path_to_save = os.path.join(dir_name, type_graph)
            plot_metrics(plot_graph, type_graph, path_to_save=path_to_save)
    except ValueError:
        continue

#################### find estimention ####################


r_dst_level = 2
beta = calculate_beta(r_dst_level, gamma_0)

## find parameters if have evolution
config_dict['use_the_random_noise'] = True


main_measure_dict = generate_the_whirlwind(500, z, beta, gamma_0, config_dict)
pred_beta, pred_gama, r_dst_pred, pred_disp = find_parameter_estimates(main_measure_dict['center'],
                                                                       main_measure_dict['z'])






#################### clustering ####################

config_dict['use_the_random_noise'] = False
z = generate_random_dots(min_value,max_value,number_of_dots=35)
main_measure_dict = generate_the_whirlwind(700, z, beta_0, gamma_0, config_dict)



stack_evolution = np.stack(main_measure_dict['z'][30:])
X = np.concatenate(stack_evolution)

algo_cluster = DBSCAN(eps=0.95, min_samples=13) ## hyper parameters
y = make_clustering(X, algo_cluster, info=True, path_to_save='')#os.path.join(dir_name, "cluster_evolution_35vortex_wt_noise.jpg"))



X, y = shuffle(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
rf = train_model_to_predict_clusters(check_values=X, embeding_values=y)


#################### clustering  testing ####################
z_testing = generate_random_dots(min_value,max_value,number_of_dots=35)
main_measure_dict = generate_the_whirlwind(600, z_testing, beta_0, gamma_0, config_dict)

dataset_evolution_testing = np.concatenate(np.stack(main_measure_dict['z'][30:]))


df_level = pd.DataFrame(rf.predict_proba(X), columns=[str(c) for c in np.unique(y)])
df_level['level'] = rf.predict(X)
df_level['vortex'] = np.stack([list(range (stack_evolution.shape[1])) for i in range(stack_evolution.shape[0])]).flatten()
df_level['vortex'] = df_level['vortex'].astype(str)

result = pd.crosstab(df_level['level'], df_level['vortex'], df_level['vortex'], aggfunc='count' , normalize='columns')#.reset_index()

print(result.idxmax().reset_index().groupby(0)['vortex'].apply(lambda x: ' '.join(x)).reset_index().rename(columns={0:'level'}))

#################### solve optimum control problem ####################
z = generate_random_dots(min_value,max_value,number_of_dots=2)

distance_level = 25
sigma = 1
func_to_optimized = lambda x: (x[0] - gamma_0) ** 2 + (x[1] - beta_0) ** 2 + sigma ** 2 / (
            1 - x[1] ** 2) + 1 / gamma_0


best_betta, best_gamma = find_best_beta_gama((beta_0, gamma_0), func_to_optimized, distance_level)

print(best_betta, best_gamma)

main_measure_dict = generate_the_whirlwind(600, main_measure_dict['z'][-1], best_betta, best_gamma, config_dict)
filter_z_memory = [main_measure_dict['z'][i] for i in range(0,300,1)]
plot_the_whirlwind(filter_z_memory, path_to_save =os.path.join(dir_name, "optt_evolution_w_noise_"))


