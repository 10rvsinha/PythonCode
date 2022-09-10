import numpy as np
import pandas as pd
import copy
import em_util_set_up as utility_fns
import get_inputs_for_em as input_fns
import optimization_for_em as optimize
import logging

logging.basicConfig(level = logging.INFO)

# initialize inputs
def em_algo_for_given_segments(num_seg, data, tolerance):
    vector_of_all_car_ids, beta_init, gamma_init = utility_fns.initialize_inputs(data, num_seg)
    num_car_ids_total = len(vector_of_all_car_ids)
    num_customer_inquiries = data.groupby(by=['created_at', 'context_device_id']).ngroups
    p_i_s_tilda_arr = np.zeros((num_customer_inquiries, num_seg))
    p_i_s_arr = np.zeros((num_customer_inquiries, num_seg))
    L_i_s_arr = np.zeros((num_customer_inquiries, num_seg))

    gamma, beta, i = copy.copy(gamma_init), copy.copy(beta_init), 0
    gamma_sol, beta_sol = gamma + .1, beta + .1
    # compute the p_is_tilda_for given beta gamma.

    while abs(np.sum(np.subtract(gamma, gamma_sol))) > tolerance or \
            abs(np.sum(np.subtract(beta, beta_sol))) > tolerance:

        gamma, beta = copy.copy(gamma_sol), copy.copy(beta_sol)
        count = 0
        for name, data_i in data.groupby(by=['created_at', 'context_device_id']):
            count += 1
            logging.info(f"parsing data, row count: {count}")
            a_i, x_i, z_i, choice_vector_i = utility_fns.get_customer_related_information(vector_of_all_car_ids, data_i)
            for seg in range(num_seg):
                p_i_s_arr[i, seg] = utility_fns.get_p_i_s(seg, num_seg, z_i, gamma)
                L_i_s_arr[i, seg] = utility_fns.get_li_seg(a_i, seg, x_i, vector_of_all_car_ids,
                                                           choice_vector_i, beta)
            for seg in range(num_seg):
                p_i_s_tilda_arr[i, seg] = utility_fns.get_p_i_s_bar(seg, num_seg, p_i_s_arr[i, :], L_i_s_arr[i, :])

            i += 1

        # optimize given p_is tilda

        logging.info(f"all rows have been parsed, triggering likelihood function")
        beta_sol, gamma_sol = optimize.optimize_max_likelihood_fn(data, num_customer_inquiries,
                                                                  num_car_ids_total, num_seg, vector_of_all_car_ids,
                                                                  p_i_s_tilda_arr)

        print('beta_sol=', beta_sol)

    return beta_sol, gamma_sol
