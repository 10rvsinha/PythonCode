from scipy.optimize import minimize
from functools import partial
import em_util_set_up as utility_fns
import get_inputs_for_em as get_inputs
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

tol = 1e-3
max_iter = 3


def objective_function(number_inquiries, number_segments, num_cargroups, p_is_tilda, partial_fn, vars):
    logging.info(f"creating objective function")
    x = 0
    for i in range(number_inquiries):
        for seg in range(number_segments):
            x += p_is_tilda[i, seg] * (np.log(partial_fn(vars)[0])[i, seg] + np.log(partial_fn(vars)[1])[i, seg])

    logging.info(f"objective function creation done")
    return -x


def get_p_is_l_is(data, all_cargroup_id_vec, number_inquiries, number_segments, num_cargroups, vars):
    beta_var = vars[: num_cargroups * number_segments * 8].reshape((number_segments, num_cargroups, 8))
    gamma_var = vars[num_cargroups * number_segments * 8:].reshape((number_segments, 4))
    p_i_s_arr = np.zeros((number_inquiries, number_segments))
    l_i_s_arr = np.zeros((number_inquiries, number_segments))
    i = 0
    for name, data_i in data.groupby(by=['created_at', 'context_device_id']):
        a_i, x_i, z_i, choice_vector_i = utility_fns.get_customer_related_information(all_cargroup_id_vec, data_i)
        for seg in range(number_segments):
            p_i_s_arr[i, seg] = utility_fns.get_p_i_s(seg, number_segments, z_i, gamma_var)
            l_i_s_arr[i, seg] = utility_fns.get_li_seg(a_i, seg, x_i, all_cargroup_id_vec,
                                                       choice_vector_i, beta_var)

        i += 1
    return p_i_s_arr, l_i_s_arr


def optimize_max_likelihood_fn(data, num_inquiries, num_all_cargroups, num_segments, car_ids, p_is_tilda):
    logging.info(f"inquiry count: {num_inquiries}, car group count: {num_all_cargroups}, "
                 f"segment count : {num_segments}, p_is_tilda: {p_is_tilda}")
    x_0 = np.zeros(num_segments * num_all_cargroups * 8 + num_segments * 4)
    # x_0 = np.array([np.zeros((num_segments, num_all_cargroups, 8)), np.zeros((num_segments, num_all_cargroups, 8))])
    # np.zeros((num_segments, 4))
    partial_fn_arr = partial(get_p_is_l_is, data, car_ids, num_inquiries, num_segments, num_all_cargroups)

    cons_1 = {'type': 'eq',
              'fun': lambda x: np.array([x[: num_all_cargroups * num_segments * 8].reshape((num_segments,
                                                                                            num_all_cargroups, 8))
                                         [segment, 0, :] == 1 for segment in range(num_segments)],
                                        dtype=object),
              'jac': lambda x: np.array([1 for i in range(num_segments)], dtype=object)}
    cons_2 = {'type': 'eq',
              'fun': lambda x: np.array(
                  [x[num_all_cargroups * num_segments * 8:].reshape((num_segments, 4))[0, :] == 1],
                  dtype=object),
              'jac': lambda x: np.array([1], dtype=object)}

    logging.info(f"triggering minimizer")
    solution = minimize(partial(objective_function, num_inquiries, num_segments, num_all_cargroups,
                                p_is_tilda, partial_fn_arr), x_0, method='SLSQP',
                        options={'ftol': tol, 'disp': True, 'maxiter': max_iter}, constraints=[cons_1, cons_2])

    logging.info("solution run is done")
    if solution.success:
        print('solution found')
        print(solution.x, -solution.fun)
        return solution.x[: num_all_cargroups * num_segments * 8].reshape(
            (num_segments, num_all_cargroups, 8)), solution.x[num_all_cargroups * num_segments * 8:].reshape(
            (num_segments, 4))

    else:
        print('No solution')
