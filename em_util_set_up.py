# car group id list
# i - indexes customers, j-indexes car groups, s- indexes customer segment
import math
import get_inputs_for_em as get_inputs
import numpy as np
import pandas as pd


# initialize inputs given data and segments#
def initialize_inputs(data, s):
    all_cargroup_id_vec = get_inputs.get_cargroup_ids(data)  # all_cargroup_id_vec = [0, 1, 2, 7, 10, 28]
                                                            # cargroup ids, first
                                                            # element will always be 0 representing null
    beta_mat = np.zeros((s, len(all_cargroup_id_vec), 8))
    #{seg: np.zeros((len(all_cargroup_id_vec), 9)) for seg in range(s)}  # this dictinary contains beta_s for each segement.
    gamma_mat = np.zeros((s, 4))
    # Xij represents attributes of cars shown on the search page
    # Xij = [price, distance, booking type, booking length, car type, car quality ] # for a particular car group 6 attributes.
    # X_i  is a 9 x len(all_cargroup_id_vec) where all_cargroup_id_vec = number of car groups, 6 rows for attributes ^  len(all_cargroup_id_vec) columns for each car group ( this is a fat matrix not tall )
    """
    booking_type: 0 -> weekday | 1 -> weekend, defined as 1 if booking start time and 
    end time contains a Saturday or Sunday or public holiday
    
    car_type: 1,2,3,4,5
    car_quality: 1 best, 0 worst
    """
    """
    for segment in s:
        #beta_segment = beta_mat[segment]
        # len(all_cargroup_id_vec) x 9 dimension, 9 attributes, this is how segment 
        seg customers percieve a car group on the 9 attributes 
    """
    """Y is len(s) x 4 matrix and then each row of this 
    matrix means that for segment s, what is the weight on each of 4 attributes. 
    """
    return all_cargroup_id_vec, beta_mat, gamma_mat


def get_customer_related_information(all_cargroup_id_vec, data_i):
    """first get the vector A_i of length I, i.e the cargroups that are shown to customer i should have 1 and ones that are not shown are 0.
    here the user i chose car groups, 0(null), 1, 38, 60, 65, 66, 67, 68 """

    A_i = get_inputs.get_availability_matrix(data_i, all_cargroup_id_vec)  # Ai = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # A_i length must be equal to n_car_groups and first element is 1. #Availability matrix
    X_i = np.zeros((8, len(all_cargroup_id_vec)))
    for j in range(len(all_cargroup_id_vec)):
        df_i_j = data_i[data_i['cargroup_id'] == all_cargroup_id_vec[j]]
        X_i[:, j] = get_inputs.get_x_vector_for_customer_and_cargroup(df_i_j).flatten()
        # [80, 4, 0, 18, 3, 0.9] # example here, definite appropriate function that gets this from data.

    # Z = [day_type, lead time, booking length, FTU/RTU ] # 4 attributes and of dimension 4 x 1.
    """
    day_type: 0 -> weekday | 1 -> weekend
    FTU -> 0
    RTU -> 1
    """
    z_i = get_inputs.get_user_attributes_for_i(data_i) # [0, 18, 24, 0]  # attributes for a particular user i.
    choice_vector_i = {car_id: 0 for car_id in all_cargroup_id_vec}
    choice_vector_i[get_inputs.get_car_chosen_by_customer_i(data_i)] = 1
    #choice_vector_i = np.zeros(len(all_cargroup_id_vec))
    #index_chosen_cargroup = np.where(all_cargroup_id_vec == get_inputs.get_car_chosen_by_customer_i(data_i))
    #choice_vector_i[index_chosen_cargroup] = 1

    return A_i, X_i, z_i, choice_vector_i


def get_p_ij_for_seg(j, A_i, seg, X_i, all_cargroup_id_vec, beta_mat):
    den_s = 0
    for k in range(len(all_cargroup_id_vec)):
        den_s += A_i[k] * math.exp(np.matmul(beta_mat[seg, k, :], X_i[:, k]))  # given segment s, denominator for P_{ij}|s.
        # beta_mat[s] is n_cargroup * 9 vector and X_i is 9 * n_cargroup vector
        # beta_mat[seg][k][:] * X_i[:][k] is 1* 9 multiplied by 9*1 vector.

    return A_i[j] * math.exp(np.matmul(beta_mat[seg, j, :], X_i[:, j])) / den_s


def get_p_i_s(seg_i, s, Z_i, Y):
    den = 0
    for segment in range(s):
        den += math.exp(np.matmul(Y[segment, :], Z_i)) # 1*4 multiplied by 4*1

    return math.exp(np.matmul(Y[seg_i, :], Z_i)) / den  # probability that customer i belongs to segment s.


def get_li_seg(A_i, seg, X_i, all_cargroup_id_vec, lambda_i, beta_mat): # lambda_i is choice vector
    # Lis
    arr = []
    available_car_groups = all_cargroup_id_vec[np.where(A_i == 1)]  # car_ids shown to customer i
    for car_id in available_car_groups:
        j = np.where(all_cargroup_id_vec == car_id)[0] # index of this car_id in total list of carids
        p_i_j_seg = get_p_ij_for_seg(j, A_i, seg, X_i, all_cargroup_id_vec, beta_mat)
        arr.append(p_i_j_seg ** lambda_i[car_id])

    L_i_seg = np.prod(arr)
    return L_i_seg


def get_p_i_s_bar(seg, s, P_i_s, L_i_s):  # P_i_s and L_i_s is column array of length = len(s)
    arr = 0
    for k in range(s):
        arr += P_i_s[k] * L_i_s[k]

    return P_i_s[seg] * L_i_s[seg] / arr  # equivalent to p_tilde in notes

