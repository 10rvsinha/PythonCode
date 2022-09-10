# car group id list
# i - indexes customers, j-indexes car groups, s- indexes customer segment 
import math
import get_inputs_from_data as get_inputs

import numpy as np

s = 2 # input any number >= 2 for number f segments.
############# Inputs from data ############################################
# data = pd.read_csv(data_file)
I = get_inputs.get_cargroup_ids(data) #I = [0, 1, 2, 7, 10, 28, 38, 60, 65, 66, 67, 68] # cargroup ids, first element will always be 0 representing null
n_car_groups = len(I)

# Xij represents attributes of cars shown on the search page
# Xij = [price, distance, booking type, booking length, car type, car quality ] # for a particular car group 6 attributes.
# X_i  is a 9 x len(I) where I = number of car groups, 6 rows for attributes ^  len(I) columns for each car group ( this is a fat matrix not tall )
"""
booking_type: 0 -> weekday | 1 -> weekend, defined as 1 if booking start time and end time contains a Saturday or Sunday or public holiday
car_type: 1,2,3,4,5
car_quality: 1 best, 0 worst
"""

beta_mat = {seg : np.zeros((len(I), 9)) for seg in range(s)} # this dictinary contains beta_s for each segement. 
beta_seg = beta_mat[seg] #  len(I) x 9 dimension, 9 attributes, this is how segment seg customers percieve a car group on the 9 attributes 

#Y is len(s) x 4 matrix and then each row of this matrix means that for segment s, what is the weight on each of 4 attributes. 
Y = np.zeros((s, 4))
# Y[seg][:] should be 1 x 4, for a segment s.

def get_car_chosen_by_customer_i(data, i): # i is context_device_id # return cargroup_id_chosen
    df = data[(data['context_device_id'] == i) & (data['is_booked'] == 1)]
    if df.shape[0] > 0:
        return df['car_group_id']
    else:
        return 0
    


def get_customer_related_information(data, I, i):
  # first get the vector A_i of length I, i.e the cargroups that are shown to customer i should have 1 and ones that are not shown are 0.
  # here the user i chose car groups, 0(null), 1, 38, 60, 65, 66, 67, 68
  Ai = get_inputs.get_availability_matrix(data, I) #Ai = [1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1] 
                                                    # A_i length must be equal to n_car_groups and first element is 1. #Availability matrix
  X_i = np.zeros((9, len(I)))
  for j in A_i:
    df_i_j = data[(data['context_device_id'] == i) & (data['cargroup_id'] == j)]
    X_i[:][j] = get_inputs.get_x_vector_for_customer_and_cargroup(df_i_j) 
    #[80, 4, 0, 18, 3, 0.9] # example here, definite appropriate function that gets this from data. 
   
  # Z = [ day_type, lead time, booking length, FTU/RTU ] # 4 attributes and of dimension 4 x 1. 
  """
  day_type: 0 -> weekday | 1 -> weekend
  FTU -> 0
  RTU -> 1
  """
  Z_i = [0, 18, 24, 0] # attributes for a particular user i. 
  lambda_i = np.zeros(len(I))
  index_chosen_cargroup = np.where(I == get_car_chosen_by_customer_i(data, i))
  lambda_i[index_chosen_cargroup] = 1
  
  return A_i, X_i, Z_i, lambda_i 



def get_p_ij_for_seg(i, j, A_i, seg, X_i, beta_mat):
    den_s = 0
    for k in I:
        den_s += A_i[k] * (math.exp((beta_mat[seg][k][:] * X_i[:][k]))) # given segment s, denominator for P_{ij}|s.
        
    return A_i[j] * math.exp((beta_mat[seg][j][:] * Xi[:][j])) / den_s    


def get_p_i_s(i, seg, Z_i, Y):
    sum = 0
    for seg in range(s):
        sum += math.exp(Y[seg][:] * Z_i)

    return math.exp(Y[seg][:] * Z_i) / sum   # probability that customer i belongs to segment s.

def get_li_seg(i, j, A_i, seg, X_i, lambda_i, beta_mat):
  # Lis
  p_i_j_seg = get_p_ij_for_seg(i, j, A_i, seg, X_i, beta_mat)
  arr = []
  available_car_groups = I(np.where(A_i==1)) # car_ids shown to customer i
  for j in available_car_groups:
      arr.append(p_i_j_seg ** lambda_i[j])

  L_i_seg = np.prod(arr)
  return L_i_seg 


def get_p_i_s_bar(seg, P_i_s, L_i_s): # P_i_s and L_i_s is column array of length = len(s) 
    arr = 0
    for k in range(s):
        arr += P_i_s[k] * L_i_s[k]
        
    return P_i_s[seg] * L_i_s[seg] / arr # equivalent to p_tilde in notes
 
  
def l_start_obj_func(data, I, p_bar, beta_mat, Y): # p_bar is the matrix of N x len(S) where an element p_bar[i, s] = p_i_s_bar.
    arr = 0
    for i in range(N):
        A_i, X_i, Z_i, lambda_i = get_customer_related_information(data, I, i)
        for seg in range(s):
            arr += p_bar[i, s] * (np.log(get_li_seg(i, j, A_i, seg, X_i, lambda_i, beta_mat)) + np.log(get_p_i_s(i, seg, Z_i, Y)))
    return arr        




############# Algorithm ########################
s = 2
beta_mat, Y = {seg : np.zeros((len(I), 6)), np.zeros((s, 4))}
N = 100
tol = 1e-5
p_bar = np.zeros((N, s))
diff_beta, diff_y = 10, 10
               
while diff_beta > tol | diff_y > tol :            
               
    for i in range(N):
        P_i, L_i = np.zeros(s), np.zeros(s)
        A_i, X_i, Z_i, lambda_i = get_customer_related_information(data, I, i)
        beta_mat_new, Y_new = beta_mat, Y
        for seg in range(s):
            P_i[seg]  =  get_p_i_s(i, seg, Z_i, Y_new)
            L_i[seg]  =  get_li_seg(i, j, A_i, seg, X_i, lambda_i, beta_mat_new)
        for seg in range(s):
            p_i_s = get_p_i_s_bar(seg, P_i, L_i)

        objective_fn = functools.partial(l_start_obj_func, data, I, p_bar) # this is the objective function as a function of beta_mat and Y.

        #### find the optimal beta and Y for objective function, #########
        ### beta_mat, Y = arg max objective_fn 
        diff_beta, diff_y = 0, 0
        for element in len(beta_mat):
            diff_beta += (beta_mat[element] - beta_mat_new[element])**2
        for element in len(Y):
            diff_y += (Y[element] - Y_new[element])**2
    
 
        
               
              








