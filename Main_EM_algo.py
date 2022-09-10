import numpy as np
import pandas as pd
import em_utility_set_up as utilify_fns


file_path = 'C:/Users/Neha Sharma/Google Drive/ZoomCar_data/ZoomCar_data/inquiry_data_2019/'
total_path = file_path + "demand_data_cleaned_for_em"
demand_data = pd.read_pickle(total_path)


# set number of segments ########################
s = 2

# initialize inputs
utilify_fns.initialize_inputs(demand_data, s)

