import EM_algo as EM_ALGO
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level = logging.INFO)
# logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
file_path = './'
# total_path = file_path + "customer_info"
# demand_data = pd.read_pickle(total_path).head(n=100)


demand_data = pd.read_csv('customer_info.csv').head(20)
logging.info("data loading done, {}".format(demand_data.shape))
demand_data['distance_from_user'] = np.random.randint(1, 20, size=len(demand_data))
demand_data['is_weekend'] = np.random.randint(0, 2, size=len(demand_data))
demand_data['booking_length'] = np.random.randint(8, 75, size=len(demand_data))
demand_data['seats'] = np.random.randint(4, 8, size=len(demand_data))
demand_data['is_automatic'] = np.random.randint(0, 2, size=len(demand_data))
demand_data['is_petrol'] = np.random.randint(0, 2, size=len(demand_data))
demand_data['lead_time'] = np.random.randint(2, 98, size=len(demand_data))
demand_data['is_ftu'] = np.random.randint(0, 2, size=len(demand_data))
demand_data['chosen_cargroup_id'] = demand_data['cargroup_id']
demand_data = demand_data.fillna(1)
demand_data['fee_per_hour'] = demand_data['pricing'] / demand_data['avg_booking_duration']
logging.info("basic feature cleaning done")
demand_data['km_driven'] = demand_data['km_driven'].astype(str).str.replace(',', '')

demand_data['km_driven'] = (demand_data['km_driven'].astype(float))


demand_data['km_driven'] = (demand_data['km_driven'] -
                            demand_data['km_driven'].min()) / (demand_data['km_driven'].max()
                                                               - demand_data['km_driven'].min())

logging.info("km_driven cleanup done")
demand_data['fee_per_hour'] = (demand_data['fee_per_hour'] -
                               demand_data['fee_per_hour'].min()) / (demand_data['fee_per_hour'].max()
                                                                     - demand_data['fee_per_hour'].min())

logging.info("fee_per_hour computation done")
# set number of segments ########################
max_segments_to_try = 5
list_segments = [i for i in range(2, max_segments_to_try)]
tol = 1e-3

logging.info(f"running to optimizer for max_segment: {max_segments_to_try}, "
             f"and tolerance: {tol}")

optimal_beta, optimal_gamma = EM_ALGO.em_algo_for_given_segments(2, demand_data, tol)
