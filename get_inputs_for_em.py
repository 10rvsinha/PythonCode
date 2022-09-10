import pandas as pd
import numpy as np
import pickle as pkl


def get_cargroup_ids(data):
    return np.insert(data['cargroup_id'].unique(), 0, 0)

"""
c_ids_total = get_cargroup_ids(demand_data)
df = demand_data[demand_data.context_device_id == '7ffb1b5468882a27']"""


def get_availability_matrix(data_i, cargroup_ids_vector): # data of customer i
    vector_available_car_groups = data_i.cargroup_id.unique()
    availability_vector = np.zeros(len(cargroup_ids_vector))
    for i in vector_available_car_groups:
        availability_vector[np.where(cargroup_ids_vector == i)[0][0]] = 1
    availability_vector[0] = 1
    return availability_vector


def get_x_vector_for_customer_and_cargroup(dataframe_for_i_j):

    # X_i[:][j] = vector of length 9 with [price, distance, booking type, booking length, seating, is automatic,
    #                                      is_petrol, car quality_km, car age years]
    if dataframe_for_i_j.shape[0] > 0:
        return np.array([dataframe_for_i_j.fee_per_hour.iloc[0], dataframe_for_i_j.distance_from_user.iloc[0],
                         dataframe_for_i_j.is_weekend.iloc[0],
                         dataframe_for_i_j.booking_length.iloc[0],
                         dataframe_for_i_j.seats.iloc[0], dataframe_for_i_j.is_automatic.iloc[0],
                         dataframe_for_i_j.is_petrol.iloc[0],
                         dataframe_for_i_j.km_driven.iloc[0]])
    else:
        return np.zeros(8)


def get_user_attributes_for_i(data_i):
    # z = [day_type, lead time, booking length, FTU/RTU ]
    return [data_i.iloc[0].is_weekend, data_i.iloc[0].lead_time,
            data_i.iloc[0].booking_length, 1 - data_i.iloc[0].is_ftu]


def get_car_chosen_by_customer_i(data_i):  # i is context_device_id # return cargroup_id_chosen
    df = data_i[data_i['is_booked'] == 1]
    if df.shape[0] > 0:
        return df.iloc[0]['chosen_cargroup_id']
    else:
        return 0
