import pandas as pd
import pickle
import datetime
import math
import numpy as np

file_path = 'C:/Users/Neha Sharma/Google Drive/ZoomCar_data/ZoomCar_data/'

public_holidays = pd.read_csv(file_path + 'public_holidays.csv')
public_holidays['DATE'] = pd.to_datetime(public_holidays.DATE, infer_datetime_format=True, errors='coerce')
public_holiday_list = public_holidays['DATE'].tolist()
demand_data = pd.read_csv(file_path + 'inquiry_data_2019/final_search_data_for_EM_algo.csv')


def get_car_group_name_id_dict(data):
    df = data[['cargroup_id', 'car_name', 'car_brand']].drop_duplicates()
    df = df.rename(columns={'cargroup_id': 'chosen_cargroup_id'})
    df['cargroup_name'] = df['car_brand'] + " " + df['car_name']
    data = pd.merge(data, df[['cargroup_name', 'chosen_cargroup_id']], how='left', on='cargroup_name')
    return data


def is_weekend_demand(row):
    start_date = row.actual_starts.date()
    end_date = row.actual_ends.date()
    if math.isnan((end_date - start_date).days):
        x = None
    elif (end_date - start_date).days == 0:
        if start_date.weekday() >= 5 or start_date in public_holiday_list:
            x = True
        else:
            x = False
    else:
        dates_list = [start_date + datetime.timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        x = False
        for i in dates_list:
            if i.weekday() >= 5 or i in public_holiday_list:
                x = True
                break

    return x


def get_distance(row):
    result_index = row.car_location.find(' km ')
    return float(row.car_location[0:result_index])


def num_weekday_weekend_hours(row):
    weekend_hours = 0
    weekday_hours = 0
    if row.is_weekend == False:
        weekday_hours = row.booking_length
    else:
        dates_list = [row.actual_starts.date() + datetime.timedelta(days=x) for x
                      in range((row.actual_ends.date() - row.actual_starts.date()).days + 1)]
        dates_list = [datetime.datetime.combine(i, datetime.datetime.min.time()) for i in dates_list]  # convert to
        # datetime
        dates_list[0], dates_list[-1] = row.actual_starts, row.actual_ends
        for i in range(len(dates_list) - 1):
            if dates_list[i].weekday() == 5:
                weekend_hours += (dates_list[i + 1] - dates_list[i]).total_seconds() / 3600
            else:
                weekday_hours += (dates_list[i + 1] - dates_list[i]).total_seconds() / 3600
    return weekday_hours, weekend_hours


def cast_data_types(data):
    # data.search_start_timestamp = pd.to_numeric(data.search_start_timestamp)
    # data.search_end_timestamp = pd.to_numeric(data.search_end_timestamp)
    data['actual_starts'] = pd.to_datetime(data['search_start_timestamp'], unit='ms', errors='ignore')
    data['actual_ends'] = pd.to_datetime(data.search_end_timestamp, unit='ms', errors='ignore')
    data['created_at'] = pd.to_datetime(data.received_at, errors='ignore', infer_datetime_format=True).\
        dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    return data


def add_new_columns(data):
    # data['start_date'] = data.actual_starts.dt.date
    # data['start_date'] = pd.to_datetime(data.start_date)
    data['booking_length'] = (data.actual_ends - data.actual_starts).dt.total_seconds() / 3600
    data['fee_per_hour'] = data.pricing / data.booking_length
    data['lead_time'] = (data.actual_starts - data.created_at).dt.total_seconds() / 3600  + 11
    data['is_weekend'] = data.apply(is_weekend_demand, axis=1)
    data['is_weekend'] = data['is_weekend'].astype(int)
    data['weekday_hours'], data['weekend_hours'] = zip(*data.apply(num_weekday_weekend_hours, axis=1))
    data['distance_from_user'] = data.apply(get_distance, axis=1)
    data['car_accessories'] = data['car_accessories'].str[1:-1].str.split(',')
    data['car_transmission'] = data['car_accessories'].str[0].str[1:-1]
    data['fuel_type'], data['seats'] = data['car_accessories'].str[1].str[2:-1], data['car_accessories'].str[2].str[2]

    data['is_automatic'] = np.where(data['car_transmission'] == 'Automatic', 1, 0)
    data['is_petrol'] = np.where(data['fuel_type'] == 'Petrol', 1, 0)
    data['is_ftu'] = np.where(data['user_type'] == 'FTU', 1, 0)
    # data[['car_transmission','fuel_type', 'seats']] = data['car_accessories'].str.rsplit(",", n=2, expand=True)
    return data


demand_data = cast_data_types(demand_data)
demand_data = add_new_columns(demand_data)
demand_data = get_car_group_name_id_dict(demand_data)
pickle.dump(demand_data, open(file_path + "demand_data_cleaned_for_em", 'wb'))

