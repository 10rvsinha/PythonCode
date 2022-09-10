import pandas as pd

df = pd.read_csv("available_cars_city_1.csv")

av_1 = list(df.av_inv_0_24.dropna().unique())
av_2 = list(df.av_inv_24_48.dropna().unique())
av_3 = list(df.av_inv_48_72.dropna().unique())

print(len(av_1), len(av_2), len(av_3))
print(len(av_1) + len(av_2) + len(av_3))
av_1.extend(av_2)
av_1.extend(av_3)
print(len(av_1))

car_id_list = list(set(av_1))
print(len(car_id_list))


car_id_str = ",".join([str(int(x))for x in car_id_list])
print(car_id_str)
