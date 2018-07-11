import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pyp
import matplotlib.dates as mpldates
import datetime
import math
import pdb

# HELPER METHODS AND CONSTANTS

conversion_constants = { "nan": None,
                         "TRUE": True,
                         "FALSE": False }

route_difficulty = { "Gray" : [0, "V0"],
                     "Yellow": [1, "V1"],
                     "Green": [2, "V2"],
                     "Red": [3, "V3"],
                     "Blue": [4, "V4"],
                     "Orange": [5, "V5"],
                     "Purple": [6, "V6"],
                     "Black": [7, "V7"] }

# GET DATA, REFORMAT AS NEEDED

# LAMBDAS
datefunc = lambda x: mpldates.date2num(datetime.date(int(x[6:]), int(x[0:2]), int(x[3:5])))
boolfunc = lambda val: True if conversion_constants.get(val.decode()) == True else False
# boolfunc = lambda val: conversion_constants.get(val.decode())

# LOADTXT
activity, route_level, features, injury_type = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(1,2,5,7), dtype='str')
completion, injury, new_route, overhang, handicap = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(4,6,8,9,10), dtype='bool', converters={4: boolfunc, 6: boolfunc, 8: boolfunc, 9: boolfunc, 10: boolfunc})
dates, quantity_time = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(0, 3), converters= {0: datefunc})

# get an np NDarray of unique dates
unique_dates = np.unique(dates)

def _get_difficulty_index(route_color):
    try:
        return route_difficulty[route_color][0]
    except KeyError:
        return None

def get_bubble_arrays(did_you_finish):
    '''
    unique_dates_list = unique_dates.tolist()
    dates_to_indices = {date: idx for idx, date in enumerate(unique_dates_list)}
    data = np.zeros([len(unique_dates_list), len(route_difficulty)])
    for date, level, successes in zip(dates, route_level, quantity_time):
        difficulty_index = _get_difficulty_index(level)
        if difficulty_index is None:
            continue
        data[dates_to_indices[date]][difficulty_index] += successes
    return data
    ''' 
    unique_dates_list = unique_dates.tolist()
    data = []
    for i in range(len(unique_dates)):
        data.append([0 for x in range(len(route_difficulty.keys()))])
    for i in range(len(dates)):
        if activity[i] == 'Bouldering' and completion[i] == did_you_finish:
            count = data[unique_dates_list.index(dates[i])][route_difficulty[route_level[i]][0]]
            data[unique_dates_list.index(dates[i])][route_difficulty[route_level[i]][0]] = count + quantity_time[i]
    return data

bubble_finish = get_bubble_arrays(True)
bubble_failed = get_bubble_arrays(False)

print('\nUnique Date List:\n' + str(unique_dates))
print('Successes:\n' + str(bubble_finish))
print('Failures:\n' + str(bubble_failed))
