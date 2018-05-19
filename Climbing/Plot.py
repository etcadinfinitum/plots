import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyp
import matplotlib.dates as mpldates
import datetime

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

def get_level_number(color):
    return route_difficulty.get(color)[0]

def get_level_name(color):
    return route_difficulty.get(color)[1]

def get_date_number_equivalent(date):
    month = date[0:2]
    day = date[3:5]
    num_rep = (month * 100) + day
    return num_rep

# GET DATA, REFORMAT AS NEEDED

datefunc = lambda x: mpldates.date2num(datetime.date(int(x[6:]), int(x[0:2]), int(x[3:5])))

activity, route_level, features, injury_type = np.loadtxt("./climbing-metrics-Route-specs.csv", delimiter=",", skiprows=1, unpack=True, usecols=(1,2,5,7), dtype='str')

pre_completion, pre_injury, pre_new_route, pre_overhang, pre_handicap = np.loadtxt("./climbing-metrics-Route-specs.csv", delimiter=",", skiprows=1, unpack=True, usecols=(4,6,8,9,10), dtype='str')

inter_completion = []
inter_injury = []
inter_new_route = []
inter_overhang = []
inter_handicap = []

def convert_boolean_arrays():
    for val in pre_completion:
        if val != 'nan':
            inter_completion.append(conversion_constants.get(val))
        else:
            inter_completion.append(False)
    for val in pre_injury:
        if val != 'nan':
            inter_injury.append(conversion_constants.get(val))
        else:
            inter_injury.append(False)
    for val in pre_new_route:
        if val != 'nan':
            inter_new_route.append(conversion_constants.get(val))
        else:
            inter_new_route.append(False)
    for val in pre_overhang:
        if val != 'nan':
            inter_overhang.append(conversion_constants.get(val))
        else:
            inter_overhang.append(False)
    for val in pre_handicap:
        if val != 'nan':
            inter_handicap.append(conversion_constants.get(val))
        else:
            inter_handicap.append(False)

convert_boolean_arrays()
completion = np.array(inter_completion)
injury = np.array(inter_injury)
new_route = np.array(inter_new_route)
overhang = np.array(inter_overhang)
handicap = np.array(inter_handicap)

dates, quantity_time = np.loadtxt("./climbing-metrics-Route-specs.csv", delimiter=",", skiprows=1, unpack=True, usecols=(0, 3), converters= {0: datefunc})

def summarize_route_completion_by_date(fin_or_fail):
    new_dict = {}
    for i in range(len(dates)):
        if new_dict.get(dates[i], None) == None:
            new_dict.update({dates[i]: {}})
        if completion[i] == fin_or_fail and activity[i] == 'Bouldering':
            if new_dict.get(dates[i]).get(route_level[i], None) == None:
                new_dict.get(dates[i]).update({route_level[i]: quantity_time[i]})
            else: 
                curr_count = new_dict.get(dates[i]).get(route_level[i])
                curr_count += quantity_time[i]
                new_dict.get(dates[i])[route_level[i]] = curr_count
    return new_dict

# summarize completed routes by date, level
finished_routes = summarize_route_completion_by_date(True)
unfinished_routes = summarize_route_completion_by_date(False)

print("Finished routes (full dict): " + str(finished_routes))
print("Failed routes (full dict): " + str(unfinished_routes))

route_dates = []
finished_routes_gray = []
finished_routes_yellow = []
finished_routes_green = []
unfinished_routes_gray = []
unfinished_routes_yellow = []
unfinished_routes_green = []

# prep dictionaries for concise plotting?
def convert_route_completion_summary():
    for key in sorted(unfinished_routes):
        if not key in route_dates:
            route_dates.append(key)
        route_color = 'Gray'
        if unfinished_routes.get(key).get(route_color, None) == None:
            unfinished_routes_gray.append(0)
        else:
            unfinished_routes_gray.append(unfinished_routes.get(key).get(route_color))
        route_color = 'Yellow'
        if unfinished_routes.get(key).get(route_color, None) == None:
            unfinished_routes_yellow.append(0)
        else:
            unfinished_routes_yellow.append(unfinished_routes.get(key).get(route_color))
        route_color = 'Green'
        if unfinished_routes.get(key).get(route_color, None) == None:
            unfinished_routes_green.append(0)
        else:
            unfinished_routes_green.append(unfinished_routes.get(key).get(route_color))
    for key in sorted(finished_routes):
        if not key in route_dates:
            route_dates.append(key)
        route_color = 'Gray'
        if finished_routes.get(key).get(route_color, None) == None:
            finished_routes_gray.append(0)
        else:
            finished_routes_gray.append(finished_routes.get(key).get(route_color))
        route_color = 'Yellow'
        if finished_routes.get(key).get(route_color, None) == None:
            finished_routes_yellow.append(0)
        else:
            finished_routes_yellow.append(finished_routes.get(key).get(route_color))
        route_color = 'Green'
        if finished_routes.get(key).get(route_color, None) == None:
            finished_routes_green.append(0)
        else:
            finished_routes_green.append(finished_routes.get(key).get(route_color))

convert_route_completion_summary()

route_dates_array = np.array(route_dates)
finished_gray = np.array(finished_routes_gray)
finished_yellow = np.array(finished_routes_yellow)
finished_green = np.array(finished_routes_green)

print("route dates array: " + str(route_dates_array))
print("finished_gray data: " + str(finished_gray))
print("finished_yellow data: " + str(finished_yellow))
print("finished_green data: " + str(finished_green))
print("original finished_gray data" + str(finished_routes_gray))

# plotting 

master_plot, (summary_bubble_plot, cost_analysis) = pyp.subplots(2, sharey=True)

pyp.xlim(min(route_dates_array) - 1, max(route_dates_array) + 1)
pyp.ylim(0, 4)

summary_bubble_plot.scatter(route_dates_array + 0.5, route_dates_array - route_dates_array + 0.5, c='gray', marker='o')
summary_bubble_plot.plot(route_dates_array + 0.5, route_dates_array - route_dates_array + 1.5, c='yellow', marker='o')
summary_bubble_plot.plot(route_dates_array + 0.5, route_dates_array - route_dates_array + 2.5, c='green', marker='o')

pyp.savefig('climbing-plot.png', bbox_inches='tight')
pyp.show()
