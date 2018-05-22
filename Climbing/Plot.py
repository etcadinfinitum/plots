import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pyp
import matplotlib.dates as mpldates
import datetime
import math

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

# GET DATA, REFORMAT AS NEEDED

datefunc = lambda x: mpldates.date2num(datetime.date(int(x[6:]), int(x[0:2]), int(x[3:5])))

activity, route_level, features, injury_type = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(1,2,5,7), dtype='str')

pre_completion, pre_injury, pre_new_route, pre_overhang, pre_handicap = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(4,6,8,9,10), dtype='str')

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

dates, quantity_time = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(0, 3), converters= {0: datefunc})

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

print("\nFinished routes (full dict): " + str(finished_routes))
print("Failed routes (full dict): " + str(unfinished_routes) + "\n")

route_dates = []
unfinished_routes_intermediate = {'Gray': [], 'Yellow': [], 'Green': [], 'Red': [], 'Blue': [], 'Orange': [], 'Purple': [], 'Black': [] }
finished_routes_intermediate = {'Gray': [], 'Yellow': [], 'Green': [], 'Red': [], 'Blue': [], 'Orange': [], 'Purple': [], 'Black': [] }


# prep dictionaries for concise plotting?
def convert_route_completion_summary():
    for key in sorted(unfinished_routes):
        if not key in route_dates:
            route_dates.append(key)
        for color in route_difficulty:
            if unfinished_routes.get(key).get(color, None) == None:
                unfinished_routes_intermediate.get(color).append(0)
            else:
                unfinished_routes_intermediate.get(color).append(unfinished_routes.get(key).get(color))
    for key in sorted(finished_routes):
        if not key in route_dates:
            route_dates.append(key)
        for color in route_difficulty:
            if finished_routes.get(key).get(color, None) == None:
                finished_routes_intermediate.get(color).append(0)
            else:
                finished_routes_intermediate.get(color).append(finished_routes.get(key).get(color))

convert_route_completion_summary()

# set up ordered (1, 2, 3, ... N) set of dates 
route_dates_array = np.array(route_dates)
route_dates_ordered = []
for i in route_dates:
    route_dates_ordered.append(route_dates.index(i) + 1)
route_dates_ordered_array = np.array(route_dates_ordered)

# set up "calendar" representation of route dates (a la github heatmap)
route_dates_weekday = []
route_dates_week = []
for i in route_dates:
    route_dates_weekday.append(mpldates.num2date(i).isoweekday())
    route_dates_week.append(((int(i) - int(route_dates[0])) / 7) + 1)
print("Weekday series: " + str(route_dates_weekday))
print("Week numbers: " + str(route_dates_week))

finished_gray = np.array(finished_routes_intermediate.get('Gray'))
finished_yellow = np.array(finished_routes_intermediate.get('Yellow'))
finished_green = np.array(finished_routes_intermediate.get('Green'))
finished_red = np.array(finished_routes_intermediate.get('Red'))
finished_blue = np.array(finished_routes_intermediate.get('Blue'))
finished_orange = np.array(finished_routes_intermediate.get('Orange'))
finished_purple = np.array(finished_routes_intermediate.get('Purple'))
finished_black = np.array(finished_routes_intermediate.get('Black'))
failed_gray = np.array(unfinished_routes_intermediate.get('Gray'))
failed_yellow = np.array(unfinished_routes_intermediate.get('Yellow'))
failed_green = np.array(unfinished_routes_intermediate.get('Green'))
failed_red = np.array(unfinished_routes_intermediate.get('Red'))
failed_blue = np.array(unfinished_routes_intermediate.get('Blue'))
failed_orange = np.array(unfinished_routes_intermediate.get('Orange'))
failed_purple = np.array(unfinished_routes_intermediate.get('Purple'))
failed_black = np.array(unfinished_routes_intermediate.get('Black'))

# helpful printouts (just to confirm data is being processed properly)
print("route dates array: " + str(route_dates_array))
print("finished_gray data: " + str(finished_gray))
print("finished_yellow data: " + str(finished_yellow))
print("finished_green data: " + str(finished_green))

# plotting 

#subplot setup
calendar_plot = pyp.subplot2grid((3, 6), (0, 0), colspan=2)
cost_benefit_plot = pyp.subplot2grid((3, 6), (0, 2))

summary_bubble_plot = pyp.subplot2grid((3, 6), (1, 0), colspan=3)
relative_frequency_plot = pyp.subplot2grid((3, 6), (2, 0), colspan=3)

# plot calendar items
# TODO: add enumerated axis labels
calendar_plot.set_xlim(min(route_dates_week) - 1, max(route_dates_week) + 1)
calendar_plot.set_ylim(0, 8)
calendar_plot.set(xlabel='Week No.', ylabel='Weekday (Mon-Sun)')
calendar_plot.axis('equal')
#TODO: get marker size dynamically based on scaled units
calendar_plot.scatter(route_dates_week, route_dates_weekday, marker='s', c='blue')



# plot bubble chart

summary_bubble_plot.set_xlim(min(route_dates_ordered_array) - 1, max(route_dates_ordered_array) + 1)
summary_bubble_plot.set_ylim(0, 8)
summary_bubble_plot.set_title('Route Completion vs Failure by Difficulty')
summary_bubble_plot.set(xlabel='Session No.', ylabel='Route Difficulty')

# plot failed routes for bubble chart
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 0.5, c='gray', marker='s', s=failed_gray*15, alpha=0.5, label="Uncompleted Attempts")
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 1.5, c='yellow', marker='s', s=failed_yellow*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 2.5, c='green', marker='s', s=failed_green*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 3.5, c='red', marker='s', s=failed_red*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 4.5, c='blue', marker='s', s=failed_blue*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 5.5, c='orange', marker='s', s=failed_orange*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 6.5, c='purple', marker='s', s=failed_purple*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 7.5, c='black', marker='s', s=failed_black*15, alpha=0.5)


# plot finished routes for bubble chart
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 0.5, c='gray', marker='o', s=finished_gray*15, edgecolors='black', label="Completed Attempts")
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 1.5, c='yellow', marker='o', s=finished_yellow*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 2.5, c='green', marker='o', s=finished_green*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 3.5, c='red', marker='o', s=finished_red*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 4.5, c='blue', marker='o', s=finished_blue*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 5.5, c='orange', marker='o', s=finished_orange*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 6.5, c='purple', marker='o', s=finished_purple*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 7.5, c='black', marker='o', s=finished_black*15, edgecolors='black')

summary_bubble_plot.legend()


# plot relative frequencies of success/fail per day
total_finished_per_day = []
total_unfinished_per_day = []
for i in route_dates_ordered:
    idx = route_dates_ordered.index(i)
    total_finished_per_day.append(finished_gray[idx] + finished_yellow[idx] + finished_green[idx] + finished_red[idx] + finished_blue[idx] + finished_orange[idx] + finished_purple[idx] + finished_black[idx])
    total_unfinished_per_day.append(failed_gray[idx] + failed_yellow[idx] + failed_green[idx] + failed_red[idx] + failed_blue[idx] + failed_orange[idx] + failed_purple[idx] + failed_black[idx])

total_finished_ratio = []
total_unfinished_ratio = []
for i in route_dates_ordered:
    idx = route_dates_ordered.index(i)
    total = float(total_finished_per_day[idx] + total_unfinished_per_day[idx])
    total_finished_ratio.append(100 * float(total_finished_per_day[idx]) / total)
    total_unfinished_ratio.append(100 * float(total_unfinished_per_day[idx]) / total)

relative_frequency_plot.bar(route_dates_ordered, total_finished_ratio, 0.8, color='green', label='Finished')
relative_frequency_plot.bar(route_dates_ordered, total_unfinished_ratio, 0.8, bottom=total_finished_ratio, color='orange', label='Failed')
relative_frequency_plot.set_title('Percentage Success vs Failure for Bouldering Attempts by Day')
relative_frequency_plot.legend()

#finishing touches

pyp.tight_layout()
pyp.savefig('climbing-plot-%s.png' % str(datetime.datetime.today()), bbox_inches='tight')
# pyp.show()
