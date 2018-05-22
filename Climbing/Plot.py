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

def convert_boolean_arrays(pre_array):
    inter_array = []
    for val in pre_array:
        if val != 'nan':
            inter_array.append(conversion_constants.get(val))
        else:
            inter_array.append(False)
    return inter_array

completion = np.array(convert_boolean_arrays(pre_completion))
injury = np.array(convert_boolean_arrays(pre_injury))
new_route = np.array(convert_boolean_arrays(pre_new_route))
overhang = np.array(convert_boolean_arrays(pre_overhang))
handicap = np.array(convert_boolean_arrays(pre_handicap))

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
pyp.rcParams["figure.figsize"] = [10,20]
calendar_plot = pyp.subplot2grid((6, 3), (0, 0), colspan=2)
weight_profile = pyp.subplot2grid((6, 3), (0, 2))

summary_bubble_plot = pyp.subplot2grid((6, 3), (1, 0), colspan=3)
relative_frequency_plot = pyp.subplot2grid((6, 3), (2, 0), colspan=3)
legend_plot_route_diff = pyp.subplot2grid((6, 3), (5, 0))
legend_plot_weight_data = pyp.subplot2grid((6, 3), (5, 1))
legend_plot_other = pyp.subplot2grid((6, 3), (5, 2))

# set up list for legend items 

# plot calendar items
# TODO: add enumerated axis labels
calendar_plot.set_xlim(min(route_dates_week) - 1, max(route_dates_week) + 1)
calendar_plot.set_ylim(0, 8)
calendar_plot.set(xlabel='Week No.', ylabel='Weekday (Mon-Sun)')
calendar_plot.axis('equal')
#TODO: get marker size dynamically based on scaled units
calendar_plot.scatter(route_dates_week, route_dates_weekday, marker='s', c='blue')

# plot weight profile
weight_handles = []
date, weight, body_fat, water, bone_mass, muscle_mass = np.loadtxt('./weigh-ins.csv', unpack=True, skiprows=1, delimiter=',', converters={0: datefunc})
weight_profile.set_xlim(np.amin(date) - 1, np.amax(date) + 1)
weight_profile.set_ylim(0, 100)
bf = weight_profile.plot(date, body_fat, color='orange', ls='--', label='Body Fat Content (%)')
h20 = weight_profile.plot(date, water, color='blue', ls='--', label='Water Content (%)')
bone = weight_profile.plot(date, bone_mass, color='gray', ls='--', label='Bone Mass (%)')
musc = weight_profile.plot(date, muscle_mass, color='green', ls='--', label='Muscle Mass (%)')
weight_profile.set(xlabel='Date', ylabel='Body Composition (%)')
weight_profile_weight = weight_profile.twinx()
weight_profile_weight.set_ylim(0, np.amax(weight))
weight = weight_profile_weight.plot(date, weight, color='red', ls='-', marker='o', label='Body Weight (lbs)')
weight_profile_weight.set(ylabel='Weight (lbs)')
weight_handles.append(bf[0])
weight_handles.append(h20[0])
weight_handles.append(bone[0])
weight_handles.append(weight[0])
weight_handles.append(musc[0])

# plot bubble chart

bubble_handles = []

summary_bubble_plot.set_xlim(min(route_dates_ordered_array) - 1, max(route_dates_ordered_array) + 1)
summary_bubble_plot.set_ylim(0, 8)
summary_bubble_plot.set_title('Route Completion vs Failure by Difficulty')
summary_bubble_plot.set(xlabel='Session No.', ylabel='Route Difficulty')

# plot failed routes for bubble chart
fails = summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 0.5, c='gray', marker='s', s=failed_gray*15, alpha=0.5, label="Uncompleted Attempts")
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 1.5, c='yellow', marker='s', s=failed_yellow*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 2.5, c='green', marker='s', s=failed_green*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 3.5, c='red', marker='s', s=failed_red*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 4.5, c='blue', marker='s', s=failed_blue*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 5.5, c='orange', marker='s', s=failed_orange*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 6.5, c='purple', marker='s', s=failed_purple*15, alpha=0.5)
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 7.5, c='black', marker='s', s=failed_black*15, alpha=0.5)


# plot finished routes for bubble chart
fins = summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 0.5, c='gray', marker='o', s=finished_gray*15, edgecolors='black', label="Completed Attempts")
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 1.5, c='yellow', marker='o', s=finished_yellow*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 2.5, c='green', marker='o', s=finished_green*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 3.5, c='red', marker='o', s=finished_red*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 4.5, c='blue', marker='o', s=finished_blue*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 5.5, c='orange', marker='o', s=finished_orange*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 6.5, c='purple', marker='o', s=finished_purple*15, edgecolors='black')
summary_bubble_plot.scatter(route_dates_ordered_array, route_dates_array - route_dates_array + 7.5, c='black', marker='o', s=finished_black*15, edgecolors='black')

bubble_handles.append(fails)
bubble_handles.append(fins)

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

fin_perc = relative_frequency_plot.bar(route_dates_ordered, total_finished_ratio, 0.8, color='green', label='Finished')
fail_perc = relative_frequency_plot.bar(route_dates_ordered, total_unfinished_ratio, 0.8, bottom=total_finished_ratio, color='orange', label='Failed')
relative_frequency_plot.set_title('Percentage Success vs Failure for Bouldering Attempts by Day')
other_handles = []
other_handles.append(fin_perc)
other_handles.append(fail_perc)

# plot all legends in single row

def get_labels_for_handles(handle_arr):
    label_arr = []
    for i in handle_arr:
        label_arr.append(i.get_label())
    return label_arr

legend_plot_route_diff.set_title('Route Difficulty Legend')
bubble_labels = get_labels_for_handles(bubble_handles)
legend_plot_route_diff.legend(bubble_handles, bubble_labels)
legend_plot_route_diff.axis('off')

legend_plot_weight_data.set_title('Weight Graph Legend')
print('Showing weight_handles data: ' + str(weight_handles))
weight_labels = get_labels_for_handles(weight_handles)
legend_plot_weight_data.legend(weight_handles, weight_labels)
legend_plot_weight_data.axis('off')

legend_plot_other.set_title('Other Icons Legend')
other_labels = get_labels_for_handles(other_handles)
legend_plot_other.legend(other_handles, other_labels)
legend_plot_other.axis('off')

#finishing touches

pyp.tight_layout()
pyp.savefig('climbing-plot-%s.png' % str(datetime.datetime.today()), bbox_inches='tight')
# pyp.show()
