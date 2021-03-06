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

# LOADTXT
activity, route_level, features, injury_type = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(1,2,5,7), dtype='str')
completion, injury, new_route, overhang, handicap = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(4,6,8,9,10), dtype='bool', converters={4: boolfunc, 6: boolfunc, 8: boolfunc, 9: boolfunc, 10: boolfunc})
dates, quantity_time = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(0, 3), converters= {0: datefunc})

# Function to find total completed/uncompleted per route color for  each distinct day
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

# get conversion dictionary to convert matplotlib dates to an ordered set of dates (1, 2, 3, ... N)
def get_ordered_dates_dict():
    date_list = dates.tolist()
    new_dict = {}
    counter = 1
    for i in sorted(date_list):
        if not i in new_dict:
            new_dict.update({i: counter})

            counter += 1
    return new_dict

ordered_dates_dict = get_ordered_dates_dict()

# set up "calendar" representation of route dates (a la github heatmap)
route_dates_weekday = []
route_dates_week = []
route_dates_color_grade = []
route_dates, route_dates_time = np.loadtxt('./session-times.csv', usecols=(0,6), converters={0: datefunc}, skiprows=1, unpack=True, delimiter=',')
for i in range(len(route_dates)):
    route_dates_weekday.append(mpldates.num2date(route_dates[i]).isoweekday())
    route_dates_week.append(math.floor((route_dates[i] - min(route_dates)) / 7) + 1)
    route_dates_color_grade.append(route_dates_time[i] / np.amax(route_dates_time))
print("Weekday series: " + str(route_dates_weekday))
print("Week numbers: " + str(route_dates_week))
print("Session times: " + str(route_dates_time))
print("Session times normalized: " + str(route_dates_color_grade))

bubble_icon_size_multiplier = 15

def get_bubble_arrays(working_dict):
    bubble_dates = []
    bubble_color = []
    bubble_count = []
    bubble_diff_no = []
    for date in working_dict:
        for color in route_difficulty:
            if color in working_dict[date]:
                bubble_dates.append(ordered_dates_dict.get(date))
                bubble_color.append(color.lower())
                bubble_diff_no.append(route_difficulty.get(color)[0] + 0.5)
                bubble_count.append(bubble_icon_size_multiplier * working_dict[date][color])
    return (bubble_dates, bubble_color, bubble_count, bubble_diff_no)

finished_bubbles = get_bubble_arrays(finished_routes)
unfinished_bubbles = get_bubble_arrays(unfinished_routes)

# plotting 

#subplot setup
pyp.rcParams["figure.figsize"] = [10,20]
fig = pyp.figure()
calendar_plot = pyp.subplot2grid((6, 3), (0, 0), colspan=2)
weight_profile = pyp.subplot2grid((6, 3), (0, 2))

summary_bubble_plot = pyp.subplot2grid((6, 3), (1, 0), colspan=3)
relative_frequency_plot = pyp.subplot2grid((6, 3), (2, 0), colspan=3)

overhang_plot = pyp.subplot2grid((6, 3), (3, 0), colspan=2)
route_feature_plot = pyp.subplot2grid((6, 3), (3, 2), rowspan=2)
time_util_plot = pyp.subplot2grid((6, 3), (4, 0))

legend_plot_route_diff = pyp.subplot2grid((6, 3), (5, 0))
legend_plot_weight_data = pyp.subplot2grid((6, 3), (5, 1))
legend_plot_other = pyp.subplot2grid((6, 3), (5, 2))

# set up list for legend items 

# plot calendar items
for x in range(1, max(route_dates_week) + 1):
    for y in range(1, 8):
        calendar_plot.add_patch(mpl.patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color='gray', alpha=0.2))
cmap = mpl.cm.get_cmap('Greens')
scale_map = pyp.cm.ScalarMappable(cmap=cmap)
scale_map.set_array([])
for i in range(len(route_dates_week)):
    calendar_plot.add_patch(mpl.patches.Rectangle((route_dates_week[i] - 0.5, route_dates_weekday[i] - 0.5), 1, 1, facecolor=cmap(route_dates_color_grade[i])))
calendar_plot.set_xlim(min(route_dates_week) - 0.5, max(route_dates_week) + 0.5)
calendar_plot.set_ylim(0.5, 7.5)
calendar_plot.set_yticklabels(['', 'M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'])
calendar_plot.set(xlabel='Week No.', ylabel='Weekday (Mon-Sun)')
calendar_plot.axis('equal')
calendar_plot.set_title('Session Calendar')
cal_plot_colorbar = pyp.colorbar(scale_map, ax=calendar_plot, orientation='vertical', ticks=[0., 1.])
cal_plot_colorbar.ax.set_yticklabels(['0 hrs', '%2.2f hrs' % np.amax(route_dates_time)])
cal_plot_colorbar.set_label('Session Length', labelpad=-25, rotation=270)

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
weight_profile.set_title('Body Composition')
weight_profile_weight = weight_profile.twinx()
weight_profile_weight.set_ylim(0, 175)
weight = weight_profile_weight.plot(date, weight, color='red', ls='-', marker='o', label='Body Weight (lbs)')
weight_profile_weight.set(ylabel='Weight (lbs)')
weight_handles += [weight[0], bf[0], h20[0], bone[0], musc[0]]

# plot bubble chart

summary_bubble_plot.set_xlim(min(ordered_dates_dict.values()) - 1, max(ordered_dates_dict.values()) + 1)
summary_bubble_plot.set_ylim(0, 8)
summary_bubble_plot.set_title('Route Completion vs Failure by Difficulty')
summary_bubble_plot.set(xlabel='Session No.', ylabel='Route Difficulty')
summary_bubble_plot.scatter(unfinished_bubbles[0], unfinished_bubbles[3], c=unfinished_bubbles[1], s=unfinished_bubbles[2], alpha=0.5, marker='s')
summary_bubble_plot.scatter(finished_bubbles[0], finished_bubbles[3], c=finished_bubbles[1], s=finished_bubbles[2], marker='o', edgecolors='black')

# plot relative frequencies of success/fail per day
other_handles = []
total_finished_ratio = []
total_unfinished_ratio = []
total_ratio_dates = []
for date in ordered_dates_dict:
    total_ratio_dates.append(ordered_dates_dict.get(date))
    fin_count = 0
    unfin_count = 0
    for color in finished_routes.get(date):
        fin_count += finished_routes[date][color]
    for color in unfinished_routes.get(date):
        unfin_count += unfinished_routes[date][color]
    total_finished_ratio.append(100 * (fin_count / (fin_count + unfin_count)))
    total_unfinished_ratio.append(100 * (unfin_count / (unfin_count + fin_count)))

fin_perc = relative_frequency_plot.bar(total_ratio_dates, total_finished_ratio, 0.8, color='green', label='Finished')
fail_perc = relative_frequency_plot.bar(total_ratio_dates, total_unfinished_ratio, 0.8, bottom=total_finished_ratio, color='orange', label='Failed')
relative_frequency_plot.set_title('Percentage Success vs Failure for Bouldering Attempts by Day')
other_handles += [fin_perc, fail_perc]

# overhang plot
overhang_dates = []
overhang_fin_count = []
overhang_fail_count = []
overhang_summary = {}
for i in range(len(dates)): 
    if overhang_summary.get(ordered_dates_dict.get(dates[i]), None) == None:
        overhang_summary.update({ordered_dates_dict.get(dates[i]): [0., 0.]})
    if completion[i] and overhang[i]:
        currval = overhang_summary.get(ordered_dates_dict.get(dates[i]))[0]
        overhang_summary.get(ordered_dates_dict.get(dates[i]))[0] = currval + quantity_time[i]
    elif overhang[i]:
        currval = overhang_summary.get(ordered_dates_dict.get(dates[i]))[1]
        overhang_summary.get(ordered_dates_dict.get(dates[i]))[1] = currval + quantity_time[i]
for key in overhang_summary.keys():
    overhang_dates.append(key)
    overhang_fin_count.append(overhang_summary.get(key)[0])
    overhang_fail_count.append(overhang_summary.get(key)[1])
print('overhang route date array: ' + str(overhang_dates) + '; overhang route success count: ' + str(overhang_fin_count) + '; overhang failure count: ' + str(overhang_fail_count))
overhang_plot.plot(overhang_dates, [(100 * (overhang_fin_count[i] / (overhang_fin_count[i] + overhang_fail_count[i]))) for i in range(len(overhang_dates))], marker='s', mfc='b', label='Overhanging Route Success Rate')
overhang_plot.set_title('Overhanging Route Completion by Day')
overhang_plot.set(xlabel='Session No.', ylabel='Completion Rate (%)')
overhang_plot.set_ylim(0, 100)

# route feature bar chart
route_feature_dict = {'overhang': [0, 0], 'volume': [0, 0], 'chip': [0, 0], 'corner': [0, 0], 'new route': [0, 0], 'handicap': [0, 0]}

# helper method to track success and attempt counts per feature
def add_feature(key, i):
    if completion[i]:
        route_feature_dict.get(key)[0] += quantity_time[i]
    route_feature_dict.get(key)[1] += quantity_time[i]

for i in range(len(dates)):
    if activity[i] == 'Bouldering':
        if overhang[i]:
            add_feature('overhang', i)
        if new_route[i]:
            add_feature('new route', i)
        if handicap[i]:
            add_feature('handicap', i)
        if 'volume' in features[i]:
            add_feature('volume', i)
        if 'chip' in features[i]:
            add_feature('chip', i)
        if 'corner' in features[i]:
            add_feature('corner', i)
route_feature_percent_dict = {}
for key in route_feature_dict.keys():
    route_feature_percent_dict.update({(100 * float(route_feature_dict.get(key)[0]) / (float(route_feature_dict.get(key)[1]))): key})

route_feature_plot.set_title('Completion rate by route feature')
route_feature_plot.set(xlabel='Success Rate (%)', ylabel='Feature Type')
route_feature_plot.set_xlim(0, 100)
route_feature_plot.xaxis.grid(True)
route_feature_plot.set_axisbelow(True)
ypos = np.arange(len(route_feature_dict.keys()))
route_feature_plot.set_yticks(ypos)
route_feature_plot.set_yticklabels([route_feature_percent_dict[key] for key in sorted(route_feature_percent_dict.keys())])
print(str(route_feature_dict))
bars = route_feature_plot.barh(ypos, sorted(route_feature_percent_dict.keys()), color='purple')
# need to add value labels for each bar
for bar in bars:
    pass


# time utilization pie chart
dates, climbing_time, yoga_class, yoga, boulder_class, weights, total_time = np.loadtxt('./session-times.csv', unpack=True, skiprows=1, converters={0: datefunc}, delimiter=',')
pie_chart_dict = {'Bouldering': sum(climbing_time), 'Yoga': sum(yoga),'Yoga Class': sum(yoga_class), 'Bouldering Class': sum(boulder_class), 'Weight Room': sum(weights)}
time_util_plot.pie(pie_chart_dict.values(), labels=pie_chart_dict.keys(), shadow=True, autopct='%1.1f%%')
time_util_plot.axis('equal')
time_util_plot.set_title('Time Utilization')

# plot all legends in single row

def get_labels_for_handles(handle_arr):
    label_arr = []
    for i in handle_arr:
        label_arr.append(i.get_label())
    return label_arr

legend_plot_route_diff.set_title('Route Difficulty Legend')
legend_plot_route_diff.set_ylim(0, 6)
legend_plot_route_diff.set_xlim(-1, 11)
legend_plot_route_diff.axis('off')
diffx = np.linspace(0, 10, 8).tolist()
legend_plot_route_diff.text(5, 5.5, 'Route Rating:', horizontalalignment='center', verticalalignment='center')
legend_plot_route_diff.scatter(diffx, [5 for x in diffx], c=[key.lower() for key in route_difficulty.keys()], s=[5 * bubble_icon_size_multiplier for x in diffx], marker='o', edgecolors='black')
for x in range(len(diffx)):
    legend_plot_route_diff.text(diffx[x], 4.5, route_difficulty.get(list(route_difficulty.keys())[x])[1], horizontalalignment='center')
legend_plot_route_diff.scatter(1, 3.5, s=100, c='green', marker='o', edgecolor='black')
legend_plot_route_diff.text(2, 3.5, 'Completed', verticalalignment='center')
legend_plot_route_diff.scatter(7, 3.5, s=100, c='green', marker='s', alpha=0.5)
legend_plot_route_diff.text(8, 3.5, 'Failed', verticalalignment='center')
legend_plot_route_diff.text(5, 2.2, 'Route Quantity Scale', horizontalalignment='center')
sample_sizes = [1, 3, 5, 7, 10, 13, 16, 20]
legend_plot_route_diff.scatter(diffx, [1.5 for x in diffx], c='gray', s=[x * bubble_icon_size_multiplier for x in sample_sizes], marker='o', edgecolors='black')
legend_plot_route_diff.scatter(diffx, [1.0 for x in diffx], c='gray', s=[x * bubble_icon_size_multiplier for x in sample_sizes], marker='s', alpha=0.5)
for x in range(len(diffx)):
    legend_plot_route_diff.text(diffx[x], 0, str(sample_sizes[x]), horizontalalignment='center')


legend_plot_weight_data.set_title('Weight Graph Legend')
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
