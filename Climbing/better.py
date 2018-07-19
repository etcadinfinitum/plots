import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pyp
import matplotlib.dates as mpldates
import datetime
import math
import pdb

# HELPER METHODS AND CONSTANTS

class Data:

    def __init__(self):

        self.conversion_constants = { "nan": None, "TRUE": True, "FALSE": False }
        self.route_difficulty = { "Gray" : [0, "V0"], "Yellow": [1, "V1"], "Green": [2, "V2"], "Red": [3, "V3"], "Blue": [4, "V4"], "Orange": [5, "V5"], "Purple": [6, "V6"], "Black": [7, "V7"] }
        
        # LAMBDAS
        datefunc = lambda x: mpldates.date2num(datetime.date(int(x[6:]), int(x[0:2]), int(x[3:5])))
        boolfunc = lambda val: True if self.conversion_constants.get(val.decode()) == True else False
        # boolfunc = lambda val: self.conversion_constants.get(val.decode())
        
        # LOADTXT
        activity, route_level, features, injury_type = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(1,2,5,7), dtype='str')
        completion, injury, new_route, overhang, handicap = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(4,6,8,9,10), dtype='bool', converters={4: boolfunc, 6: boolfunc, 8: boolfunc, 9: boolfunc, 10: boolfunc})
        dates, quantity_time = np.loadtxt("./routes.csv", delimiter=",", skiprows=1, unpack=True, usecols=(0, 3), converters= {0: datefunc})
        
        # get an np NDarray of unique dates
        self.unique_dates = np.unique(dates)
        
        self.bubble_icon_size_multiplier = 15
        
        # get bubble arrays
        self.bubble_finish = self.get_bubble_arrays(True, dates, completion, activity, quantity_time, route_level, overhang, False)
        self.bubble_failed = self.get_bubble_arrays(False, dates, completion, activity, quantity_time, route_level, overhang, False)
        
        # get overhang arrays
        self.overhang_finish = self.get_bubble_arrays(True, dates, completion, activity, quantity_time, route_level, overhang, True)
        self.overhang_failed = self.get_bubble_arrays(False, dates, completion, activity, quantity_time, route_level, overhang, True)
        
        self.session_dates, climbing_time, yoga, yoga_class, boulder_class, weights, self.session_times = np.loadtxt('session-times.csv', converters={0: datefunc}, skiprows=1, unpack=True, delimiter=',')
        self.pie_chart_dict = {'Bouldering': sum(climbing_time), 'Yoga': sum(yoga),'Yoga Class': sum(yoga_class), 'Bouldering Class': sum(boulder_class), 'Weight Room': sum(weights)}
        
        # this is the lambda function which converts the mpl date format to a given week number
        weekfunc = lambda date: math.floor((datefunc(date) - np.amin(self.unique_dates))/ 7) + 1
        
        self.weight_week, self.weight, self.fat_mass, self.muscle_mass = np.loadtxt('./weigh-ins.csv', delimiter=',', skiprows=1, unpack=True, usecols=(0, 1, 2, 5), converters={0: weekfunc})
        
        # dictionary of route fails/successes
        # for any given key, the list value will show [succ, fail]
        self.route_feature_dict = {'overhang': [0, 0], 'volume': [0, 0], 'chip': [0, 0], 'corner': [0, 0], 'new route': [0, 0], 'handicap': [0, 0]}
        for i in range(len(dates)):
            if activity[i] == 'Bouldering':
                if overhang[i]:
                    self.add_feature('overhang', completion[i], quantity_time[i])
                if new_route[i]:
                    self.add_feature('new route', completion[i], quantity_time[i])
                if handicap[i]:
                    self.add_feature('handicap', completion[i], quantity_time[i])
                if 'volume' in features[i]:
                    self.add_feature('volume', completion[i], quantity_time[i])
                if 'chip' in features[i]:
                    self.add_feature('chip', completion[i], quantity_time[i])
                if 'corner' in features[i]:
                    self.add_feature('corner', completion[i], quantity_time[i])
    

    # helper method to track success and attempt counts per feature
    def add_feature(self, key, completion_state, count):
        if completion_state:
            self.route_feature_dict.get(key)[0] += count
        else: 
            self.route_feature_dict.get(key)[1] += count
    
    def _get_difficulty_index(self, route_color):
        try:
            return self.route_difficulty[route_color][0]
        except KeyError:
            return None
    
    def get_bubble_arrays(self, did_you_finish, dates, completion, activity, quantity_time, route_level, overhang, overhang_flag):
        unique_dates_list = self.unique_dates.tolist()
        data = np.zeros([len(unique_dates_list), len(self.route_difficulty)])
        for i in range(len(dates)):
            if activity[i] != 'Bouldering':
                continue
            if overhang_flag and not overhang[i]:
                continue
            if completion[i] == did_you_finish:
                date_idx = unique_dates_list.index(dates[i])
                diff_idx = self.route_difficulty[route_level[i]][0]
                # get old value for cumulative count
                count = data[date_idx][diff_idx]
                data[date_idx][diff_idx] = count + (quantity_time[i] * self.bubble_icon_size_multiplier)
        return data
   
# the function which produces the graphs 
def make_plots():
    # generate the data to reference
    data = Data()

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
    
    summary_bubble_plot.set_xlim(0, len(data.unique_dates) + 1)
    summary_bubble_plot.set_ylim(0, len(data.route_difficulty))
    summary_bubble_plot.set_title('Route Completion vs Failure by Difficulty')
    summary_bubble_plot.set(xlabel='Session No.', ylabel='Route Difficulty')
    
    inv_route_difficulty = { value[0]: key for key, value in data.route_difficulty.items() } 
    for i in range(len(data.route_difficulty)):
        summary_bubble_plot.scatter(np.array([data.unique_dates.tolist().index(x) + 1 for x in data.unique_dates]), np.array([i + 0.5 for x in data.unique_dates]), s=data.bubble_finish[:,i], c=np.array([inv_route_difficulty[i].lower() for x in data.unique_dates]), marker='o', edgecolors='black')
        summary_bubble_plot.scatter(np.array([data.unique_dates.tolist().index(x) + 1 for x in data.unique_dates]), np.array([i + 0.5 for x in data.unique_dates]), s=data.bubble_failed[:,i], c=np.array([inv_route_difficulty[i].lower() for x in data.unique_dates]), marker='s', alpha=0.5)
    
    # calendar plot gray squares (baseline)
    for x in range(1, 13):
        for y in range(1, 8):
            calendar_plot.add_patch(mpl.patches.Rectangle((x - 0.5, y - 0.5), 1, 1, color='gray', alpha=0.2))
    # color map constants for calendar plot scale
    cmap = mpl.cm.get_cmap('Greens')
    scale_map = pyp.cm.ScalarMappable(cmap=cmap)
    scale_map.set_array([])
    scale_normalizer = mpl.colors.Normalize(vmin = 0., vmax = np.amax(data.session_times))
    # plot calendar squares
    for session in data.session_dates:
        week_num = math.floor((session - np.amin(data.session_dates)) / 7) + 1
        weekday = mpldates.num2date(session).isoweekday()
        index = data.session_dates.tolist().index(session)
        calendar_plot.add_patch(mpl.patches.Rectangle((week_num - 0.5, weekday - 0.5), 1, 1, facecolor=cmap(scale_normalizer(data.session_times[index]))))
    calendar_plot.set_xlim(0.5, 12.5)
    calendar_plot.set_ylim(0.5, 7.5)
    calendar_plot.set_yticklabels(['', 'M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'])
    calendar_plot.set(xlabel='Week No.', ylabel='Weekday (Mon-Sun)')
    calendar_plot.axis('equal')
    calendar_plot.set_title('Session Calendar')
    cal_plot_colorbar = pyp.colorbar(scale_map, ax=calendar_plot, orientation='vertical', ticks=[0., 1.])
    cal_plot_colorbar.ax.set_yticklabels(['0 hrs', '%2.2f hrs' % np.amax(data.session_times)])
    cal_plot_colorbar.set_label('Session Length', labelpad=-25, rotation=270)
    
    # plot weight data
    weight_profile.set_xlim(np.amin(data.weight_week) - 1, np.amax(data.weight_week) + 1)
    weight_profile.set_ylim(0, 50)
    bf = weight_profile.plot(data.weight_week, data.fat_mass, color='orange', ls='--', label='Body Fat Content (%)')
    musc = weight_profile.plot(data.weight_week, data.muscle_mass, color='red', ls='--', label='Muscle Mass (%)')
    weight_profile.set(xlabel='Week No.', ylabel='Body Composition (%)')
    weight_profile_weight = weight_profile.twinx()
    weight_profile_weight.set_ylim(0, 175)
    lbs = weight_profile_weight.plot(data.weight_week, data.weight, color='green', ls='-', marker='o', label='Body Weight (lbs')
    weight_profile_weight.set(ylabel='Weight (lbs)')
    weight_profile.legend(loc=3)
    weight_profile_weight.legend(loc=6)
    
    # success/fail bar graph
    success_vals = np.array([(100 * float(np.sum(data.bubble_finish[i,:])) / (np.sum(data.bubble_failed[i,:]) + np.sum(data.bubble_finish[i,:]))) for i in range(len(data.unique_dates))])
    fail_vals = np.array([(100 * float(np.sum(data.bubble_failed[i,:])) / (np.sum(data.bubble_failed[i,:]) + np.sum(data.bubble_finish[i,:]))) for i in range(len(data.unique_dates))])
    relative_frequency_plot.set_title('Percentage Success vs Failure for Bouldering Attempts by Day')
    fin_perc = relative_frequency_plot.bar(np.array([idx + 1 for idx in range(len(data.unique_dates))]), success_vals, 0.8, color='green', label='Finished')
    fail_perc = relative_frequency_plot.bar(np.array([idx + 1 for idx in range(len(data.unique_dates))]), fail_vals, 0.8, bottom=success_vals, color='gray', label='Failed')
    relative_frequency_plot.legend(loc=3)
    
    # time utilization pie chart
    time_util_plot.pie(data.pie_chart_dict.values(), labels=data.pie_chart_dict.keys(), shadow=True, autopct='%1.1f%%')
    time_util_plot.axis('equal')
    time_util_plot.set_title('Time Utilization')
    
    # route feature completion rates
    route_feature_plot.set_title('Completion rate by route feature')
    route_feature_plot.set(xlabel='Success Rate (%)', ylabel='Feature Type')
    route_feature_plot.set_xlim(0, 100)
    route_feature_plot.xaxis.grid(True)
    route_feature_plot.set_axisbelow(True)
    ypos = np.arange(len(data.route_feature_dict.keys()))
    route_feature_plot.set_yticks(ypos)
    inverted_dict = { 100 * float(val[0]) / (val[0] + val[1]): key for key, val in data.route_feature_dict.items() }
    route_feature_plot.set_yticklabels([inverted_dict[key] for key in sorted(inverted_dict.keys())])
    bars = route_feature_plot.barh(ypos, sorted(inverted_dict.keys()), color='purple')
    
    # overhang completion plot
    pdb.set_trace()
    overhang_plot.plot(np.array([idx + 1 for idx in range(len(data.unique_dates))]), np.array([ 100 * np.sum(data.overhang_finish[i,:]) / (np.sum(data.overhang_finish[i,:]) + np.sum(data.overhang_failed[i,:])) for i in range(len(data.unique_dates))]), marker='s', mfc='b', label='Overhanging Route Success Rate')
    overhang_plot.set_title('Overhanging Route Completion by Day')
    overhang_plot.set(xlabel='Session No.', ylabel='Completion Rate (%)')
    overhang_plot.set_ylim(0, 100)
    
    #finishing touches
    
    pyp.tight_layout()
    pyp.savefig('climbing-plot-%s.png' % str(datetime.datetime.today()), bbox_inches='tight')
    
make_plots()
