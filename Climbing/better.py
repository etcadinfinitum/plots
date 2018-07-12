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
        
        # GET DATA, REFORMAT AS NEEDED
        
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
        
        self.bubble_finish = self.get_bubble_arrays(True, dates, completion, activity, quantity_time, route_level)
        self.bubble_failed = self.get_bubble_arrays(False, dates, completion, activity, quantity_time, route_level)
    
    def _get_difficulty_index(self, route_color):
        try:
            return self.route_difficulty[route_color][0]
        except KeyError:
            return None
    
    def get_bubble_arrays(self, did_you_finish, dates, completion, activity, quantity_time, route_level):
        '''
        unique_dates_list = unique_dates.tolist()
        dates_to_indices = {date: idx for idx, date in enumerate(unique_dates_list)}
        data = np.zeros([len(unique_dates_list), len(self.route_difficulty)])
        for date, level, successes in zip(dates, route_level, quantity_time):
            difficulty_index = _get_difficulty_index(level)
            if difficulty_index is None:
                continue
            data[dates_to_indices[date]][difficulty_index] += successes
        return data
        ''' 
        unique_dates_list = self.unique_dates.tolist()
        data = np.zeros([len(unique_dates_list), len(self.route_difficulty)])
        # data = []
        #        for i in range(len(unique_dates)):
        #    data.append([0 for x in range(len(self.route_difficulty))])
        for i in range(len(dates)):
            if activity[i] == 'Bouldering' and completion[i] == did_you_finish:
                count = data[unique_dates_list.index(dates[i])][self.route_difficulty[route_level[i]][0]]
                data[unique_dates_list.index(dates[i])][self.route_difficulty[route_level[i]][0]] = count + quantity_time[i]
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
    
    summary_bubble_plot.set_xlim(0, len(data.unique_dates))
    summary_bubble_plot.set_ylim(0, len(data.route_difficulty))
    summary_bubble_plot.set_title('Route Completion vs Failure by Difficulty')
    summary_bubble_plot.set(xlabel='Session No.', ylabel='Route Difficulty')
    #    summary_bubble_plot.scatter(unfinished_bubbles[0], unfinished_bubbles[3], c=unfinished_bubbles[1], s=unfinished_bubbles[2], alpha=0.5, marker='s')
    for i in range(len(data.route_difficulty)):
        pass
    # summary_bubble_plot.scatter(finished_bubbles[0], finished_bubbles[3], c=finished_bubbles[1], s=finished_bubbles[2], marker='o', edgecolors='black')
    
    
make_plots()
