import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyp
from matplotlib.axes import Axes

'''
graph setup below
'''

# import data tables from .csv files
mm_volt, mm_volt_uct, mm_current, mm_current_uct = np.loadtxt("./resistor-diode-data.csv", delimiter=",", skiprows=3, unpack=True)


new_graph = pyp.figure()

# set up grid axes
pyp.rc('axes', axisbelow=True)
pyp.grid(b=True, which='major', color='grey', linestyle='-')
pyp.grid(b=True, which='minor', color='grey', linestyle=':')
pyp.minorticks_on()

pyp.xlim(-5,400)

'''
plotting commands below
'''

# data plotting here

# plot measured data for each resistor
pyp.errorbar(mm_current, mm_volt, xerr=mm_current_uct, yerr=mm_volt_uct, ecolor='g', mfc='b', ls='None', marker='o', label="Measured Volt-Milliampere Values for Diode with Forward-Voltage Configuration", capsize=5, elinewidth=2, markeredgewidth=2)

# label axes
pyp.ylabel('Resistor Voltage (V)')
pyp.xlabel('Circuit Current (mA)')

# add an automatically generated legend for each plot type 
leg = pyp.legend(bbox_to_anchor=(-0.07, -0.62), loc=3, borderaxespad=0.)

# graph title
pyp.title("Voltage vs Current for Forward-Voltage Diode")

pyp.savefig("diode-graph-activity-3.png", bbox_inches='tight')
pyp.show()
