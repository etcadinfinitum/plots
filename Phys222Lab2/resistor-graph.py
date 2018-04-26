import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyp

'''
graph setup below
'''

# import data tables from .csv files
resistor1_ps_volt, resistor1_ps_current, resistor1_mm_volt, resistor1_mm_volt_uct, resistor1_mm_current, resistor1_mm_current_uct, resistor1_volt_delta = np.loadtxt("./resistor-1-data.csv", delimiter=",", skiprows=3, unpack=True)
resistor2_ps_volt, resistor2_ps_current, resistor2_mm_volt, resistor2_mm_volt_uct, resistor2_mm_current, resistor2_mm_current_uct, resistor2_volt_delta = np.loadtxt("./resistor-2-data.csv", delimiter=",", skiprows=3, unpack=True)

# set up x-values for best fit line plotting
r1_sample_points = np.linspace(0, 35, 50)
r2_sample_points = np.linspace(0, 45, 50)

new_graph = pyp.figure()

# set up grid axes
pyp.rc('axes', axisbelow=True)
pyp.grid(b=True, which='major', color='grey', linestyle='-')
pyp.grid(b=True, which='minor', color='grey', linestyle=':')
pyp.minorticks_on()

'''
plotting commands below
'''

# data plotting here

# get polyfit coefficients for best fit curves
resistor1_coefs = np.polyfit(resistor1_mm_current, resistor1_mm_volt, 1)
resistor2_coefs = np.polyfit(resistor2_mm_current, resistor2_mm_volt, 1)

# plot best fit curves - plotting before the measured data values because graph objects are laid in ascending order (last item on top)
pyp.plot(r1_sample_points, np.polyval(resistor1_coefs, r1_sample_points), 'r', label="Best-fit line for Resistor 1")
pyp.plot(r2_sample_points, np.polyval(resistor2_coefs, r2_sample_points), 'b', label="Best-fit line for Resistor 2")

# plot measured data for each resistor
pyp.errorbar(resistor1_mm_current, resistor1_mm_volt, xerr=resistor1_mm_current_uct, yerr=resistor1_mm_volt_uct, ecolor='g', mfc='r', ls='None', marker='o', label="Measured volt-milliampere values for Resistor 1", capsize=5, elinewidth=2, markeredgewidth=2)
pyp.errorbar(resistor2_mm_current, resistor2_mm_volt, xerr=resistor2_mm_current_uct, yerr=resistor2_mm_volt_uct, ecolor='g', mfc='b', ls='None', marker='o', label="Measured volt-milliampere values for Resistor 2", capsize=5, elinewidth=2, markeredgewidth=2)

# label axes
pyp.ylabel('Resistor Voltage (V)')
pyp.xlabel('Circuit Current (mA)')

# add an automatically generated legend for each plot type 
leg = pyp.legend(bbox_to_anchor=(-0.07, -0.62), loc=3, borderaxespad=0.)

# graph title
pyp.title("Voltage vs Current for Two Different Resistors")

pyp.savefig("resistor-activity-1-graph.png", bbox_inches='tight')
pyp.show()
