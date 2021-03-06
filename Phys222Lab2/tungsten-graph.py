import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as pyp

'''
graph setup below
'''

# import data tables from .csv files
ps_volt, ps_current, mm_volt, mm_volt_uct, mm_current, mm_current_uct = np.loadtxt("./resistor-tungsten-data.csv", delimiter=",", skiprows=2, unpack=True)

# create x value coords for best fit line/curve
sample_points = np.linspace(0,220,100)

new_graph = pyp.figure()

# set up grid axes
pyp.rc('axes', axisbelow=True)
pyp.grid(b=True, which='major', color='grey', linestyle='-')
pyp.grid(b=True, which='minor', color='grey', linestyle=':')
pyp.minorticks_on()

'''
plotting commands below
'''

# best fit line coefficients 
line_coefs = np.polyfit(mm_current, mm_volt, 1)

# best fit quadratic curve coefficients
quad_coefs = np.polyfit(mm_current, mm_volt, 2)

# plot line/curve fits 
pyp.plot(sample_points, np.polyval(line_coefs, sample_points), 'y', label="Linear best-fit line")
pyp.plot(sample_points, np.polyval(quad_coefs, sample_points), 'm', label="Quadratic best-fit curve")
# print out coefficient values formatted as line/curve equation
print("Line equation is: y = %f * x + %f" % (line_coefs[0], line_coefs[1]))
print("Quadratic equation is: y = %f * x^2 + %f * x + %f" % (quad_coefs[0], quad_coefs[1], quad_coefs[2]))

# plot measured data for each resistor
pyp.errorbar(mm_current, mm_volt, xerr=mm_current_uct, yerr=mm_volt_uct, ecolor='g', mfc='b', ls='None', marker='o', label="Measured Volt-Milliampere Values for Tungsten Lamp", capsize=5, elinewidth=2, markeredgewidth=2)

# label axes
pyp.ylabel('Resistor Voltage (V)')
pyp.xlabel('Circuit Current (mA)')

# add an automatically generated legend for each plot type 
leg = pyp.legend(bbox_to_anchor=(-0.07, -0.3), loc=3, borderaxespad=0.)

# graph title
pyp.title("Voltage vs Current for Tungsten Lamp")

pyp.savefig("tungsten-graph-activity-2.png", bbox_inches='tight')
pyp.show()
