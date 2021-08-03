import _import_helper  # noqa

from Source import Truss_Calculator as tc

'''
It is reccommended to use the factory functions to build trusses (Example 1).
However it is also possible to directly create the objects using classes (Example 2).
'''

#############################################################
###  Example 1: Building a truss using factory functions  ###
#############################################################

# set some initial parameters
default_bar_params = {"b": 16, "t": 1.1, "D": 5, "E": 210, "strength_max": 0.216}
units_system = 'kN, mm'

# initialise a truss (and set it as the active truss)
tc.init_truss('My First Truss', default_bar_params, units_system)

# create the joints: name, x, y
tc.create_joint('A', 0, 0)
tc.create_joint('B', 0, 100)
tc.create_joint('C', 100, 0)
tc.create_joint('D', 100, 100)
tc.create_joint('E', 200, 0)
tc.create_joint('F', 200, 100)

# create the bars: name, *between_which_joints_names
tc.create_bar('AB', 'A', 'B')
tc.create_bar('AC', 'A', 'C')
tc.create_bar('AD', 'A', 'D')
tc.create_bar('BD', 'B', 'D')
tc.create_bar('CD', 'C', 'D')
tc.create_bar('CE', 'C', 'E')
tc.create_bar('DE', 'D', 'E')
tc.create_bar('DF', 'D', 'F')
tc.create_bar('EF', 'E', 'F')

# apply loads: name, joint, x, y
tc.create_load('W', 'D', 0, -100)

# put supports at joints: name, joint, type
tc.create_support('A', 'A', 'pin')
tc.create_support('E', 'E', 'roller', roller_normal=(0, 1))

# make calculations and get results of analysis
my_results = tc.active_truss.Result(tc.active_truss, sig_figs=3, solution_method=tc.SolveMethod.SCIPY)

# show the results in text form
print(my_results)

# show the results in graphical form
tc.plot_diagram(tc.active_truss, my_results, show_reactions=True)


###################################################
###  Example 2: Building a truss using objects  ###
###################################################

# initialise a truss (and set it as the active truss)
my_truss = tc.Truss('My Second Truss', default_bar_params, units_system)

# create the joints: truss object, name, x, y
joint_a = my_truss.Joint(my_truss, 'A', 0, 0)
joint_b = my_truss.Joint(my_truss, 'B', 0, 100)
joint_c = my_truss.Joint(my_truss, 'C', 100, 0)
joint_d = my_truss.Joint(my_truss, 'D', 100, 100)
joint_e = my_truss.Joint(my_truss, 'E', 200, 0)
joint_f = my_truss.Joint(my_truss, 'F', 200, 100)

# create the bars: truss, name, *between_which_joints_objects
bar_ab = my_truss.Bar(my_truss, 'AB', joint_a, joint_b)
bar_ac = my_truss.Bar(my_truss, 'AC', joint_a, joint_c)
bar_ad = my_truss.Bar(my_truss, 'AD', joint_a, joint_d)
bar_bd = my_truss.Bar(my_truss, 'BD', joint_b, joint_d)
bar_cd = my_truss.Bar(my_truss, 'CD', joint_c, joint_d)
bar_ce = my_truss.Bar(my_truss, 'CE', joint_c, joint_e)
bar_de = my_truss.Bar(my_truss, 'DE', joint_d, joint_e)
bar_df = my_truss.Bar(my_truss, 'DF', joint_d, joint_f)
bar_ef = my_truss.Bar(my_truss, 'EF', joint_e, joint_f)

# apply loads: name, joint object, x, y
load_w = my_truss.Load('W', joint_d, 0, -100)

# put supports at joints: name, joint, type
support_a = my_truss.Support('A', joint_a, 'pin')
support_e = my_truss.Support('E', joint_e, 'roller', (0, 1))

# make calculations and get results of analysis
my_results = my_truss.Result(my_truss, sig_figs=3, solution_method=tc.SolveMethod.SCIPY)

# show the results in text form
print(my_results)

# show the results in graphical form
tc.plot_diagram(my_truss, my_results, show_reactions=True)
