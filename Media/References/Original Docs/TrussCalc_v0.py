from matplotlib import pyplot as plt
import math, sigfig, warnings # module "sigfig" requires "pip install sigfig" at command line
import numpy as np

def get_all_trusses():
          return [truss for truss in Truss]
     
class IterTruss(type):
     def __iter__(cls):
          return iter(cls._allTrusses)
     
class Truss(metaclass = IterTruss):
     _allTrusses = []

     # TRUSS METACLASS INITIATORS

     class IterJoint(type):
          def __iter__(cls):
               return iter(cls._allJoints)
     class IterBar(type):
          def __iter__(cls):
               return iter(cls._allBars)

     class IterLoad(type):
          def __iter__(cls):
               return iter(cls._allLoads)

     class IterSupport(type):
          def __iter__(cls):
               return iter(cls._allSupports)

     def __init__(self, bar_params: dict = None, units = 'kN, mm'):
          self._allTrusses.append(self)
          
          if bar_params == None:
               if units == 'N, m':
                    self.default_params = {"b" : 0.016, "t" : 0.004, "D" : 0.020, "E" : 2.1e11}
               elif units == 'kN, mm':
                    self.default_params = {"b" : 1.6, "t" : 4, "D" : 20, "E" : 210}
               else:
                    raise ValueError('Units must be either "N, m" or "kN, mm".')
          else:
               self.default_params = bar_params
          self.units = units

     # PARTS OF THE TRUSS (INNER CLASSES)
     
     class Joint(metaclass = IterJoint):
          _allJoints = []
          
          def __init__(self, truss: object, name: str, x: float, y: float):
               self._allJoints.append(self)
               
               self.name = name
               self.truss = truss
               self.x = x
               self.y = y
               self.loads = {}

          def form_equation(self):
               self.truss.get_all_bars_connected_to_joint(self)

     class Bar(metaclass = IterBar):
          _allBars = []
          
          def __init__(self, truss: object, name: str, first_joint: object, second_joint: object,
                       my_params: dict = None):
               self._allBars.append(self)

               self.name = name
               self.first_joint, self.first_joint_name = first_joint, first_joint.name
               self.second_joint, self.second_joint_name = second_joint, second_joint.name
               if my_params == None:
                    self.params = truss.default_params
               else:
                    self.params = my_params
                    
               self.b, self.t, self.D, self.E, self.σ_max = self.params["b"], self.params["t"], self.params["D"], self.params["E"], self.params["σ_max"]

          def length(self):
               self.L = math.sqrt((self.first_joint.x - self.second_joint.x)**2 + (self.first_joint.y - self.second_joint.y)**2)
               return self.L

          def area(self):
               self.A = (self.b ** 2 - (self.b - self.t) ** 2) * 1.03

          def effective_area(self):
               self.A_eff = (1.5 * self.b - self.D) * 0.9 * self.t
               return self.A_eff

          def buckling_ratio(self):
               self.buckling_ratio = self.length() / self.b
               return self.buckling_ratio
               

     class Load(metaclass = IterLoad):
          _allLoads = []
          
          def __init__(self, name: str, joint: object, x_comp: float = 0.0, y_comp: float = 0.0):
               self._allLoads.append(self)
               
               self.name = name
               self.joint = joint
               self.x, self.y = x_comp, y_comp
               self.magnitude = math.sqrt(self.x ** 2 + self.y ** 2)
               self.direction = math.atan2(self.y, self.x)
               joint.loads[self.name] = (self.x, self.y)

     class Support(metaclass = IterSupport):
          _allSupports = []
          
          def __init__(self, truss: object, name: str, joint: object, support_type: str = 'encastre',
                       roller_normal_vector: tuple = (1, 0)):
               self._allSupports.append(self)
               
               self.name = name
               self.joint = joint
               self.type = support_type
               self.dir = roller_normal_vector
               if self.type in ('encastre', 'pin'):
                    joint.loads['Reaction @ {}'.format(self.name)] = (None, None)
                    # Independent unknowns: fill in later
               elif self.type == 'roller':
                    joint.loads['Reaction @ {}'.format(self.name)] = (None * self.dir[0], None * self.dir[1])
                    # Dependent unknowns: fill in later
               else:
                    raise ValueError('Support type must be "encastre", "roller" or " pin".')


     # TRUSS METACLASS METHODS

     def get_all_bars(self, str_names_only: bool = False):
          if not str_names_only:
               return [bar for bar in Truss.Bar]
          else:
               return [bar.name for bar in Truss.Bar]

     def get_all_joints(self, str_names_only: bool = False):
          if not str_names_only:
               return [joint for joint in Truss.Joint]
          else:
               return [joint.name for joint in Truss.Joint]

     def get_all_bars_connected_to_joint(self, joint: object, str_names_only: bool = False):
          if not str_names_only:
               return [bar for bar in Truss.Bar if joint.name in (bar.first_joint.name, bar.second_joint.name)]
          else:
               return [bar.name for bar in Truss.Bar if joint.name in (bar.first_joint.name, bar.second_joint.name)]

     def get_all_joints_connected_to_bar(self, bar: object, str_names_only: bool = False):
          if not str_names_only:
               return [bar.first_joint, bar.second_joint]
          else:
               return [bar.first_joint.name, bar.second_joint.name]

     def get_all_loads(self):
          return [load for load in Truss.Load]

     def get_all_loads_at_joint(self, joint: object):
          return [load for load in Truss.Load if load.joint == joint]

     def get_all_loads_at_joint_by_name(self, joint_name: str):
          return [load for load in Truss.Load if load.joint.name == joint_name]

     def get_all_supports(self):
          return [support for support in Truss.Support]

     def get_bar_by_name(self, bar_name: str):
          for bar in Truss.Bar:
               if bar.name == bar_name:
                    return bar
          
     def is_statically_determinate(self):
          b = len(self.get_all_bars(str_names_only = True))
          F = sum([2 if support.type in ('encastre', 'pin') else 1 for support in Truss.Support])
          j = len(self.get_all_joints(str_names_only = True))
          return b + F == 2 * j

     def calculate(self):
          # Get a list of the distinct joint names, number of equations to form = 2 * number of joints
          joint_names = self.get_all_joints(str_names_only = True)
          number_of_unknowns = 2 * len(joint_names)
     
          # List of dictionaries for unknowns, given default zero values
          unknowns = {}
          wanted_vars = []
          for bar in self.get_all_bars():
               unknowns['Tension in ' + bar.name] = 0
               wanted_vars.append('Tension in ' + bar.name)
          for support in self.get_all_supports():
               unknowns['Horizontal reaction at ' + support.name] = 0
               wanted_vars.append('Horizontal reaction at ' + support.joint.name)
               unknowns['Vertical reaction at ' + support.name] = 0
               wanted_vars.append('Vertical reaction at ' + support.joint.name)

          unknowns = [unknowns for x in range(number_of_unknowns)]
          
          # Create a list of joint names, with each entry included twice and then flatten the list
          joint_enum = [[joint_names[i], joint_names[i]] for i in range(len(joint_names))]
          joint_enum = [item for sublist in joint_enum for item in sublist]
          
          # Create empty dictionary of all equations in all unknowns
          unknowns = {"Equation {}, resolve {} at {}".format(
               x + 1, 'horizontally' if (x + 1) % 2 == 1 else 'vertically',
               joint_enum[x]) : unknowns[x] for x in range(number_of_unknowns)}
          
          all_directions = {}
          for joint in self.get_all_joints():
               # Reset the directions dictionary for this joint
               directions = {}
               connected_bars = self.get_all_bars_connected_to_joint(joint)
               
               # Get the anticlockwise (polar) angle of each connected joint relative to this joint which have bars
               for bar in connected_bars:
                    connected_joints = self.get_all_joints_connected_to_bar(bar)
                    if joint == connected_joints[0]:
                         angle = math.atan2(connected_joints[1].y - joint.y, connected_joints[1].x - joint.x)
                    elif joint == connected_joints[1]:
                         angle = math.atan2(connected_joints[0].y - joint.y, connected_joints[0].x - joint.x)
                    directions['Tension in ' + bar.name] = angle
                    
               # If there are reactions at this joint, store their directions too       
               if any([bool(s.joint.name == joint.name) for s in self.get_all_supports()]):
                    directions['Horizontal reaction at ' + joint.name] = 0
                    directions['Vertical reaction at ' + joint.name] = math.pi/2
                    
               # If there are external loads at this joint, store their directions too
               for l in self.get_all_loads_at_joint(joint):
                    directions['Horizontal component of {} at {}'.format(l.name , joint.name)] = 0
                    directions['Vertical component of {} at {}'.format(l.name , joint.name)] = math.pi/2

               all_directions[joint.name] = directions
          
          # Store the coefficients of the unknowns in each equation
          coefficients = []    
          for joint_name in joint_names:
               current_line = []
               for var in wanted_vars:
                    try:
                         current_line.append(round(math.cos(all_directions[joint_name][var]), 10))
                    except KeyError:
                         current_line.append(0)
               coefficients.append(current_line)
               current_line = []
               for var in wanted_vars:
                    try:
                         current_line.append(round(math.sin(all_directions[joint_name][var]), 10))
                    except KeyError:
                         current_line.append(0)
               coefficients.append(current_line)

          # Store the constants of each equation
          constants = []
          for joint_name in joint_names:
               try:
                    constants.append([-1 * sum(L.x) for L in self.get_all_loads_at_joint_by_name(joint_name)])
                    constants.append([-1 * sum(L.y) for L in self.get_all_loads_at_joint_by_name(joint_name)])
               except TypeError:
                    constants.append([-1 * L.x for L in self.get_all_loads_at_joint_by_name(joint_name)])
                    constants.append([-1 * L.y for L in self.get_all_loads_at_joint_by_name(joint_name)])
          # Sanitise load data
          for i in range(len(constants)):
               if constants[i] == [] or constants[i] == [None]:
                    constants[i] = [0]

          # Solve the system          
          M, B = np.matrix(np.array(coefficients)), np.matrix(constants)
          X = np.linalg.inv(M) * B

          # Match values back to variable names and return
          output_dict = {}
          i = 0
          for bar in self.get_all_bars():
               output_dict[bar.name] = float(X[i])
               i += 1      
          for support in self.get_all_supports():
               output_dict[support.name] = (float(X[i]), float(X[i+1]))
               i += 2
          return output_dict

     # TRUSS RESULTS CLASS

     class Result:
          def __init__(self, truss, sig_figs = None):
               self.results = truss.calculate()
               self.tensions, self.reactions, self.stresses, self.strains, self.buckling_ratios = {}, {}, {}, {}, {}
               self.sig_figs = sig_figs
               warnings.filterwarnings('ignore')
               self.get_tensions(truss)
               self.get_reactions(truss)
               self.get_stresses(truss)
               self.get_buckling_ratios(truss)
               self.get_strains(truss)
               self.round_data()
               
          def round_data(self):
               for item in list(self.tensions.keys()):
                    try:
                         self.tensions[item] = sigfig.round(self.tensions[item], self.sig_figs)
                         self.stresses[item] = sigfig.round(self.stresses[item], self.sig_figs)
                         self.strains[item] = sigfig.round(self.strains[item], self.sig_figs)
                         self.buckling_ratios[item] = sigfig.round(self.buckling_ratios[item], self.sig_figs)
                    except KeyError:
                         continue
               for item in list(self.reactions.keys()):
                    try:
                         self.reactions[item] = (sigfig.round(self.reactions[item][0], self.sig_figs),
                                                             sigfig.round(self.reactions[item][1], self.sig_figs))
                    except KeyError:
                         continue
                      
          def get_tensions(self, truss):
               for item in self.results:
                    if type(self.results[item]) == float:
                         if abs(self.results[item]) < 1e-10:
                              self.tensions.update({item : 0})
                         else:
                              self.tensions.update({item : self.results[item]})

          def get_reactions(self, truss):
               for item in self.results:
                    if type(self.results[item]) == tuple:
                         self.reactions.update({item : self.results[item]})

          def get_stresses(self, truss):
               for item in self.results:
                    if type(self.results[item]) == float:
                         self.stresses.update({item : self.tensions[item] / truss.get_bar_by_name(item).effective_area()})

          def get_strains(self, truss):
               for item in self.results:
                    if type(self.results[item]) == float:
                         self.strains.update({item : self.stresses[item] / truss.get_bar_by_name(item).E})

          def get_buckling_ratios(self, truss):
               for item in self.results:
                    if type(self.results[item]) == float and self.results[item] < 0:
                         self.buckling_ratios.update({item : truss.get_bar_by_name(item).buckling_ratio()})

# TRUSS INNER CLASSES END HERE

def print_results(results: object, truss: object, as_str: bool = True):
     if as_str:

          print('Axial forces are: (positive = tension; negative = compression) \n' + str(results.tensions))

          print('\nAxial stresses are: \n' + str(results.stresses))
          '''
          print('\nReaction forces are (horizontal, vertical) components (signs consistent with coordinate system): \n'
                + str(results.reactions))
          '''
          print('Buckling ratios are: \n' + str(results.buckling_ratios))
          print('Strains are: \n' + str(results.strains))
          if results.sig_figs == None:
               print('\nUnits are {}, {}'.format(truss.units.split(',')[0], 'values not rounded'))
          else:
               print('\nUnits are {}, {}'.format(truss.units.split(',')[0],
                                                 'values rounded to {} sig figs'.format(results.sig_figs)))

def plot_diagram(truss: object, results: object, show_reactions = False):

     # Find a suitable length-scale to make the annotations look nicer
     arrow_sizes = [x.length() for x in truss.get_all_bars()]
     arrow_sizes = sum(arrow_sizes)/len(arrow_sizes) * 0.1
     
     # Plot all joints
     plt.plot([joint.x for joint in truss.get_all_joints()], [joint.y for joint in truss.get_all_joints()], 'o')
     
     # Plot all bars and label their axial forces in the legend
     for bar in truss.get_all_bars():
          plt.plot([bar.first_joint.x, bar.second_joint.x], [bar.first_joint.y, bar.second_joint.y],
                   label = '{}'.format(bar.name + ': ' + str(results.tensions[bar.name]) + ' ' + truss.units.split(',')[0]),
                   zorder = 0)
          # If the bar is nearly vertical, label its name to its right, otherwise label it above
          if 80 * (math.pi / 180) <= abs(math.atan2(bar.second_joint.y - bar.first_joint.y,
                            bar.second_joint.x - bar.first_joint.x)) <= 100 * (math.pi / 180):
               plt.text(sum([bar.first_joint.x, bar.second_joint.x])/2 + arrow_sizes / 3,
                        sum([bar.first_joint.y, bar.second_joint.y])/2, bar.name)
          else:
               plt.text(sum([bar.first_joint.x, bar.second_joint.x])/2,
                        sum([bar.first_joint.y, bar.second_joint.y])/2 + arrow_sizes / 3, bar.name)

     # Plot all support points with their reactions as arrows
     for support in truss.get_all_supports():
          plt.plot(support.joint.x, support.joint.y, '*', color = 'red',
              label = support.name + ': ' + str(results.reactions[support.name]) + ' ' + truss.units.split(',')[0])
     for support in truss.get_all_supports():
          if show_reactions == True:
               direction_of_reaction = math.atan2(results.reactions[support.name][1], results.reactions[support.name][0])
               plt.arrow(support.joint.x, support.joint.y, arrow_sizes, 0, head_width = arrow_sizes / 5, head_length = arrow_sizes / 4)
               plt.arrow(support.joint.x, support.joint.y, 0, arrow_sizes, head_width = arrow_sizes / 5, head_length = arrow_sizes / 4)
          plt.text(support.joint.x + arrow_sizes / 4, support.joint.y + arrow_sizes / 4, support.name,
                   label = support.name + ': ' + str(results.reactions[support.name]) + ' ' + truss.units.split(',')[0])

     # Plot all loads
     for load in truss.get_all_loads():
          direction_of_load = math.atan2(load.y, load.x)
          plt.arrow(load.joint.x, load.joint.y, arrow_sizes * math.cos(direction_of_load),
                    arrow_sizes * math.sin(direction_of_load),
                    head_width = arrow_sizes / 5, head_length = arrow_sizes / 4)
          plt.text(sum([load.joint.x, load.joint.x + arrow_sizes * math.cos(direction_of_load)])/2 + arrow_sizes / 3,
                   sum([load.joint.y + load.joint.y, arrow_sizes * math.sin(direction_of_load)])/2,
                   load.name + ': (' + str(load.x) + ', ' + str(load.y) + ') ' + truss.units.split(',')[0])

     # Graphical improvements        
     plt.legend(loc = 'upper right')
     plt.autoscale(enable = True, axis = 'both')
     plt.axis('equal')
     plt.show()
     

# MAIN FUNCTIONS END HERE

def build_truss(x, print_res = True):      

     # Step 0: set the physical properties and name the truss
     custom_params = {"b" : 12.5, "t" : 0.7, "D" : 5, "E" : 210, "σ_max": 0.216}
     myTruss = Truss(custom_params, 'kN, mm')

     # Step 1: Define the joints (nodes)
     joint_A = myTruss.Joint(myTruss, "Joint A", 0, 0)
     joint_B = myTruss.Joint(myTruss, "Joint B", 290, -90)
     joint_C = myTruss.Joint(myTruss, "Joint C", 815, 127.5)
     joint_D = myTruss.Joint(myTruss, "Joint D", 290, 345)
     joint_E = myTruss.Joint(myTruss, "Joint E", 0, 255)
     joint_F = myTruss.Joint(myTruss, "Joint F", 220.836, 127.5)

     weak = {"b" : 12.5, "t" : 0.7, "D" : 5, "E" : 210, "σ_max": 0.216}
     medium_1 = {"b" : 16, "t" : 0.9, "D" : 5, "E" : 210, "σ_max": 0.216}
     medium_2 = {"b" : 16, "t" : 1.1, "D" : 5, "E" : 210, "σ_max": 0.216}
     strong = {"b" : 19, "t" : 1.1, "D" : 5, "E" : 210, "σ_max": 0.216}
     
     # Step 2: Define the bars going between any pair of joints
     bar_1 = myTruss.Bar(myTruss, "Bar 1", joint_A, joint_B, medium_2)
     bar_2 = myTruss.Bar(myTruss, "Bar 2", joint_B, joint_C, strong)
     bar_3 = myTruss.Bar(myTruss, "Bar 3", joint_C, joint_D, medium_1)
     bar_4 = myTruss.Bar(myTruss, "Bar 4", joint_D, joint_E, medium_1)
     bar_5 = myTruss.Bar(myTruss, "Bar 5", joint_E, joint_F, medium_1)
     bar_6 = myTruss.Bar(myTruss, "Bar 6", joint_F, joint_A, medium_2)
     bar_7 = myTruss.Bar(myTruss, "Bar 7", joint_F, joint_D, medium_1)
     bar_8 = myTruss.Bar(myTruss, "Bar 8", joint_F, joint_B, weak)

     # Step 3: Define the loads acting on any joint
     load_1 = myTruss.Load("W", joint_C, 0, -0.675 * 2)

     # Step 4: Define the supports acting at any joint
     support_1 = myTruss.Support(myTruss, "Support 1", joint_A, 'encastre')
     support_2 = myTruss.Support(myTruss, "Support 2", joint_E, 'encastre')

     # Step 5: Calculate the truss and print the results
     my_results = myTruss.Result(myTruss, sig_figs = 3)

     
     if print_res == True:
          print_results(my_results, myTruss, as_str = True)
          if True:
               plot_diagram(myTruss, my_results)
     else:
          return my_results

build_truss(815, True)

