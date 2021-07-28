"""
2D Truss Calculator
Version 1.4
"""

import math, warnings
from functools import total_ordering

import matplotlib                    # builtin modules

# Automatically install missing modules, least likely to be already installed first
try:
    import sigfig                                       # required for rounding values nicely
    from scipy.sparse import linalg, csr_matrix         # used for faster solving
    from matplotlib import pyplot as plt                # used to display graphical output
    import numpy as np                                  # used to do matrix operations

except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sigfig", "scipy", "numpy", "matplotlib"])
    print(' \t ~~ All dependencies succesfully installed. ~~ \n\n')

finally:
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.sparse import linalg as linsolver, csr_matrix
    import sigfig


# Utility/helper classes

class ClassIter(type):

    """
    A metaclass to support iteration over class instances. For reference see
    https://codereview.stackexchange.com/questions/126100/recording-all-instances-of-a-class-python
    https://stackoverflow.com/questions/28676399/iteration-over-class-instances-using-iter
    """

    def __iter__(cls):
        return iter(cls._ClassRegistry)

    def __len__(cls):
        return len(cls._ClassRegistry)

class SolveMethod:

    """
    A class to contain the different methods for solving the truss force balance equation
    Ax = B. Can see the different methods using get_constants(SolveMethod).
    """

    NUMPY_STD = "numpy_std"; NUMPY_SOLVE = "numpy_solve"; SCIPY = "scipy"

class Unit:

    """
    A class to contain the different units which can be used for measurements of the truss.
    Also defines the unit conversion factors. Can see the different units using get_constants(Unit).
    TODO: implement a similar constants system as SolveMethod.
    """

    # units of force
    NEWTONS = "N"; KILONEWTONS = "kN"; POUND_FORCE = "lbf"
    # units of length
    METRES = "m"; CENTIMETRES = "cm"; MILLIMETRES = "mm"; INCHES = "in"
    # units of pressure (stress)
    PASCALS = "Pa"; KILOPASCALS = "kPa"; MEGAPASCALS = "MPa"; GIGAPASCALS = "GPa"; PSI = "psi"

    # conversion lookup table, all units are converted to metric N-m-Pa internally.
    # value in N-m-Pa = value given * _CONV[unit given]
    _CONV = {
        NEWTONS: 1,  KILONEWTONS: 1e3,   POUND_FORCE: 0.224809,
        METRES:  1,  CENTIMETRES: 1e-2,  MILLIMETRES: 1e-3,      INCHES:      0.0254,
        PASCALS: 1,  KILOPASCALS: 1e3,   MEGAPASCALS: 1e6,       GIGAPASCALS: 1e9,     PSI: 6894.76
    }


# MAIN CLASS FOR TRUSSES

@total_ordering
class Truss(metaclass=ClassIter):
    """
    A class containing the truss to be worked with.
    """

    _ClassRegistry = []

    def __init__(self, name: str = "My First Truss", bar_params: dict = None, units='kN, mm'):
        """
        Initialise a truss by setting the units system to be used
        and the default properties (thickness, modulus etc) which
        bars will have when added.
        """
        self._ClassRegistry.append(self)  # add the new truss object to the list of trusses
        if bar_params is None:  # set the units that the calculations should be done in
            if units == 'N, m':
                self.default_params = {"b": 0.016, "t": 0.004, "D": 0.020,
                                       "E": 2.1e11}  # some default values. symbols defined on databook pg. 8
            elif units == 'kN, mm':
                self.default_params = {"b": 1.6, "t": 4, "D": 20, "E": 210}  # same values as above but in other units
            else:
                raise ValueError('Units must be either "N, m" or "kN, mm".')
        else:
            self.default_params = bar_params
        
        self.name = name
        self.units = units

    # PARTS OF THE TRUSS (INNER CLASSES)

    class Joint(metaclass=ClassIter):
        """
          Joints define the locations where other objects can go.
          Bars go between two joints.
          Each joint can have loads and supports applied.
          """
        _ClassRegistry = []

        def __init__(self, truss: object, name: str, x: float, y: float):

            self.name = name

            if not self.name in (i.name for i in self._ClassRegistry):
                            self._ClassRegistry.append(self)
    
            self.truss = truss
            self.x = x
            self.y = y
            self.loads = {}

    class Bar(metaclass=ClassIter):
        """
          Bars go between the first_joint and the second_joint.
          Each bar can have different thickness, strength etc in my_params.
          """
        _ClassRegistry = []

        def __init__(self, truss: object, name: str, first_joint: object, second_joint: object,
                     my_params: dict = None):
            """
            Initialise a bar with a given name, which joints it should connect between, and
            its physical properties.
            """

            self.name = name  # The user-defined name

            if not self.name in (i.name for i in self._ClassRegistry):
                self._ClassRegistry.append(self)
            
            self.truss = truss                                                            # the class which this bar belongs to
            self.first_joint, self.first_joint_name = first_joint, first_joint.name       # the object and name of the connected joints
            self.second_joint, self.second_joint_name = second_joint, second_joint.name
            self.params = truss.default_params if my_params is None else my_params        # take the truss's default if bar not given any

            # physical and geometric properties of the bar, as defined on databook pg. 8
            [setattr(self, attr, self.params[attr]) for attr in ["b", "t", "D", "E", "strength_max"]]
            self.length = math.sqrt((self.first_joint.x - self.second_joint.x)**2 + (self.first_joint.y - self.second_joint.y)**2)
            self.section_area = (self.b ** 2 - (self.b - self.t) ** 2) * 1.03
            self.effective_area = (1.5 * self.b - self.D) * 0.9 * self.t
            self.buckling_ratio = self.length / self.b

        def get_direction(self, origin_joint: object = None, as_degrees: bool = False):
            """
            Calculates the (polar) angle this bar makes with the horizontal, 
            with the origin taken as the origin_joint. 0 = horizontal right, +pi/2 = vertical up, 
            -pi/2 = vertical down, pi = horizontal left, etc. (-pi < angle <= pi).
            """
            connected_joints = self.truss.get_all_joints_connected_to_bar(self)

            if origin_joint in connected_joints:
                other_joint_index = 1 - connected_joints.index(origin_joint)
                angle = math.atan2(connected_joints[other_joint_index].y - origin_joint.y,
                                   connected_joints[other_joint_index].x - origin_joint.x)

            elif origin_joint is None:  # if no joint is specified, the joint is chosen such that the angle 
                                        # is not upside-down (used to allign the text along the bars)
                angle_from_first = self.get_direction(self.first_joint, as_degrees=as_degrees)
                angle_from_second = self.get_direction(self.second_joint, as_degrees=as_degrees)
                if as_degrees and -90 < angle_from_first <= 90 or \
                    not as_degrees and -1 * math.pi / 2 < angle_from_first <= math.pi / 2:
                    return angle_from_first
                else:
                    return angle_from_second

            else:
                raise SyntaxError(f'The bar "{self.name}" has an invalid origin joint when finding \n'
                                  f'its direction. It should be the objects associated with either \n'
                                  f'{self.first_joint_name} or {self.second_joint_name}.')
            
            return angle if not as_degrees else math.degrees(angle)

    class Load(metaclass=ClassIter):
        """
          Loads can be applied at any joint.
          Their components are specified as (x_comp, y_comp)
          aligned with the coordinate system used to define the joints.
          """
        _ClassRegistry = []

        def __init__(self, name: str, joint: object, x_comp: float = 0.0, y_comp: float = 0.0):
            """
            Initialise a load with a name, a joint to be applied at, and a force value in terms
            of x and y components (as defined by the coordinate and unit systems).
            """

            self.name = name

            if not self.name in (i.name for i in self._ClassRegistry):
                self._ClassRegistry.append(self)

            self.joint = joint
            self.x, self.y = x_comp, y_comp
            self.magnitude = math.sqrt(self.x ** 2 + self.y ** 2)  # magnitude of the force
            self.direction = math.atan2(self.y, self.x)  # direction of the force clockwise from the positive x-axis
            joint.loads[self.name] = (self.x, self.y)  # add this load's components to the joint's dict attribute

    class Support(metaclass=ClassIter):
        """
        Supports are points from which external reaction forces can be applied.
        """
        _ClassRegistry = []

        def __init__(self, truss: object, name: str, joint: object, support_type: str = 'pin',
                     roller_normal_vector: tuple = None, pin_rotation: float = 0):
            """
            Initialise a support with a name, a joint object to convert to a support, the type of support
            and a direction if a roller joint is chosen.

            support_type:         can be 'pin' or 'roller' or 'encastre'
            roller_normal_vector: only relevant with roller joints, sets the direction of their reaction force
            pin_rotation:         only relevant with pin joints, sets the direction which they are displayed
            """

            self.name = name

            if not self.name in (i.name for i in self._ClassRegistry):
                self._ClassRegistry.append(self)
            
            self.joint = joint
            self.support_type = support_type
            self.pin_rotation = pin_rotation

            if roller_normal_vector not in [None, (0, 0)]:
                self.roller_normal_vector = np.array(roller_normal_vector) / np.linalg.norm(roller_normal_vector)
                self.direction_of_reaction = math.atan2(*reversed(self.roller_normal_vector))
            else:
                self.roller_normal_vector = None
                self.direction_of_reaction = None
                
            if self.support_type in {'encastre', 'pin', 'roller'}:
                joint.loads[f'Reaction @ {self.name}'] = (None, None)
            else:
                raise ValueError('Support type must be "encastre", "pin" or "roller".')


    # TRUSS RESULTS CLASS

    class Result:
        """
        Allows the results to be analysed and manipulated.
        """

        def __init__(self, truss, sig_figs=None, solution_method=SolveMethod.NUMPY_SOLVE, delete_truss_after=False):
            self.truss = truss
            self.results = truss.calculate(solution_method=solution_method)
            self.tensions, self.reactions, self.stresses, self.strains, self.buckling_ratios = {}, {}, {}, {}, {}
            self.sig_figs = sig_figs
            warnings.filterwarnings('ignore')
            self.get_data(truss)
            self.round_data()
            if delete_truss_after:
                truss._delete_truss()

        def __repr__(self):
            repr_str  = f'\n Axial forces are: '\
                        f'(positive = tension; negative = compression) \n \t {str(self.tensions)}'
            repr_str += f'\n Axial stresses are: \n \t {str(self.stresses)}'
            repr_str += f'\n Reaction forces are (horizontal, vertical) components (signs '\
                        f'consistent with coordinate system): \n \t {str(self.reactions)}'
            repr_str += f'\n Buckling ratios are: \n \t {str(self.buckling_ratios)}'
            repr_str += f'\n Strains are: \n \t {str(self.strains)}'
            repr_str += f'\n\n Units are {self.truss.units.split(",")[0]}, values '\
                        f'{f"not rounded" if self.sig_figs is None else f"rounded to {self.sig_figs} s.f."}'
            return repr_str

        def round_data(self):
            """
            Replaces the calculated data with rounded values, to precision given by Truss.Result.sig_figs.
            """
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

        def get_data(self, truss):
            """
            Calculate tensions, stresses, strains, reaction forces and buckling ratios
            from the calculate() function.
            """
            for item in self.results:
                if isinstance(self.results[item], float):
                    if abs(self.results[item]) < 1e-10:
                        self.tensions.update({item: 0})
                    else:
                        self.tensions.update({item: self.results[item]})
                    self.stresses.update({item: self.tensions[item] / truss.get_bar_by_name(item).effective_area})
                    self.strains.update({item: self.stresses[item] / truss.get_bar_by_name(item).E})
                    self.buckling_ratios.update({item: truss.get_bar_by_name(item).buckling_ratio})  
                    # check if the bar is in compression using: if self.results[item] < 0:

                elif isinstance(self.results[item], tuple):
                    self.results[item] = tuple(map(lambda x: round(x, self.sig_figs), self.results[item]))
                    self.reactions.update({item: self.results[item]})
                else:
                    warnings.warn(f'''A result appears to have been formed incorrectly. This is an internal
                                    error. Bad value ignored: {self.results[item]}''', RuntimeWarning)
                    continue


    # TRUSS METHODS

    def calculate(self, solution_method=SolveMethod.SCIPY):

        """
        The main part of the program. Calculates the forces in the truss's bars and supports
        in order to maintain force equilibrium with the given loads. Outputs as a dictionary in the form
        {bar_name: axial_force_value} + {support_name: (reaction_force_value_x, reaction_force_value_y)}
        """

        # Get a list of the distinct joint names, number of equations to form = 2 * number of joints
        joint_names = self.get_all_joints(str_names_only=True)

        # List of dictionaries for unknowns, given default zero values
        wanted_vars = []
        for bar in self.get_all_bars():
            wanted_vars.append('Tension in ' + bar.name)
        for support in self.get_all_supports():
            if support.support_type in {'pin', 'encastre'}:
                wanted_vars.append('Horizontal reaction at ' + support.joint.name)
                wanted_vars.append('Vertical reaction at ' + support.joint.name)
            elif support.support_type == 'roller':
                wanted_vars.append('Magnitude of reaction at ' + support.joint.name)
            else:
                continue

        all_directions = {}
        for joint in self.get_all_joints():
            # Reset the directions dictionary for this joint
            directions = {}
            connected_bars = self.get_all_bars_connected_to_joint(joint)

            # Get the anticlockwise (polar) angle of each connected joint relative to this joint which have bars
            for bar in connected_bars:
                angle = bar.get_direction(joint)
                directions['Tension in ' + bar.name] = angle

            # If there are reactions at this joint, store their directions too
            if any([s.joint.name == joint.name for s in self.get_all_supports()]):
                if self.get_support_by_joint(joint).support_type == 'roller':
                    directions['Magnitude of reaction at ' + joint.name] = math.atan2(
                        *reversed(list(self.get_support_by_joint(joint).roller_normal_vector)))
                else:
                    directions['Horizontal reaction at ' + joint.name] = 0
                    directions['Vertical reaction at ' + joint.name] = math.pi / 2

            # If there are external loads at this joint, store their directions too
            for load in self.get_all_loads_at_joint(joint):
                directions['Horizontal component of {} at {}'.format(load.name, joint.name)] = 0
                directions['Vertical component of {} at {}'.format(load.name, joint.name)] = math.pi / 2

            all_directions[joint.name] = directions

        # Store the coefficients of the unknowns in each equation
        coefficients = []
        for joint_name in joint_names:
            current_line = [round(math.cos(all_directions[joint_name].get(var, math.pi/2)), 10) for var in wanted_vars]
            coefficients.append(current_line)
            current_line = [round(math.sin(all_directions[joint_name].get(var, 0)), 10) for var in wanted_vars]
            coefficients.append(current_line)

        # Store the constants of each equation, negative since they are on the other side of the system of equations
        constants = []
        for joint_name in joint_names:
            try:
                constants.append([-1 * sum(load.x) for load in 
                                  self.get_all_loads_at_joint_by_name(joint_name)])
                constants.append([-1 * sum(load.y) for load in
                                  self.get_all_loads_at_joint_by_name(joint_name)])
            except TypeError:
                constants.append([-1 * load.x for load in
                                  self.get_all_loads_at_joint_by_name(joint_name)])
                constants.append([-1 * load.y for load in
                                  self.get_all_loads_at_joint_by_name(joint_name)])

        # Sanitise load data
        for i in range(len(constants)):
            if constants[i] == [] or constants[i] == [None]:
                constants[i] = [0]
        
        # Solve the system - both coefficient and constant matrices are 
        # sparse (for most practical cases) so ideally the SCIPY method is faster.

        if solution_method is SolveMethod.NUMPY_STD:
            m, b = np.matrix(coefficients), np.matrix(constants)
            x = np.linalg.inv(m) * b
        elif solution_method is SolveMethod.NUMPY_SOLVE:
            m, b = np.matrix(coefficients), np.matrix(constants)
            x = np.linalg.solve(m, b)
        elif solution_method is SolveMethod.SCIPY:
            m, b = csr_matrix(coefficients), csr_matrix(constants)
            x = linsolver.spsolve(m, b)
        else:
            raise SyntaxError(f"The solution method {solution_method} is not supported. \n"
                              f"The allowed methods are (either using constants or string literals): \n"
                              f"{get_constants(SolveMethod)}\n"
                              f"For example: \t solution_method=SolveMethod.NUMPY_SOLVE \t or \t solution_method='numpy_solve'")

        # Match values back to variable names
        output_dict = {}
        for i, bar in enumerate(self.get_all_bars()):
            output_dict[bar.name] = float(x[i])
        else:
            _i = i
        for support in self.get_all_supports():
            output_dict[support.name] = (float(x[_i]), float(x[_i + 1]))
            _i += 2

        # For whatever reason, sometimes the pin jointed reaction forces are wrong.
        # Couldn't be bothered fixing the root cause so correct them here by resolving at the supports.
        for support in self.get_all_supports():
            reaction_corrected = [0, 0]
            for bar in self.get_all_bars_connected_to_joint(support.joint):
                angle = bar.get_direction(support.joint)
                reaction_corrected[0] -= output_dict[bar.name] * math.cos(angle)
                reaction_corrected[1] -= output_dict[bar.name] * math.sin(angle)

            output_dict[support.name] = tuple(reaction_corrected)
        
        # Return the values in dict form
        return output_dict

    def is_statically_determinate(self):
        """
        Does a simple arithmetic check to estimate if the truss
        is statically determinate (b + F = 2j). Also stores attributes for later quick use.
        """
        self.b = len(self.get_all_bars(str_names_only=True))
        self.F = sum([2 if support.support_type in {'encastre', 'pin'} 
                 else 1 if support.support_type == 'roller' 
                 else 0 for support in Truss.Support])
        self.j = len(self.get_all_joints(str_names_only=True))
        return self.b + self.F == 2 * self.j

    def classify_error_in_truss(self, e: np.linalg.LinAlgError):
        """
        If there was an exception raised when solving, attempt to find the cause and raise
        a more user-friendly exception message.
        """
        valid = self.is_statically_determinate()
        if not valid:
            raise ArithmeticError(f'''The truss is not statically determinate. 
                It cannot be solved. \nBars: {self.b} \t Reactions: {self.F} \t Joints: {self.j}.
                \n b + F = {self.b + self.F}, 2j = {2 * self.j}''')
        elif str(e) == "Singular matrix":
            raise TypeError('''
            The truss contains mechanistic and/or overconstrained components despite
            being globally statically determinate. It cannot be solved.''')
        else:
            raise TypeError("Something else went wrong. Couldn't identify the problem.")

    @classmethod
    def _delete_truss(cls):
        """
        Delete the truss and clear the _ClassRegistry when the calculation for a truss 
        is done. Required to prevent the _ClassRegistry adding duplicate objects.
        """
        from inspect import isclass

        all_objs = {**cls.__dict__, 'Truss': cls}.values()
        current_classes = filter(lambda c: isclass(c) and hasattr(c, '_ClassRegistry'), all_objs)
        for c in current_classes:
            setattr(c, '_ClassRegistry', [])

    '''
    Allow ordering of the trusses by their position in the _ClassRegistry
    which represents the order they were created in. Used by @functools.total_ordering.
    '''
    def __le__(self, other):
        return self._ClassRegistry.index(self) <= self._ClassRegistry.index(other)

    def __eq__(self, other):
        return self._ClassRegistry.index(self) == self._ClassRegistry.index(other)
    
    
    """
    Returns all objects of a given type by their name or object reference.
    """

    @staticmethod
    def get_all_bars(str_names_only: bool = False):
        """
        Returns a list of bar objects or strings in this truss.
        """
        if not str_names_only:
            return [bar for bar in Truss.Bar]
        else:
            return {bar.name for bar in Truss.Bar}

    @staticmethod
    def get_all_joints(str_names_only: bool = False):
        """
        Returns a list of all joint objects or strings in this truss.
        """
        if not str_names_only:
            return [joint for joint in Truss.Joint]
        else:
            return {joint.name for joint in Truss.Joint}

    @staticmethod
    def get_all_bars_connected_to_joint(joint: Joint, str_names_only: bool = False):
        """
        Returns a list of bar objects or names which are connected to a given joint object.
        """
        if not str_names_only:
            return [bar for bar in Truss.Bar if joint.name in {bar.first_joint.name, bar.second_joint.name}]
        else:
            return {bar.name for bar in Truss.Bar if joint.name in {bar.first_joint.name, bar.second_joint.name}}

    @staticmethod
    def get_all_joints_connected_to_bar(bar: Bar, str_names_only: bool = False):
        """
        Returns a list of joint objects or names which are connected to a given bar object.
        """
        if not str_names_only:
            return (bar.first_joint, bar.second_joint)
        else:
            return (bar.first_joint.name, bar.second_joint.name)

    @staticmethod
    def get_all_loads():
        """
        Returns a list of load objects in the truss.
        """
        return [load for load in Truss.Load]

    @staticmethod
    def get_all_loads_at_joint(joint: Joint):
        """
        Returns a list of load objects which are applied at a given joint object.
        """
        return [load for load in Truss.Load if load.joint == joint]

    @staticmethod
    def get_all_loads_at_joint_by_name(joint_name: str):
        """
        Returns a list of load objects which are applied at a given joint name.
        """
        return [load for load in Truss.Load if load.joint.name == joint_name]

    @staticmethod
    def get_all_supports():
        """
        Returns a list of support objects in the truss.
        """
        return [support for support in Truss.Support]

    @staticmethod
    def get_support_by_joint(joint: Joint):
        """
        Returns the support object placed at a given joint, 
        or None if there is no support there.
        """
        _supports = [support for support in Truss.Support if support.joint == joint]
        return _supports[0] if len(_supports) >= 1 else None

    @staticmethod
    def get_bar_by_name(bar_name: str):
        """
        Returns the corresponding bar object given a bar name.
        """
        for bar in Truss.Bar:
            if bar.name == bar_name:
                return bar


# TRUSS INNER CLASSES END HERE, MAIN RESULTS FUNCTIONS START HERE

def plot_diagram(truss: Truss, results: Truss.Result, show_reactions=False, delete_truss_after=True):
    """
     Create a matplotlib output image showing the truss geometry,
     annotated with arrows and labels.
     """

    global LEN

    # Find a suitable length-scale to make the annotations look nicer - 10% of the average length of the bars
    LEN = [x.length for x in truss.get_all_bars()]
    LEN = np.average(LEN) * 0.1

    # Plot all joints without supports
    _xjl, _yjl = map(list, zip(*[(joint.x, joint.y) for joint in truss.get_all_joints() 
                                 if truss.get_support_by_joint(joint) is None]))
    
    plt.plot(_xjl, _yjl, 'o', color='black', markersize=5)
    plt.plot(_xjl, _yjl, 'o', color='white', markersize=3.5)  # small circle with white centre

    # Plot all bars
    for bar in truss.get_all_bars():

        rot = bar.get_direction(as_degrees=True)
        norm = math.radians(rot + 90)

        plt.plot([bar.first_joint.x, bar.second_joint.x], [bar.first_joint.y, bar.second_joint.y],
                 label='{}'.format(bar.name + ': ' + str(results.tensions[bar.name]) + ' ' + truss.units.split(',')[0]),
                 zorder=0)

        plt.text((bar.first_joint.x + bar.second_joint.x) / 2 + LEN / 3 * math.cos(norm), 
                 (bar.first_joint.y + bar.second_joint.y) / 2 + LEN / 3 * math.sin(norm), 
                 bar.name, ha='center', va='center', rotation=rot, rotation_mode='anchor',
                 transform_rotates_text=True)

    # Plot all supports
    for support in truss.get_all_supports():

        plt.plot(support.joint.x, support.joint.y, '*', markersize=0,
                 label=support.name + ': ' + \
                       str(results.reactions[support.name]) + ' ' + truss.units.split(',')[0])

    for support in truss.get_all_supports():
        if show_reactions:
            direction_of_reaction = math.atan2(*reversed(list(results.reactions[support.name])))

            plt.arrow(support.joint.x, support.joint.y, 
                      LEN * math.cos(direction_of_reaction),
                      LEN * math.sin(direction_of_reaction),
                      head_width=LEN/5, head_length=LEN/4, facecolor='red')

        label_angle = find_free_space_around_joint(support.joint, results, show_reactions=show_reactions)

        plt.text(support.joint.x + 0.9 * LEN * math.cos(label_angle),
                 support.joint.y + 0.9 * LEN * math.sin(label_angle),
                 support.name, va='center', ha='left' if -90 < math.degrees(label_angle) <= 90 else 'right',
                 label=f'{support.name}: {str(results.reactions[support.name])} {truss.units.split(",")[0]}')

        draw_support(support.joint.x, support.joint.y, LEN*0.9, 
                     support_type=support.support_type, roller_normal_vector=support.roller_normal_vector,
                     pin_rotation=support.pin_rotation)

    # Plot all loads
    for load in truss.get_all_loads():

        label_angle = find_free_space_around_joint(load.joint)

        plt.arrow(load.joint.x, load.joint.y, LEN * math.cos(load.direction), LEN * math.sin(load.direction),
                  head_width=LEN / 5, head_length=LEN / 4)

        plt.text(load.joint.x + LEN / 3 * math.cos(label_angle), load.joint.y + LEN / 3 * math.sin(label_angle),
                 f'{load.name}: ({str(load.x)}, {str(load.y)}) {truss.units.split(",")[0]}',
                 va='center', ha='left' if -90 < math.degrees(label_angle) <= 90 else 'right')

    # Delete the truss registry to avoid issues if building another truss
    if delete_truss_after:
        truss._delete_truss()

    # Graphical improvements
    AXES_COLOUR = '#BBBBBB'
    plt.title(truss.name)
    plt.legend(loc='upper right'); plt.autoscale(); plt.axis('equal')
    plt.xlabel(f'$x$-position / {truss.units.split(",")[1]}')
    plt.ylabel(f'$y$-position / {truss.units.split(",")[1]}')
    ax, spines = plt.gca(), plt.gca().spines 
    spines['right'].set_visible(False); spines['top'].set_visible(False)
    spines['left'].set_color(AXES_COLOUR); spines['bottom'].set_color(AXES_COLOUR)
    ax.tick_params(axis='x', colors=AXES_COLOUR); ax.tick_params(axis='y', colors=AXES_COLOUR)
    ax.xaxis.label.set_color(AXES_COLOUR); ax.yaxis.label.set_color(AXES_COLOUR)
    set_matplotlib_fullscreen(); plt.show()


# HELPER AND UTILITY FUNCTIONS

def validate_var_name(var_name: str, allow_existing_vars=True):
    """
    Checks if a var_name, which is used internally to instantiate the
    subclass objects (Joint, Bars, Load, Support) is as valid as if it
    were declared explicitly i.e. var_name = Class(...). They are set using
    globals() where the key is var_name and the object reference is
    the value.
    """
    import keyword

    if var_name in globals() and not allow_existing_vars:
        raise NameError(f'A global variable {var_name} (with the value {globals()[var_name]}) is already in use, '
                        f'possibly because it is a builtin. \nIt cannot be used in the truss.')
    elif not var_name.isidentifier() or keyword.iskeyword(var_name) or var_name.startswith('__'):
        raise NameError(f'{var_name} is not a valid variable name. \n'
                         'It can only contain alphanumerics and underscores \n'
                         'and cannot start with double underscore (__).')
    else:
        return True

def convert_to_valid_var_name(name: str, allow_existing_vars=True):
    """
    Given a user-defined name, converts it to a similar looking valid variable name.
    e.g. convert_to_valid_var_name("My First Truss") = my_first_truss
    If this already exists and allow_existing_vars = False, a number is appended to the name
    to make it distinct, e.g. my_first_truss_2, my_first_truss_3, etc.
    """

    import re

    # remove trailing whitespace, convert to lowercase and replace spaces with underscores
    new_name = name.strip().lower().replace(' ', '_')
    # remove non-alphanumeric characters
    pattern = re.compile('[\W_]+', re.UNICODE)
    new_name = pattern.sub('', new_name)

    if not allow_existing_vars and new_name in globals().keys():
        suffix = 2
        while not validate_var_name(new_name + '_' + str(suffix)):
            suffix += 1
    
    if validate_var_name(new_name):
        return new_name
    else:
        raise SyntaxError(f'Unable to convert the name {name} to a suitable internal variable name'
                          f'(attempt was {new_name}). Please change to a simpler name and try again.')

def get_constants(cls: type):
    """
    Used to get a dict of constants {const_name: const_value}
    from the utility classes.
    """
    names = list(filter(
        lambda a: not callable(getattr(cls(), a)) and not a.startswith('_') and a == a.upper(), dir(cls())))
    vals = [getattr(cls(), a) for a in names]

    return dict(zip(names, vals))

def set_active_truss(var_name: str):
    """
    Sets which truss is currently being built on.
    """
    global active_truss
    active_truss = globals()[var_name]

def get_active_truss():
    """
    Gets the truss which is currently being built on, or None if there is none.
    NOTE: active_truss is a global var.
    """
    return active_truss if has_active_truss() else None

def is_active_truss(var_name: str):
    """
    Determines whether the given truss variable name is being built on.
    """
    return globals()[var_name] is active_truss

def has_active_truss():
    """
    Determines whether an active truss has been set yet, returning True or False.
    """
    return 'active_truss' in globals().keys()

def set_matplotlib_fullscreen():
    """
    Automatically set the matplotlib output to fullscreen.
    """
    import os
    from matplotlib import pyplot as plt

    backend = str(plt.get_backend())
    mgr = plt.get_current_fig_manager()
    if backend == 'TkAgg':
        if os.name == 'nt':
            mgr.window.state('zoomed')
        else:
            mgr.resize(*mgr.window.maxsize())
    elif backend == 'wxAgg':
        mgr.frame.Maximize(True)
    elif backend in ['Qt4Agg', 'Qt5Agg']:
        mgr.window.showMaximized()
    else:
        raise RuntimeWarning(f'The backend in use, {backend}, is not supported in fullscreen mode.')

def find_free_space_around_joint(joint: Truss.Joint, results: Truss.Result = None, truss: Truss = None, 
                                 show_reactions: bool = True, as_degrees: bool = False):
    """
    Helper function to find a place to label text around a joint. Finds a location
    at a fixed small distance from the joint, such that the surrounding bars, loads
    and supports/reaction arrows are as far away as possible.
    """

    truss = active_truss if truss is None else truss
    support = truss.get_support_by_joint(joint)
    used_angles = [bar.get_direction(origin_joint=joint) for bar in truss.get_all_bars_connected_to_joint(joint)]
    used_angles += [load.direction for load in truss.get_all_loads_at_joint(joint)]

    if support is not None:
        if show_reactions:
            if support.support_type == 'roller':
                used_angles.append(math.pi + support.direction_of_reaction)
            used_angles.append(math.atan2(*reversed(results.reactions[support.name])))
        else:
            if support.support_type == 'pin':
                used_angles.append(math.pi / 2 - support.pin_rotation)

    used_angles = sorted([i % (2 * math.pi) for i in used_angles])

    differences = [(used_angles[i] - used_angles[i-1]) % (2 * math.pi) for i in range(len(used_angles))]
    max_i = differences.index(max(differences))
    most_free_angle = np.average([used_angles[max_i], used_angles[max_i-1]])
    if used_angles[max_i] < used_angles[max_i-1]:
        most_free_angle -= math.pi

    return math.degrees(most_free_angle) if as_degrees else most_free_angle

def draw_support(x: float, y: float, size: float,
                 support_type: str = 'pin', pin_rotation: float = 0, roller_normal_vector: tuple = None):
    """
    Draw a particular type of support, using the standard conventional symbols, on
    the matplotlib truss diagram. If roller is chosen, its direction is 
    shown by rotating the drawing. Optional pin rotation in clockwise degrees from vertical.
    """
    import math
    from matplotlib import pyplot as plt

    # Helper function to rotate the drawing
    if (pin_rotation != 0) ^ (roller_normal_vector is not None):  # either but not both: cannot be encastre
        if support_type == 'roller':
            a = math.pi / 2 - math.atan2(*reversed(roller_normal_vector))
        elif support_type == 'pin':
            a = math.radians(pin_rotation)
        else:
            raise TypeError(f'''
            'The combination of supplied information: support type ({support_type}), pin rotation angle'
            '({pin_rotation}) and roller direction ({roller_normal_vector}) is invalid.''')
        
        # function for rotating a given coordinate tuple _p = (_x, _y) by a radians clockwise about (x, y)
        rot = lambda _p: (x + (_p[0] - x) * math.cos(a) + (_p[1] - y) * math.sin(a),
                        y - (_p[0] - x) * math.sin(a) + (_p[1] - y) * math.cos(a))

    if support_type == 'encastre':

        # Encastre symbol: solid line and hashed lines representing ground
        plt.plot((x - size / 2, x + size / 2), (y, y),                              # horizontal line
                 linewidth=1, color='black', zorder=0)
        for x_pos in np.linspace(x - 0.3 * size, x + 0.5 * size, 5):
            plt.plot((x_pos, x_pos - size / 5), (y, y - size / 5),                  # hashed lines
                     linewidth=1, color='black', zorder=0)

    if (support_type == 'pin' and pin_rotation != 0) or support_type == 'roller':
        # NOTE: element indices are
        # 0: triangle top left, 1: triangle bottom left, 2: triangle bottom right, 3: triangle top right
        # 4,5,6,7,8: ground top right diagonal points, 9,10,11,12,13: ground bottom left diagonal points
        # 14: ground left point, 15: ground right point
        _old_pts = [
                (x - size / 20, y - math.sqrt(3) * size / 20),
                (x - (1 / (3 * math.sqrt(3))) * size, y - size / 3),
                (x + (1 / (3 * math.sqrt(3))) * size, y - size / 3),
                (x + size / 20, y - math.sqrt(3) * size / 20)
            ] + [(x_pos, y - (size / 3 if support_type == 'pin' else 8/15 * size)) for x_pos, y_pos in \
                zip(list(np.linspace(x - 0.3 * size, x + 0.5 * size, 5)), [y] * 5)
            ] + [(x_pos - size / 5, y - (8/15 * size if support_type == 'pin' else 11/15 * size)) for x_pos, y_pos in \
                zip(list(np.linspace(x - 0.3 * size, x + 0.5 * size, 5)), [y] * 5)
            ] + [(x - size / 2, y - (size / 3 if support_type == 'pin' else 8/15 * size)), 
                 (x + size / 2, y - (size / 3 if support_type == 'pin' else 8/15 * size))]

    if support_type == 'pin':
        if pin_rotation == 0:
            # Pin symbol: triangle resting on ground
            plt.plot((x - size / 20, x - (1 / (3 * math.sqrt(3))) * size,               # equilateral triangle
                    x + (1 / (3 * math.sqrt(3))) * size, x + size / 20),
                    (y - math.sqrt(3) * size / 20, y - size / 3, 
                    y - size / 3, y - math.sqrt(3) * size / 20), 
                    linewidth=1, color='black', zorder=0)

            plt.gca().add_patch(                                                        # circle pin
                plt.Circle((x, y), size / 10, color='black', linewidth=1, zorder=1))
            plt.gca().add_patch(
                plt.Circle((x, y), size / 14, color='white', linewidth=1, zorder=1))

            plt.plot((x - size / 2, x + size / 2), (y - size / 3, y - size / 3),        # ground
                    linewidth=1, color='black', zorder=0)
            for x_pos in np.linspace(x - 0.3 * size, x + 0.5 * size, 5):
                plt.plot((x_pos, x_pos - size / 5), (y - size / 3, y - 8/15 * size),
                        linewidth=1, color='black', zorder=0)
        else:
            # Transform the important points to be plotted
            _new_pts = list(map(rot, _old_pts))
            xtl, ytl = map(list, zip(*_new_pts))

            plt.plot(xtl[0:4], ytl[0:4], linewidth=1, color='black', zorder=0)          # triangle

            plt.gca().add_patch(                                                        # circle pin
                plt.Circle((x, y), size / 10, linewidth=1, zorder=1,
                        color='black'))
            plt.gca().add_patch(
                plt.Circle((x, y), size / 14, linewidth=1, zorder=1,
                        color='white'))
            
            plt.plot(xtl[14:], ytl[14:], linewidth=1, color='black', zorder=0)          # ground
            for i, (x_tr, y_tr) in enumerate(_new_pts[4:9]):
                n = i + 4
                plt.plot([x_tr, _new_pts[n+5][0]], [y_tr, _new_pts[n+5][1]],
                    linewidth=1, color='black', zorder=0)

    if support_type == 'roller':
        # Roller symbol: pin with wheels, rotated about pin circle to show direction
        # Transform the important points to be plotted 
        # NOTE: element indices are (0-15 unchanged) from pin
        # 16: wheel left centre point, 17: wheel right centre point

        _old_pts += [(x - (0.7 / (3 * math.sqrt(3))) * size, y - 13/30 * size),
             (x + (0.7 / (3 * math.sqrt(3))) * size, y - 13/30 * size)]

        _new_pts = list(map(rot, _old_pts))
        xtl, ytl = map(list, zip(*_new_pts))

        plt.plot(xtl[0:4], ytl[0:4], linewidth=1, color='black', zorder=0)          # triangle
        
        plt.gca().add_patch(                                                        # circle pin
            plt.Circle((x, y), size / 10, linewidth=1, zorder=1,
                       color='black'))
        plt.gca().add_patch(
            plt.Circle((x, y), size / 14, linewidth=1, zorder=1,
                       color='white'))
        
        plt.plot(xtl[14:16], ytl[14:16], linewidth=1, color='black', zorder=0)      # ground
        for i, (x_tr, y_tr) in enumerate(_new_pts[4:9]):
            n = i + 4
            plt.plot([x_tr, _new_pts[n+5][0]], [y_tr, _new_pts[n+5][1]],
                linewidth=1, color='black', zorder=0)
        
        plt.gca().add_patch(                                                        # wheels
            plt.Circle((xtl[16], ytl[16]), size / 10, color='black', linewidth=1, zorder=1))
        plt.gca().add_patch(
            plt.Circle((xtl[16], ytl[16]), size / 14, color='white', linewidth=1, zorder=1))
        plt.gca().add_patch(
            plt.Circle((xtl[17], ytl[17]), size / 10, color='black', linewidth=1, zorder=1))
        plt.gca().add_patch(
            plt.Circle((xtl[17], ytl[17]), size / 14, color='white', linewidth=1, zorder=1))


# OBJECT BUILDING FUNCTIONS

'''
Allows trusses to be constructed with user-defined names instead of fixed variable names.
Objects are still stored internally with names given by var_name
but displayed to the user as joint_name, bar_name, load_name, support_name.

This is done by directly accessing the globals() dictionary
and adding {var_name : some_object_reference} to it.
'''

def create_truss(truss_name: str, bar_params: dict = None, units: str = 'kN, mm', 
                 set_as_active_truss: bool = True, var_name: str = None, print_info = False):
    """
    Create an instance of a support in a truss, with a user defined name support_name,
    stored internally as var_name, at joint variable name string joint_var_name.
    """

    var_name = convert_to_valid_var_name(truss_name) if var_name is None else var_name

    for var, val in globals().copy().items():
        if hasattr(val, 'name') and val.name == truss_name:
            var_name = var
        
    if validate_var_name(var_name):
        globals()[var_name] = Truss(name=truss_name, bar_params=bar_params, units=units)

    if print_info:
        print(f'The truss with name "{globals()[var_name].name}", internally stored as "{var_name}", '
        f'has been created with bar parameters {bar_params} and units {units}.')

    if set_as_active_truss: set_active_truss(var_name)

def create_joint(joint_name: str, x: float, y: float, 
                 truss: Truss = None, var_name: str = None, print_info=False):
    """
    Create an instance of a joint in a truss, with a user defined name joint_name,
    stored internally as var_name, at position (x, y).
    """

    truss = active_truss if truss is None else truss
    var_name = convert_to_valid_var_name(joint_name) if var_name is None else var_name

    if validate_var_name(var_name):
        globals()[var_name] = truss.Joint(truss, joint_name, x, y)

    if print_info:
        print(f'The joint with name "{globals()[var_name].name}", internally stored as "{var_name}", '
        f'has been assigned the location ({globals()[var_name].x}, {globals()[var_name].y})')

def create_bar(bar_name: str, first_joint_name: str, second_joint_name: str, 
               params: dict = None, truss: Truss = None, var_name: str = None, print_info=False):
    """
    Create an instance of a bar in a truss, with a user defined name bar_name,
    stored internally as var_name, between two joints with string names, with bar_params.
    """

    truss = active_truss if truss is None else truss
    var_name = convert_to_valid_var_name(bar_name) if var_name is None else var_name

    for var, val in globals().copy().items():
        if hasattr(val, 'name') and val.name == first_joint_name:
            first_joint_var_name = var
        if hasattr(val, 'name') and val.name == second_joint_name:
            second_joint_var_name = var

    if validate_var_name(var_name):
        globals()[var_name] = truss.Bar(truss, bar_name, globals()[first_joint_var_name],
                                        globals()[second_joint_var_name], params)

    if print_info:
        print(f'The bar with name "{globals()[var_name].name}", internally stored as "{var_name}", '
        f'has been placed between joints named ({globals()[first_joint_var_name].name}, '
        f'{globals()[second_joint_var_name].name}), internally stored as '
        f'({first_joint_var_name}, {second_joint_var_name}).')

def create_load(load_name: str, joint_name: str, x: float, y: float, 
                truss: Truss = None, var_name: str = None, print_info=False):
    """
    Create an instance of a load in a truss, with a user defined name load_name,
    stored internally as var_name, at joint string joint_var_name, with components (x, y).
    """

    truss = active_truss if truss is None else truss
    var_name = convert_to_valid_var_name(load_name) if var_name is None else var_name

    for var, val in globals().copy().items():
        if hasattr(val, 'name') and val.name == joint_name:
            joint_var_name = var

    if validate_var_name(var_name):
        globals()[var_name] = truss.Load(load_name, globals()[joint_var_name], x, y)

    if print_info:
        print(f'The load with name "{globals()[var_name].name}", internally stored as "{var_name}", '
        f'has been applied at joint named {globals()[joint_var_name].name}, '
        f'internally stored as "{joint_var_name}", with components ({x}, {y}).')

def create_support(support_name: str, joint_name: str, support_type: str, 
                   roller_normal_vector: tuple = None, pin_rotation: float = 0,
                   truss: Truss = None, var_name: str = None, print_info=False):
    """
    Create an instance of a support in a truss, with a user defined name support_name,
    stored internally as var_name, at joint variable name string joint_var_name.
    """

    truss = active_truss if truss is None else truss
    var_name = convert_to_valid_var_name(support_name) if var_name is None else var_name

    for var, val in globals().copy().items():
        if hasattr(val, 'name') and val.name == joint_name:
            joint_var_name = var
        
    if validate_var_name(var_name):
        globals()[var_name] = truss.Support(truss, support_name,
                                            globals()[joint_var_name], support_type=support_type, 
                                            roller_normal_vector=roller_normal_vector, pin_rotation=pin_rotation)

    if print_info:
        print(f'The support with name "{globals()[var_name].name}", internally stored as "{var_name}", '
        f'has been applied at joint named {globals()[joint_var_name].name}, internally stored as "{joint_var_name}", '
        f'with type "{support_type}" in direction {roller_normal_vector}, with pin rotation {pin_rotation} degrees.')


"""---------------------------------------------------------------------------------------"""
                    #####################################################
                    #           PROGRAM EXECUTION STARTS HERE           #
                    #####################################################
"""---------------------------------------------------------------------------------------"""

import os

#  Fix issue with warning appearing when run from .exe
if os.path.basename(__file__).endswith('.exe'):
    warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*",
                            category=UserWarning)  # deprecation warning inherits from UserWarning

if __name__ == "__main__":

    # -- An example truss - cantilever used in SDC --

    # Define some example bar parameters, four choices of bar
    weak = {"b": 12.5, "t": 0.7, "D": 5, "E": 210, "strength_max": 0.216}
    medium_1 = {"b": 16, "t": 0.9, "D": 5, "E": 210, "strength_max": 0.216}
    medium_2 = {"b": 16, "t": 1.1, "D": 5, "E": 210, "strength_max": 0.216}
    strong = {"b": 19, "t": 1.1, "D": 5, "E": 210, "strength_max": 0.216}

    # Define some custom bar parameters and initialise the truss
    custom_params = weak
    create_truss('My First Truss', bar_params=custom_params, units='kN, mm')

    # Step 1. Create the joints
    create_joint('Joint A', 0, 0)
    create_joint('Joint B', 290, -90)
    create_joint('Joint C', 815, 127.5)
    create_joint('Joint D', 290, 345)
    create_joint('Joint E', 0, 255)
    create_joint('Joint F', 220.836, 127.5)

    # Step 2. Create the bars
    create_bar('Bar AB', 'Joint A', 'Joint B', medium_2)
    create_bar('Bar BC', 'Joint B', 'Joint C', strong)
    create_bar('Bar CD', 'Joint C', 'Joint D', medium_1)
    create_bar('Bar DE', 'Joint D', 'Joint E', medium_1)
    create_bar('Bar EF', 'Joint E', 'Joint F', medium_1)
    create_bar('Bar AF', 'Joint F', 'Joint A', medium_2)
    create_bar('Bar DF', 'Joint F', 'Joint D', medium_1)
    create_bar('Bar BF', 'Joint F', 'Joint B', weak)

    # Step 3. Create the loads
    create_load('W', 'Joint C', 0, -0.675 * 1)

    # Step 4. Create the supports
    create_support('Support A', 'Joint A', support_type='encastre')
    create_support('Support E', 'Joint E', support_type='pin', pin_rotation=90)

    try:  # Get the results of the truss calculation and display graphic
        my_results = active_truss.Result(active_truss, sig_figs=3, solution_method=SolveMethod.NUMPY_STD)
        print(my_results)
    except np.linalg.LinAlgError as e:  # The truss was badly made, so could not be solved
        active_truss.classify_error_in_truss(e)

    plot_diagram(active_truss, my_results, show_reactions=True)
    