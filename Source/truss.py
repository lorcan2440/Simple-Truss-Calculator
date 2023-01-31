# pyright: reportMissingModuleSource=false

"""
Simple Truss Calculator

Version:    1.5
Source:     https://github.com/lorcan2440/Simple-Truss-Calculator
By:         Lorcan Nicholls
Contact:    lnick2440@gmail.com
Tests:      test_TrussCalc.py

Calculator and interactive program for finding internal/reaction forces,
stresses and strains of a pin-jointed, straight-membered, plane truss.
Intended for personal use only; documented but not in a module form.
Soon I hope to make it more user-friendly and interactive.
"""

# builtin modules
from typing import Optional
from enum import Enum
import warnings
import math
import json
import re
import os

import utils

# auto install missing modules, least likely to be already installed first
import sigfig                                                   # for rounding values nicely
from scipy.sparse import (csr_matrix, linalg as linsolver)      # for faster solving
from matplotlib import pyplot as plt                            # to display graphical output
import numpy as np                                              # to do matrix operations


# PARTS OF THE TRUSS (INNER CLASSES)

class Joint:

    """
    Joints define the locations where other objects can go.
    Bars go between two joints.
    Each joint can have loads and supports applied.
    """

    def __init__(self, truss: object, name: str, x: float, y: float):

        self.truss = truss
        self.name = name
        self.x = x
        self.y = y
        self.loads = {}


class Bar:

    """
    Bars go between the `first_joint` and the `second_joint`.
    Each bar can have different thickness, strength, etc in `my_params`.
    """

    def __init__(self, truss: object, name: str, first_joint: object, second_joint: object,
            my_params: Optional[dict] = None):

        """
        Initialise a bar with a given name, which joints it should connect between, and
        its physical properties.
        """

        self.name = name
        self.truss = truss

        self.first_joint, self.first_joint_name = first_joint, first_joint.name
        self.second_joint, self.second_joint_name = second_joint, second_joint.name

        self.params = truss.default_params if my_params is None else my_params

        # physical and geometric properties of the bar, as defined on databook pg. 8
        [setattr(self, attr, self.params.get(attr, None)) for attr in ["b", "t", "D", "E", "strength_max"]]
        self.length = math.sqrt((self.first_joint.x - self.second_joint.x)**2 +  # noqa \
            (self.first_joint.y - self.second_joint.y)**2)
        self.section_area = (self.b ** 2 - (self.b - self.t) ** 2) * 1.03
        self.effective_area = (1.5 * self.b - self.D) * 0.9 * self.t
        self.buckling_ratio = self.length / self.b

    def get_direction(self, origin_joint: Optional[object] = None, as_degrees: bool = False) -> float:

        """
        Calculates the (polar) angle this bar makes with the horizontal,
        with the origin taken as the origin_joint. 0 = horizontal right, +pi/2 = vertical up,
        -pi/2 = vertical down, pi = horizontal left, etc. (-pi < angle <= pi).
        """

        if origin_joint in (connected_joints := self.truss.get_all_joints_connected_to_bar(self)):
            other_joint_index = 1 - connected_joints.index(origin_joint)
            angle = math.atan2(connected_joints[other_joint_index].y - origin_joint.y,
                connected_joints[other_joint_index].x - origin_joint.x)

        elif origin_joint is None:
            # if no joint is specified, the joint is chosen such that the angle
            # is not upside-down (used to allign the text along the bars)
            angle_from_first = self.get_direction(self.first_joint, as_degrees=as_degrees)
            if as_degrees and -90 < angle_from_first <= 90 or \
                    not as_degrees and -1 * math.pi / 2 < angle_from_first <= math.pi / 2:
                return angle_from_first

            else:
                return self.get_direction(self.second_joint, as_degrees=as_degrees)

        else:
            raise SyntaxError(f'The bar "{self.name}" has an invalid origin joint when finding \n'
                f'its direction. It should be the objects associated with either \n'
                f'{self.first_joint_name} or {self.second_joint_name}.')

        return angle if not as_degrees else math.degrees(angle)


class Load:

    """
    Loads can be applied at any joint.
    Their components are specified as (x_comp, y_comp)
    aligned with the coordinate system used to define the joints.
    """

    def __init__(self, name: str, joint: object, x_comp: float = 0.0, y_comp: float = 0.0):

        """
        Initialise a load with a name, a joint to be applied at, and a force value in terms
        of x and y components (as defined by the coordinate and unit systems).
        """

        self.name = name
        self.joint = joint
        self.x, self.y = x_comp, y_comp

        self.magnitude = math.sqrt(self.x ** 2 + self.y ** 2)
        self.direction = math.atan2(self.y, self.x)
        joint.loads[self.name] = (self.x, self.y)


class Support:

    """
    Supports are points from which external reaction forces can be applied.
    """

    def __init__(self, name: str, joint: object, support_type: str = 'pin', pin_rotation: float = 0):

        """
        Initialise a support with a name, a joint object to convert to a support, the type of support
        and a direction if a roller joint is chosen.

        support_type:   can be 'pin' or 'roller' or 'encastre'
        pin_rotation:   angle of 'ground' direction for support, in radians, clockwise from the x-axis
        """

        self.name = name
        self.joint = joint
        self.support_type = support_type
        self.pin_rotation = pin_rotation
        self.normal_direction = pin_rotation + math.pi / 2

        self.roller_normal = [math.cos(self.pin_rotation + math.pi / 2),
            math.sin(self.pin_rotation + math.pi / 2)]
        self.roller_normal = np.array(self.roller_normal) / np.linalg.norm(self.roller_normal)

        if self.support_type in {'encastre', 'pin', 'roller'}:
            joint.loads[f'Reaction @ {self.name}'] = (None, None)
        else:
            raise ValueError('Support type must be "encastre", "pin" or "roller".')


# TRUSS RESULTS CLASS

class Result:

    """
    Allows the results to be analysed and manipulated.
    """

    def __init__(self, truss: object, sig_figs: Optional[int] = 4,
            solution_method: Enum = utils.SolveMethod.NUMPY_SOLVE,
            _override_res: Optional[tuple[dict]] = None):

        self.truss = truss
        self.sig_figs = sig_figs

        warnings.filterwarnings('ignore')

        if _override_res is None:
            self.results = truss.solve(solution_method=solution_method)
            self.tensions, self.reactions, self.stresses, self.strains = {}, {}, {}, {}
            self.buckling_ratios = {}

            # populate the tensions, reactions, etc. dictionaries from the results
            self.get_data(truss)

        else:
            self.tensions, self.reactions, self.stresses, self.strains, self.buckling_ratios = \
                (*_override_res,)

        # set the truss's results before rounding but after zeroing small numbers
        self.truss.results = {'internal_forces': self.tensions.copy(),
            'reaction_forces': self.reactions.copy(),
            'stresses': self.stresses.copy(), 'strains': self.strains.copy(),
            'buckling_ratios': self.buckling_ratios.copy()}

        # round these results to the required precision
        self.round_data()

    def __repr__(self):
        repr_str = f'\n Axial forces are: '\
            f'(positive = tension; negative = compression) \n \t {str(self.tensions)}'
        repr_str += f'\n Axial stresses are: \n \t {str(self.stresses)}'
        repr_str += f'\n Reaction forces are (horizontal, vertical) components (signs '\
            f'consistent with coordinate system): \n \t {str(self.reactions)}'
        repr_str += f'\n Buckling ratios are: \n \t {str(self.buckling_ratios)}'
        repr_str += f'\n Strains are: \n \t {str(self.strains)}'
        repr_str += f'\n\n Units are {self.truss.units.split(",")[0]}, values '\
            f'{f"not rounded" if self.sig_figs is None else f"rounded to {self.sig_figs} s.f."}'
        return repr_str

    def round_data(self) -> None:
        """
        Replaces the calculated data with rounded values, to precision given by Result.sig_figs.
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

    def get_data(self, truss: object) -> None:
        """
        Calculate tensions, stresses, strains, reaction forces and buckling ratios
        from the calculate() function.
        """

        # any forces smaller than `SMALL_NUM` will be set to zero (assumed to be due to rounding
        # errors in the solver function). Currently set to 10 times smaller than the least
        # significant digit of the smallest internal force value.
        # NOTE: maybe move this functionality into `round_data()`.
        try:
            SMALL_NUM = 0.1 * 10 ** (-1 * self.sig_figs) * min(
                [abs(f) for f in self.results.values()
                if type(f) is not tuple and f > (0.1 * 10 ** (-1 * self.sig_figs))])
        except ValueError:  # triggered if all forces are zero
            SMALL_NUM = 1e-8

        for item in self.results:

            if isinstance(self.results[item], float):

                if abs(self.results[item]) < SMALL_NUM:
                    self.tensions.update({item: 0})
                else:
                    self.tensions.update({item: self.results[item]})

                self.stresses.update({
                    item: self.tensions[item] / truss.bars[item].effective_area})
                self.strains.update({item: self.stresses[item] / truss.bars[item].E})
                self.buckling_ratios.update({item: truss.bars[item].buckling_ratio})
                # NOTE: could check if the bar is in compression using: if self.results[item] < 0:

            elif isinstance(self.results[item], tuple):

                self.reactions.update({item: (
                    self.results[item][0] if abs(self.results[item][0]) > SMALL_NUM else 0,
                    self.results[item][1] if abs(self.results[item][1]) > SMALL_NUM else 0)})

            else:
                warnings.warn(f'''A result appears to have been formed incorrectly. This is an internal
                    error. Bad value ignored: {self.results[item]}''', RuntimeWarning)
                continue


# MAIN CLASS FOR TRUSSES

class Truss:
    """
    A class containing the truss to be worked with.
    """

    # some default values. symbols defined on databook pg. 8
    DEFAULT_BAR_PARAMS = {"b": 0.016, "t": 0.004, "D": 0.020,
        "E": 2.1e11, "strength_max": 3e9}  # in (N, m)

    def __init__(self, **kwargs):
        '''
        Initialise a truss.

        #### Arguments

        `name` (str, default = "My Truss"): a name for the truss. Not used internally,
        but appears on the plot.
        `bar_params` (Optional[dict], default = None): bar properties will inherit from
        this if they are not set when building.
        `units` (tuple[Unit], default = (Unit.KILONEWTONS, Unit.MILLIMETRES)): unit system
        to use in calculations (Force, Length).

        #### Raises

        `ValueError`: _description_
        '''

        self.name = kwargs.get('name')
        self.units = kwargs.get('units')

        self.joints = dict()
        self.bars = dict()
        self.loads = dict()
        self.supports = dict()

        if kwargs.get('bar_params', None) is None:
            # some default values. symbols defined on databook pg. 8
            self.default_params = Truss.DEFAULT_BAR_PARAMS
            if self.units[0] is utils.Unit.KILONEWTONS:
                self.default_params['E'] *= 1e-3
            if self.units[1] is utils.Unit.MILLIMETRES:
                self.default_params['b'] *= 1e3
                self.default_params['t'] *= 1e3
                self.default_params['D'] *= 1e3
                self.default_params['E'] *= 1e-6
                self.default_params['strength_max'] *= 1e-6
        else:
            self.default_params = kwargs.get('bar_params')

    # object builders

    def add_joints(self, list_of_joints: list[dict], replace_if_same_name: bool = True):

        '''
        Params to be specified:
        name (str)
        x (float)
        y (float)
        '''

        curr_joint_names = self.get_all_objs(self.joints, str_names_only=True)

        for kwargs in list_of_joints:
            joint_name = kwargs.get('name', None)
            x, y = kwargs.get('x'), kwargs.get('y')  # required

            new_joint_name = utils.convert_to_valid_var_name(
                joint_name, self.joints, replace_if_same_name)

            if joint_name in curr_joint_names and not replace_if_same_name:
                warnings.warn(f'The joint {joint_name} was renamed to {new_joint_name}'
                f'because it already exists in the truss and `replace_if_same_name`'
                'was set to `False`.')

            self.joints[new_joint_name] = Joint(self, new_joint_name, x, y)
            curr_joint_names.add(new_joint_name)

        return self

    def add_bars(self, list_of_bars: list[dict], replace_if_same_name: bool = True):

        '''
        Params to be specified:
        name (str)
        first_joint_name (str)
        second_joint_name (str)
        bar_params (dict)
        '''

        curr_bar_names = self.get_all_objs(self.bars, str_names_only=True)
        curr_joint_names = self.get_all_objs(self.joints, str_names_only=True)

        for kwargs in list_of_bars:
            bar_name = kwargs.get('name', None)
            first_joint_name = kwargs.get('first_joint_name', None)
            second_joint_name = kwargs.get('second_joint_name', None)
            bar_params = kwargs.get('bar_params', None)

            if all([s in curr_joint_names or s is None for s in (first_joint_name, second_joint_name)]):
                if bar_name:
                    if first_joint_name is None:
                        if len(bar_name) == 2:
                            first_joint_name = bar_name[0]
                        else:
                            raise ValueError(f'The bar {bar_name} was not given a first joint name, '
                            'and it could not be inferred from its name.')
                    if second_joint_name is None:
                        if len(bar_name) == 2:
                            second_joint_name = bar_name[1]
                        else:
                            raise ValueError(f'The bar {bar_name} was not given a second joint name, '
                            'and it could not be inferred from its name.')
                else:
                    bar_name = f'{first_joint_name}{second_joint_name}'
            else:
                raise ValueError(f'The bar {bar_name} was requested connected to joints named '
                f'{first_joint_name} and {second_joint_name}, but the joint '
                f'{first_joint_name if (first_joint_name not in curr_joint_names) else second_joint_name} '
                f'does not exist in the truss. The available joint names are: {curr_joint_names}.')

            new_bar_name = utils.convert_to_valid_var_name(
                bar_name, self.bars, replace_if_same_name)

            if bar_name in curr_bar_names and not replace_if_same_name:
                warnings.warn(f'The bar {kwargs.get("name")} was renamed to {bar_name}'
                f'because it already exists in the truss and `replace_if_same_name`'
                'was set to `False`.')

            joint_1 = self.joints[first_joint_name]
            joint_2 = self.joints[second_joint_name]
            self.bars[new_bar_name] = Bar(self, new_bar_name, joint_1, joint_2, bar_params)
            curr_bar_names.add(new_bar_name)

        return self

    def add_loads(self, list_of_loads: list[dict], replace_if_same_name: bool = True):

        '''
        Params to be specified:
        name (str)
        joint_name (str)
        x (float)
        y (float)
        '''

        curr_joint_names = self.get_all_objs(self.joints, str_names_only=True)
        curr_load_names = self.get_all_objs(self.loads, str_names_only=True)

        for kwargs in list_of_loads:
            load_name = kwargs.get('name', None)
            joint_name = kwargs.get('joint_name', None)
            x, y = kwargs.get('x'), kwargs.get('y')  # required

            if joint_name not in curr_joint_names:
                raise ValueError(f'The load {load_name} was requested at a joint named '
                f'{joint_name} but a joint with this name does not exist in the truss. '
                f'The available joint names are: {curr_joint_names}.')

            new_load_name = utils.convert_to_valid_var_name(
                load_name, self.loads, replace_if_same_name)

            if load_name in curr_load_names and not replace_if_same_name:
                warnings.warn(f'The load {load_name} was renamed to {new_load_name} '
                f'because it already exists in the truss and `replace_if_same_name` '
                'was set to `False`.')

            joint_obj = self.joints[joint_name]
            self.loads[new_load_name] = Load(new_load_name, joint_obj, x, y)
            curr_load_names.add(new_load_name)

        return self

    def add_supports(self, list_of_supports: list[dict], replace_if_same_name: bool = True):

        '''
        Params to be specified:
        name (str)
        joint_name (str)
        support_type (str)
        pin_rotation (float)
        '''

        curr_support_names = self.get_all_objs(self.supports, str_names_only=True)
        curr_joint_names = self.get_all_objs(self.joints, str_names_only=True)

        for kwargs in list_of_supports:

            support_name = kwargs.get('name', None)
            joint_name = kwargs.get('joint_name', None)
            support_type = kwargs.get('support_type', None)
            pin_rotation = kwargs.get('pin_rotation', None)

            if support_name is None and joint_name is not None:
                joint_name = support_name
            elif support_name is not None and joint_name is None:
                support_name = joint_name
            elif support_name is None and joint_name is None:
                raise ValueError('Neither a support name nor a joint at which it is located '
                'was given. At least one must be given, and the name of the other is inferred.')

            if joint_name not in curr_joint_names:
                raise ValueError(f'The support {support_name} was requested at a joint named '
                f'{joint_name} but a joint with this name does not exist in the truss. '
                f'The available joint names are: {curr_joint_names}.')

            pin_rotation = pin_rotation or 0
            support_type = support_type or 'pin'

            new_support_name = utils.convert_to_valid_var_name(
                support_name, self.supports, replace_if_same_name)

            if support_name in curr_support_names and not replace_if_same_name:
                warnings.warn(f'The support {support_name} was renamed to {new_support_name} '
                f'because it already exists in the truss and `replace_if_same_name` '
                'was set to `False`.')

            joint_obj = self.joints[joint_name]
            self.supports[new_support_name] = Support(new_support_name, joint_obj,
                support_type, pin_rotation)
            curr_support_names.add(new_support_name)

    # TRUSS METHODS

    def solve(self, solution_method: Enum = utils.SolveMethod.SCIPY) -> dict[str, float | tuple]:

        """
        The main part of the program. Calculates the forces in the truss's bars and supports
        in order to maintain force equilibrium with the given loads. Outputs as a dictionary in the form
        `{bar_name: axial_force_value} + {support_name: (reaction_force_value_x, reaction_force_value_y)}`
        """

        all_bars = self.get_all_objs(self.bars)
        all_joints = self.get_all_objs(self.joints)
        all_supports = self.get_all_objs(self.supports)

        # List of dictionaries for unknowns, given default zero values
        wanted_vars = []
        for bar in all_bars:
            wanted_vars.append('Tension in ' + bar.name)
        for support in all_supports:
            if support.support_type in {'pin', 'encastre'}:
                wanted_vars.append('Horizontal reaction at ' + support.joint.name)
                wanted_vars.append('Vertical reaction at ' + support.joint.name)
            elif support.support_type == 'roller':
                wanted_vars.append('Magnitude of reaction at ' + support.joint.name)
            else:
                continue

        all_directions = {}
        for joint in all_joints:
            # Reset the directions dictionary for this joint
            directions = {}

            # Get the anticlockwise (polar) angle of each connected joint relative to this joint which have bars
            for bar in self.get_all_bars_connected_to_joint(joint):
                angle = bar.get_direction(joint)
                directions['Tension in ' + bar.name] = angle

            # If there are reactions at this joint, store their directions too
            if any([s.joint.name == joint.name for s in all_supports]):
                if self.get_support_by_joint(joint).support_type == 'roller':
                    directions['Magnitude of reaction at ' + joint.name] = \
                        self.get_support_by_joint(joint).normal_direction
                else:
                    directions['Horizontal reaction at ' + joint.name] = 0
                    directions['Vertical reaction at ' + joint.name] = math.pi / 2

            # If there are external loads at this joint, store their directions too
            for load in self.get_all_loads_at_joint(joint):
                directions['Horizontal component of {} at {}'.format(load.name, joint.name)] = 0
                directions['Vertical component of {} at {}'.format(load.name, joint.name)] = math.pi / 2

            all_directions[joint.name] = directions

        # Populate the coefficients and constants matrices (initially lists of lists)
        # in preparation to solve the matrix equation M * x = B
        coefficients, constants = [], []
        for joint in all_joints:

            joint_name = joint.name
            support = self.get_support_by_joint(joint)

            if getattr(support, 'support_type', None) != 'roller':
                # pin joint or pin/encastre support, resolve in x and y

                # get the coefficients (matrix M), representing the unknown internal/reaction forces
                current_line = [round(math.cos(all_directions[joint_name].get(
                    var, math.pi / 2)), 10) for var in wanted_vars]
                coefficients.append(current_line)
                current_line = [round(math.sin(all_directions[joint_name].get(
                    var, 0)), 10) for var in wanted_vars]
                coefficients.append(current_line)

                # get the constants (vector B), representing the external loads, -ve since on other side of eqn
                loads_here = self.get_all_loads_at_joint_by_name(joint_name)
                constants.append([-1 * sum([load.x for load in loads_here])])
                constants.append([-1 * sum([load.y for load in loads_here])])

            else:  # roller support, resolve parallel and perpendicular to roller normal

                # get the coefficients (matrix M), representing the unknown internal/reaction forces
                current_line = [round(math.cos(all_directions[joint_name].get(
                    var, support.normal_direction) - support.normal_direction), 10) for var in wanted_vars]
                coefficients.append(current_line)
                current_line = [round(math.sin(all_directions[joint_name].get(
                    var, support.normal_direction + math.pi / 2) - support.normal_direction),
                    10) for var in wanted_vars]
                coefficients.append(current_line)

                # get the constants (vector B), representing the external loads, -ve since on other side of eqn
                sum_parallel, sum_perp = 0, 0
                for load in self.get_all_loads_at_joint_by_name(joint_name):
                    sum_parallel += np.dot([load.x, load.y], support.roller_normal)
                    tangent_vector = [-1 * math.cos(support.pin_rotation), -1 * math.sin(support.pin_rotation)]
                    sum_perp += np.dot([load.x, load.y], tangent_vector)

                constants.append([-1 * sum_parallel])
                constants.append([-1 * sum_perp])

        # Sanitise load data
        for i in range(len(constants)):
            if constants[i] == [] or constants[i] == [None]:
                constants[i] = [0]

        # Solve the system - both coefficient and constant matrices are sparse (for most practical cases)
        # so ideally the SCIPY method is faster. NOTE: However testing showed that the difference is not huge,
        # possibly because the solution itself is not sparse.
        if solution_method is utils.SolveMethod.NUMPY_STD:
            m, b = np.matrix(coefficients), np.matrix(constants)
            x = np.linalg.inv(m) * b

        elif solution_method is utils.SolveMethod.NUMPY_SOLVE:
            m, b = np.matrix(coefficients), np.matrix(constants)
            x = np.linalg.solve(m, b)

        elif solution_method is utils.SolveMethod.SCIPY:
            m, b = csr_matrix(coefficients), csr_matrix(constants)
            x = linsolver.spsolve(m, b)

        else:
            raise SyntaxError(f"The solution method {solution_method} is not supported. \n"
                              f"The allowed methods are (either using constants or string literals): \n"
                              f"{utils.get_constants(utils.SolveMethod)}\n"
                              f"For example: \t solution_method=SolveMethod.NUMPY_SOLVE \t or \t"
                              f"solution_method='numpy_solve'")

        # Map solution back to each object name

        print(wanted_vars)

        output_dict = {}
        for i, (var_name, val) in enumerate(zip(wanted_vars, x)):
            if 'Tension in ' in var_name:
                bar_name = var_name.split('Tension in ')[-1]
                output_dict[bar_name] = float(val)
            elif 'reaction' in var_name:
                support_name = self.get_support_by_joint(
                    self.joints[var_name.split('reaction at ')[-1]], str_names_only=True)
                if support_name not in output_dict:
                    if 'magnitude' in var_name:
                        output_dict[support_name] = float(val)
                    else:
                        output_dict[support_name] = (float(val), float(x[i + 1]))
                else:
                    continue

        # Return the values in dict form
        return output_dict

    def is_statically_determinate(self) -> bool:

        """
        Does a simple arithmetic check to estimate if the truss
        is statically determinate (b + F = 2j). Also stores attributes for later quick use.
        """

        # b: number of bars in the truss
        # F: number of degrees of freedom for the reactions at the supports
        # j: number of joints in the truss
        # if b + F > 2j, the truss is overconstrained, while if b + F < 2j, the truss is a mechanism
        self.b = len(self.get_all_objs(self.bars, str_names_only=True))
        self.F = sum([2 if support.support_type in {'encastre', 'pin'}
                 else 1 if support.support_type == 'roller'
                 else 0 for support in self.get_all_objs(self.supports)])
        self.j = len(self.get_all_objs(self.joints, str_names_only=True))

        return self.b + self.F == 2 * self.j

    def classify_error_in_truss(self, e: np.linalg.LinAlgError) -> None:

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
            being globally statically determinate. It cannot be solved without compatibility.''')

        else:
            raise TypeError("Something else went wrong. Requires attention.")

    def dump_truss_to_json(self, filedir: Optional[str] = None, filename: Optional[str] = None) -> None:

        """
        Writes the details of the truss, with the results if available, to
        a JSON file which can be read using `load_truss_from_json()`.

        NOTE: If this truss is deleted before this function is called, only the results
        will be available.
        """

        # create the output directory if specified and it does not already exist
        if filedir is not None:
            if not os.path.exists(filedir):
                os.mkdir(filedir)

        # get filename
        new_name = self.name.strip().lower().replace(' ', '_')
        pattern = re.compile(r'[\W]+', re.UNICODE)
        new_name = pattern.sub('', new_name)

        # set the file name as the truss's var_name
        out_file_dir = os.path.join('' if filedir is None else filedir,
                                    (new_name + '.json') if filename is None else filename)

        # fill out the dictionary, using dict.get() where values may be unavailable (will appear as nulls)
        json_dict = {
            'truss': {
                'name': self.name,
                'default_bar_params': {'b': self.default_params.get('b'), 't': self.default_params.get('t'),
                                       'D': self.default_params.get('D'), 'E': self.default_params.get('E'),
                                       'strength_max': self.default_params.get('strength_max')},
                'units': self.units
            },
            'joints': [
                {'name': j.name, 'x': j.x, 'y': j.y} for j in self.get_all_objs(self.joints)
            ],
            'bars': [
                {'name': b.name, 'first_joint_name': b.first_joint_name,
                 'second_joint_name': b.second_joint_name,
                 'bar_params': {'b': b.params.get('b'), 't': b.params.get('t'),
                                'D': b.params.get('D'), 'E': b.params.get('E'),
                                'strength_max': b.params.get('strength_max')}
                 } for b in self.get_all_objs(self.bars)
            ],
            'loads': [
                {'name': load.name, 'joint_name': load.joint.name,
                 'x': load.x, 'y': load.y} for load in self.get_all_objs(self.loads)
            ],
            'supports': [
                {'name': s.name, 'joint_name': s.joint.name,
                 'support_type': s.support_type,
                 'roller_normal': tuple(s.roller_normal) if s.roller_normal is not None else None,
                 'pin_rotation': s.pin_rotation} for s in self.get_all_objs(self.supports)
            ],
            'results': {
                'internal_forces': self.results.get('internal_forces'),
                'reaction_forces': self.results.get('reaction_forces'),
                'stresses': self.results.get('stresses'),
                'strains': self.results.get('strains'),
                'buckling_ratios': self.results.get('buckling_ratios')
            } if hasattr(self, 'results') else None,
        }

        # write to the chosen JSON file location
        with open(out_file_dir, 'w') as f:
            json.dump(json_dict, f, indent=4)

    # object and name getters

    @staticmethod
    def get_all_objs(data_dict, str_names_only: bool = False) -> list[Bar] | set[str]:
        """
        Returns a list of objects or set of string names in this truss of a given type.
        """
        if str_names_only:
            return set(data_dict.keys())
        else:
            return list(data_dict.values())

    @staticmethod
    def get_all_bars_connected_to_joint(joint: Joint, str_names_only: bool = False) -> list[Bar] | set[str]:
        """
        Returns a list of bar objects or names which are connected to a given joint object.
        """
        if str_names_only:
            return {bar.name for bar in joint.truss.get_all_objs(joint.truss.bars)
                if joint.name in {bar.first_joint.name, bar.second_joint.name}}
        else:
            return [bar for bar in joint.truss.get_all_objs(joint.truss.bars)
                if joint.name in {bar.first_joint.name, bar.second_joint.name}]

    @staticmethod
    def get_all_joints_connected_to_bar(bar: Bar, str_names_only: bool = False) -> tuple[Joint] | tuple[str]:
        """
        Returns a list of joint objects or names which are connected to a given bar object.
        The order is arbitrary but consistent.
        """
        if str_names_only:
            return (bar.first_joint.name, bar.second_joint.name)
        else:
            return (bar.first_joint, bar.second_joint)

    @staticmethod
    def get_all_loads_at_joint(joint: Joint, str_names_only: bool = False) -> list[Load] | set[str]:

        """
        Returns a list of load objects which are applied at a given joint object.
        """

        if str_names_only:
            return {load.name for load in joint.truss.get_all_objs(joint.truss.loads)
                if load.joint is joint}
        else:
            return [load for load in joint.truss.get_all_objs(joint.truss.loads)
                if load.joint is joint]

    def get_all_loads_at_joint_by_name(self, joint_name: str,
            str_names_only: bool = False) -> list[Load] | set[str]:
        """
        Returns a list of load objects which are applied at a given joint name.
        """
        if str_names_only:
            return {load.name for load in self.get_all_objs(self.loads)
                if load.joint.name == joint_name}
        else:
            return [load for load in self.get_all_objs(self.loads)
                if load.joint.name == joint_name]

    @staticmethod
    def get_support_by_joint(joint: Joint, only_return_one: bool = True,
            str_names_only: bool = False) -> Support | str | list[Support] | set[str]:
        """
        Returns the support object placed at a given joint, or None if there is no support there.
        FIXME: if there are multiple supports, returns only the first one, which may be inconsistent.
        """
        if str_names_only:
            _supports = {support.name for support in joint.truss.get_all_objs(joint.truss.supports)
                if support.joint is joint}
        else:
            _supports = [support for support in joint.truss.get_all_objs(joint.truss.supports)
                if support.joint is joint]

        if _supports:
            if only_return_one:
                (_supports,) = _supports
            return _supports
        else:
            return None


# Classes end here, main program functions start here

def plot_diagram(truss: Truss, results: Result, show_reactions: bool = True) -> None:

    """
    Create a matplotlib output image showing the truss geometry, annotated with arrows, labels and supports.
    """

    # Find a suitable length-scale to make the annotations look nicer.
    # All drawing dimensions are relative to this. As a rough value, this is 10% of the average bar length.
    LEN = np.average([b.length for b in truss.get_all_objs(truss.bars)]) * 0.1

    # Plot all joints without supports
    _xjl, _yjl = map(list, zip(*[(joint.x, joint.y) for joint in truss.get_all_objs(truss.joints)
                                 if truss.get_support_by_joint(joint) is None]))

    plt.plot(_xjl, _yjl, 'o', color='black', markersize=5)
    plt.plot(_xjl, _yjl, 'o', color='white', markersize=3.5)  # small circle with white centre

    # Plot all bars
    for bar in truss.get_all_objs(truss.bars):

        rot = bar.get_direction(as_degrees=True)
        norm = math.radians(rot + 90)

        # connect the two joints with a line
        plt.plot([bar.first_joint.x, bar.second_joint.x], [bar.first_joint.y, bar.second_joint.y],
                 label=bar.name + ': ' + str(results.tensions[bar.name]) + ' ' + truss.units.split(',')[0],
                 zorder=0)

        # label the bar with its name
        plt.text((bar.first_joint.x + bar.second_joint.x) / 2 + LEN / 3 * math.cos(norm),
                 (bar.first_joint.y + bar.second_joint.y) / 2 + LEN / 3 * math.sin(norm),
                 bar.name, ha='center', va='center', rotation=rot, rotation_mode='anchor',
                 transform_rotates_text=True)

    # Plot all supports
    for support in truss.get_all_objs(truss.supports):

        plt.plot(support.joint.x, support.joint.y, '*', markersize=0,
                 label=support.name + ': ' + str(results.reactions[support.name]) + ' ' +  # noqa \
                 truss.units.split(',')[0])

    for support in truss.get_all_objs(truss.supports):
        if show_reactions:
            reaction_direction = math.atan2(*reversed(list(results.reactions[support.name])))

            # draw an arrow of fixed length to show the direction of the reaction
            plt.arrow(support.joint.x, support.joint.y,
                      LEN * math.cos(reaction_direction),
                      LEN * math.sin(reaction_direction),
                      head_width=LEN / 5, head_length=LEN / 4, facecolor='red')

        # TODO: if there is another support at this `support.joint`,
        # label it at an angle of `180 + pin_rotation` instead
        label_angle = utils.find_free_space_around_joint(
            support.joint, results, truss=truss, show_reactions=show_reactions)
        plt.text(support.joint.x + 0.9 * LEN * math.cos(label_angle),
                 support.joint.y + 0.9 * LEN * math.sin(label_angle),
                 support.name, va='center', ha='left' if -90 < math.degrees(label_angle) <= 90 else 'right',
                 label=f'{support.name}: {str(results.reactions[support.name])} {truss.units.split(",")[0]}')

        # draw a icon-like symbol representing the type of support
        # TODO: maybe make this into a matplotlib patch to use it in the legend
        utils.draw_support(support.joint.x, support.joint.y, LEN * 0.9,
                     support_type=support.support_type, roller_normal=support.roller_normal,
                     pin_rotation=support.pin_rotation)

    # Plot all loads
    for load in truss.get_all_objs(truss.loads):

        # draw an arrow of fixed length to show the direction of the load force
        plt.arrow(load.joint.x, load.joint.y, LEN * math.cos(load.direction), LEN * math.sin(load.direction),
                  head_width=LEN / 5, head_length=LEN / 4)

        # TODO: if there is another load at this `load.joint`, label it at the arrow midpoint + normal a bit
        label_angle = utils.find_free_space_around_joint(load.joint, results=results, truss=truss)
        plt.text(load.joint.x + LEN / 3 * math.cos(label_angle),
                 load.joint.y + LEN / 3 * math.sin(label_angle),
                 f'{load.name}: ({str(load.x)}, {str(load.y)}) {truss.units.split(",")[0]}',
                 va='center', ha='left' if -90 < math.degrees(label_angle) <= 90 else 'right')

    # Graphical improvements
    AXES_COLOUR = '#BBBBBB'  # light grey

    plt.title(truss.name)
    plt.legend(loc='upper right')
    plt.autoscale()
    plt.axis('equal')
    plt.xlabel(f'$x$-position / {truss.units.split(",")[1]}')
    plt.ylabel(f'$y$-position / {truss.units.split(",")[1]}')
    plt.style.use('./proplot_style.mplstyle')

    ax = plt.gca()
    spines = ax.spines
    spines['right'].set_visible(False)  # make upper-right spines disappear
    spines['top'].set_visible(False)
    spines['left'].set_color(AXES_COLOUR)  # axis lines
    spines['bottom'].set_color(AXES_COLOUR)
    ax.tick_params(axis='x', colors=AXES_COLOUR, grid_alpha=0.5)  # axis ticks and their number labels
    ax.tick_params(axis='y', colors=AXES_COLOUR, grid_alpha=0.5)
    ax.xaxis.label.set_color(AXES_COLOUR)  # axis name labels
    ax.yaxis.label.set_color(AXES_COLOUR)

    utils.set_matplotlib_fullscreen()
    plt.show()


def load_truss_from_json(file: str, show_if_results: bool = True,
        _delete_truss_after: Optional[bool] = False) -> Truss:

    """
    Builds a truss from a JSON file provided by `dump_truss_to_json()`.
    If the results are available, they can be showed.
    """

    import json

    with open(file) as json_file:

        f = json.load(json_file)

        t_attr = f['truss']
        truss = init_truss(t_attr['name'], t_attr['default_bar_params'], t_attr['units'])

        truss.add_joints(f['joints'])
        truss.add_bars(f['bars'])
        truss.add_loads(f['loads'])
        truss.add_supports(f['supports'])

        if show_if_results and (res := f['results']) is not None:

            bar_names = truss.get_all_objs(truss.bars, str_names_only=True)
            support_names = truss.get_all_objs(truss.supports, str_names_only=True)

            truss_results = Result(truss, sig_figs=3, solution_method=None,
                _override_res=(
                    {bn: res['internal_forces'][bn] for bn in bar_names},
                    {sn: res['reaction_forces'][sn] for sn in support_names},
                    {bn: res['stresses'][bn] for bn in bar_names},
                    {bn: res['strains'][bn] for bn in bar_names},
                    {bn: res['buckling_ratios'][bn] for bn in bar_names}
                )
            )

            print(truss_results)
            plot_diagram(truss, truss_results, show_reactions=True,
                         _delete_truss_after=_delete_truss_after)

        return truss


def init_truss(truss_name: str = None, bar_params: dict = None, units: str = None, **kwargs) -> Truss:

    truss_name = truss_name or 'My Truss'
    bar_params = bar_params
    units = units or (utils.Unit.KILONEWTONS, utils.Unit.MILLIMETRES)

    return Truss(name=truss_name, bar_params=bar_params, units=units, **kwargs)
