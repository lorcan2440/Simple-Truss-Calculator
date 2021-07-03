import Truss_Calculator as t
import numpy as np
import string
import timeit


def build_truss(truss, joints: tuple[tuple], bars: tuple[tuple], loads: tuple[tuple], supports: tuple):

    _alpha = string.ascii_uppercase
    _nums = string.digits

    for i, (x, y) in enumerate(joints):
        t.create_joint(truss, 'joint_' + _alpha[i], 'Joint ' + _alpha[i], x, y)
    for i, (c, bar_type) in enumerate(bars):
        t.create_bar(truss, 'bar_' + _nums[i], 'Bar ' + c, 'joint_' + c[0], 'joint_' + c[1], bar_type)
    for i, (j, x, y) in enumerate(loads):
        t.create_load(truss, 'load_' + j.lower(), r'$W_{' + _nums[i] + r'}$', 'joint_' + j, x, y)
    for i, (j, kwargs) in enumerate(supports):
        t.create_support(truss, 'support_' + j, 'Support ' + j, 'joint_' + j, **kwargs)


def solve_truss(truss, show_outputs=True):

    try:
        my_results = truss.Result(truss, sig_figs=3, solution_method="NUMPY.STANDARD", delete_truss_after=not show_outputs)
        if show_outputs:
            print(my_results)
            t.plot_diagram(truss, my_results, show_reactions=False)
    except np.linalg.LinAlgError as e:
        valid = truss.is_statically_determinate()
        if not valid:
           raise ArithmeticError(f'''The truss is not statically determinate. 
              It cannot be solved. \nBars: {truss.b} \t Reactions: {truss.F} \t Joints: {truss.j}.
              \n b + F = {truss.b + truss.F}, 2j = {2 * truss.j}''')
        elif str(e) == "Singular matrix":
            raise TypeError('''The truss is a mechanism or contains 
                mechanistic components. It cannot be solved.''')
        else:
            raise TypeError("Something else went wrong. Couldn't identify the problem.")


def set_constants(run_test_case):
    def wrapper():
        global custom_params, weak, medium_1, medium_2, strong
        custom_params =     {"b": 12.5, "t": 0.7,   "D": 5,   "E": 210,     "strength_max": 0.216}
        weak =              {"b": 12.5, "t": 0.7,   "D": 5,   "E": 210,     "strength_max": 0.216}
        medium_1 =          {"b": 16,   "t": 0.9,   "D": 5,   "E": 210,     "strength_max": 0.216}
        medium_2 =          {"b": 16,   "t": 1.1,   "D": 5,   "E": 210,     "strength_max": 0.216}
        strong =            {"b": 19,   "t": 1.1,   "D": 5,   "E": 210,     "strength_max": 0.216}
        truss, joints, bars, loads, supports = run_test_case()
        build_truss(truss, joints, bars, loads, supports)
        solve_truss(truss, show_outputs=True)
    return wrapper


@set_constants
def test_case_1():
    joints = ((0, 0), (290, -90), (815, 127.5), (290, 345), (0, 255), (220.836, 127.5))
    bars = (('AB', medium_2), ('BC', strong), ('CD', medium_1), ('DE', medium_1), 
        ('EF', medium_1), ('AF', medium_2), ('DF', medium_1), ('BF', weak))
    loads = [('C', 0, -0.675)]
    supports = (('A', {'support_type': 'encastre'}), ('E', {'support_type': 'pin'}))
    truss = t.Truss(custom_params, 'kN, mm')
    return truss, joints, bars, loads, supports

@set_constants
def test_case_2():
    joints = ((0, 0), (5, 0), (0, 4), (2, 6))
    bars = (('AC', strong), ('BC', strong), ('CD', strong), ('BD', strong))
    loads = [('C', 1000, 0), ('D', 0, -750)]
    supports = (('A', {'support_type': 'pin'}), ('B', {'support_type': 'pin'}))
    truss = t.Truss(custom_params, 'N, m')
    return truss, joints, bars, loads, supports

@set_constants
def test_case_3():
    joints = ((0, 1.5), (1, 0), (1, 1.5))
    bars = (('AB', strong), ('BC', strong), ('AC', strong))
    loads = [('C', 1, 2)]
    supports = (('A', 'pin', None), ('B', 'roller', (1, 1)))
    supports = (('A', {'support_type': 'pin', 'pin_rotation': 90}), 
                ('B', {'support_type': 'roller', 'roller_normal_vector': (-1, 1)})
                )
    truss = t.Truss(custom_params, 'N, m')
    return truss, joints, bars, loads, supports

'''
time = timeit.timeit(test_case_1, number=1)
print(time)
'''


test_case_1()

test_case_2()

test_case_3()
