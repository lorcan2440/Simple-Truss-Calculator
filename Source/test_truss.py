from truss import Result, init_truss, plot_diagram

import utils

import math


def test_build_standard():

    medium_2 = {"b": 16, "t": 1.1, "D": 5, "E": 210, "strength_max": 0.216}
    custom_params = medium_2

    truss = init_truss('SDC: Steel Cantilever', custom_params, 'kN, mm')

    truss.add_joints([
        {'name': 'A', 'x': 0, 'y': 0},
        {'name': 'B', 'x': 290, 'y': -90},
        {'name': 'C', 'x': 815, 'y': 127.5},
        {'name': 'D', 'x': 290, 'y': 345},
        {'name': 'E', 'x': 0, 'y': 255},
        {'name': 'F', 'x': 220.836, 'y': 127.5}
    ])

    truss.add_bars([
        {'name': 'AB', 'first_joint_name': 'A', 'second_joint_name': 'B', 'bar_params': medium_2},
        {'name': 'BC', 'first_joint_name': 'B', 'second_joint_name': 'C', 'bar_params': medium_2},
        {'name': 'CD', 'first_joint_name': 'C', 'second_joint_name': 'D', 'bar_params': medium_2},
        {'name': 'DE', 'first_joint_name': 'D', 'second_joint_name': 'E', 'bar_params': medium_2},
        {'name': 'EF', 'first_joint_name': 'E', 'second_joint_name': 'F', 'bar_params': medium_2},
        {'name': 'AF', 'first_joint_name': 'F', 'second_joint_name': 'A', 'bar_params': medium_2},
        {'name': 'DF', 'first_joint_name': 'D', 'second_joint_name': 'F', 'bar_params': medium_2},
        {'name': 'BF', 'first_joint_name': 'B', 'second_joint_name': 'F', 'bar_params': medium_2},
    ])

    truss.add_loads([
        {'name': 'W', 'joint_name': 'C', 'x': 0, 'y': -0.675 * 1}
    ])

    truss.add_supports([
        {'name': 'A', 'joint_name': 'A', 'support_type': 'encastre'},
        {'name': 'E', 'joint_name': 'E', 'support_type': 'pin', 'pin_rotation': -math.pi / 2}
    ])

    results = Result(truss, solution_method=utils.SolveMethod.NUMPY_STD)

    plot_diagram(truss, results, show_reactions=True)


if __name__ == '__main__':
    test_build_standard()
