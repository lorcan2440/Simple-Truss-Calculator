if __name__ == '__main__':
    import __init__  # noqa

from truss import Result, init_truss, plot_diagram, load_truss_from_json, BadTrussError
import pytest
import os
import numpy as np

import utils

import math


# utility - quick build
def build_sdc_truss():

    my_truss = init_truss('SDC: Steel Cantilever')
    my_truss.add_joints(
        [(0, 0), (290, -90), (815, 127.5), (290, 345), (0, 255), (220.836, 127.5)]
    )
    my_truss.add_bars(['AB', 'BC', 'CD', 'DE', 'EF', 'AF', 'DF', 'BF'])
    my_truss.add_loads([('W', 'C', 0, -0.675)])
    my_truss.add_supports([('A', 'encastre'), ('E', 'pin', -math.pi / 2)])

    results = Result(my_truss)

    return my_truss, results


# test cases to check general functionality


def test_sdc_truss_verbose_creation():

    medium_2 = {"b": 16, "t": 1.1, "D": 5, "E": 210, "strength_max": 0.216}
    custom_params = medium_2

    truss = init_truss("SDC: Steel Cantilever", custom_params, "kN mm")

    truss.add_joints(
        [
            {"name": "A", "x": 0, "y": 0},
            {"name": "B", "x": 290, "y": -90},
            {"name": "C", "x": 815, "y": 127.5},
            {"name": "D", "x": 290, "y": 345},
            {"name": "E", "x": 0, "y": 255},
            {"name": "F", "x": 220.836, "y": 127.5},
        ]
    )

    truss.add_bars(
        [
            {
                "name": "AB",
                "first_joint_name": "A",
                "second_joint_name": "B",
                "bar_params": medium_2,
            },
            {
                "name": "BC",
                "first_joint_name": "B",
                "second_joint_name": "C",
                "bar_params": medium_2,
            },
            {
                "name": "CD",
                "first_joint_name": "C",
                "second_joint_name": "D",
                "bar_params": medium_2,
            },
            {
                "name": "DE",
                "first_joint_name": "D",
                "second_joint_name": "E",
                "bar_params": medium_2,
            },
            {
                "name": "EF",
                "first_joint_name": "E",
                "second_joint_name": "F",
                "bar_params": medium_2,
            },
            {
                "name": "AF",
                "first_joint_name": "F",
                "second_joint_name": "A",
                "bar_params": medium_2,
            },
            {
                "name": "DF",
                "first_joint_name": "D",
                "second_joint_name": "F",
                "bar_params": medium_2,
            },
            {
                "name": "BF",
                "first_joint_name": "B",
                "second_joint_name": "F",
                "bar_params": medium_2,
            },
        ]
    )

    truss.add_loads([{"name": "W", "joint_name": "C", "x": 0, "y": -0.675 * 1}])

    truss.add_supports(
        [
            {"name": "A", "joint_name": "A", "support_type": "encastre"},
            {
                "name": "E",
                "joint_name": "E",
                "support_type": "pin",
                "pin_rotation": -math.pi / 2,
            },
        ]
    )

    results = Result(truss)

    print(results)

    plot_diagram(truss, results, full_screen=False, show_reactions=True)

    fields = ("tensions", "reactions", "stresses", "strains", "buckling_ratios")

    # check determinate
    assert truss.check_determinacy_type() == 'determinate'

    # check results fields have been set
    assert hasattr(truss, "results")
    assert all([hasattr(results, attr) for attr in fields])
    assert set(truss.results.keys()) == set(fields)

    correct_tensions = {
        "AB": -1.0710587,
        "BC": -0.8817989,
        "CD": 0.8817989,
        "DE": 1.07105869,
        "EF": 1.3099198,
        "AF": -1.3099198,
        "DF": -0.6872788,
        "BF": 0.6872788,
    }
    correct_reactions = {"A": [2.1573529, 0.3375], "E": [-2.1573529, 0.3375]}
    correct_stresses = {
        "AB": -0.05694092,
        "BC": -0.04687926,
        "CD": 0.04687926,
        "DE": 0.05694092,
        "EF": 0.0696395,
        "AF": -0.06963954,
        "DF": -0.03653795,
        "BF": 0.03653795,
    }
    correct_strains = {
        "AB": -0.000271147235,
        "BC": -0.000223234576314,
        "CD": 0.0002232345763,
        "DE": 0.000271147234681,
        "EF": 0.0003316168665,
        "AF": -0.0003316168665,
        "DF": -0.0001739902284,
        "BF": 0.0001739902284,
    }
    correct_buckling_ratios = {
        "AB": 18.9777831,
        "BC": 35.5169,
        "CD": 35.5169,
        "DE": 18.9777831,
        "EF": 15.93747413,
        "AF": 15.93747413,
        "DF": 14.264509,
        "BF": 14.264509,
    }

    # check rounding is correct
    assert results.tensions == pytest.approx(truss.results["tensions"], rel=1e-3)

    # check values are correct
    assert correct_tensions == pytest.approx(truss.results["tensions"], rel=1e-6)
    for key in correct_reactions:
        assert correct_reactions[key] == pytest.approx(
            truss.results["reactions"][key], rel=1e-6
        )
    assert correct_stresses == pytest.approx(truss.results["stresses"], rel=1e-6)
    assert correct_strains == pytest.approx(truss.results["strains"], rel=1e-6)
    assert correct_buckling_ratios == pytest.approx(
        truss.results["buckling_ratios"], rel=1e-6
    )


def test_truss_semi_lazy_creation():

    bar_params = {"b": 0.018, "t": 0.005, "D": 0.024, "E": 2.1e11, "strength_max": 3.2e9}

    joints = (('P', 0, 0), ('Q', 100, 0), ('R', 200, 0), ('S', 100, 100))  # set names
    bars = (('Left Span', 'P', 'Q', bar_params), ('Right Span', 'Q', 'R', bar_params), ('Left Mast', 'P', 'S'),
        ('Right Mast', 'R', 'S'), ('Pillar', 'Q', 'S'))
    loads = [('Truck', 'Q', 0, -100),]
    supports = (('Enbankment', 'P', 'encastre'), ('Rail', 'R', 'roller', 0.3))

    t = init_truss('BTEC Bridge')
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)
    r = Result(t)
    plot_diagram(t, r)


def test_truss_very_lazy_creation_with_multiple_steps():

    joints = ((0, 0), (100, 0), (200, 0), (300, 0), (400, 0), (100, 100), (200, 100), (300, 100))
    bars = ('AB', 'BC', 'CD', 'DE', 'AF', 'BF', 'CF', 'CG', 'CH', 'DH', 'EH', 'FG', 'GH')
    loads = [('A', 0, -100), ('B', 0, -200), ('C', 0, -200), ('D', 0, -200), ('E', 0, -100)]
    supports = (('A', 'pin'), ('E', 'roller'))

    t = init_truss('Big Bridge')
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    bar_params = {"b": 0.018, "t": 0.005, "D": 0.024, "E": 1.0e11, "strength_max": 1.3e9}

    extra_joints = ((200, 200), (150, 50), (250, 50))
    extra_bars = (('F', 'I', bar_params), ('I', 'H', bar_params),
        ('I', 'J',), ('I', 'K',))

    t.add_joints(extra_joints).add_bars(extra_bars)

    extra_bars = (('JB', bar_params), ('KD',))
    t.add_bars(extra_bars)

    r = Result(t)
    plot_diagram(t, r)


def test_build_large_bridge_with_angled_roller():

    joints = ((0, 0), (100, 0), (200, 0), (300, 0), (400, 0), (100, 100), (200, 100), (300, 100))
    bars = ('AB', 'BC', 'CD', 'DE', 'AF', 'BF', 'CF', 'CG', 'CH', 'DH', 'EH', 'FG', 'GH')
    loads = [('A', 0, -100), ('B', 0, -200), ('C', 0, -200), ('D', 0, -200), ('E', 0, -100)]
    supports = (('A', 'pin'), ('E', 'roller', math.pi / 4))

    t = init_truss('Big Bridge')
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)
    r = Result(t)
    plot_diagram(t, r)


def test_small_arch():

    joints = ((0, 0), (100, 100 * math.sqrt(3)), (200, 0))
    bars = ('AB', 'BC', 'AC')
    loads = (('B', 0, -100),)
    supports = (('A', 'pin'), ('C', 'roller', math.pi / 4))

    t = init_truss('Triangle')
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    r = Result(t)
    plot_diagram(t, r)


def test_underconstrained_arch():

    joints = ((0, 0), (100, 100 * math.sqrt(3)), (200, 0))
    bars = ('AB', 'BC')
    loads = (('B', 0, -100), ('C', 0, -100))
    supports = (('A', 'pin'), ('C', 'roller', math.pi / 4))

    t = init_truss('Simple Arch')
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    assert not t.is_statically_determinate()
    assert t.check_determinacy_type() == 'underconstrained'

    with pytest.raises(BadTrussError, match=r'.*\(underconstrained; mechanistic\). '
            r'Bars = \d+, Forces = \d+, Joints = \d+ .*'):
        r = Result(t)
        del r


def test_overconstrained_truss():

    joints = ((0, 0), (100, 0), (200, 0), (100, 100))
    bars = ('AB', 'BC', 'AD', 'CD', 'BD')
    loads = [('B', 0, -100),]
    supports = (('A', 'pin'), ('C', 'pin'))

    t = init_truss('Small bridge')
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    assert not t.is_statically_determinate()
    assert t.check_determinacy_type() == 'overconstrained'

    with pytest.raises(BadTrussError, match=r'.*\(overconstrained\). '
            r'Bars = \d+, Forces = \d+, Joints = \d+ .*'):
        r = Result(t)
        del r


def test_determinate_but_internally_singular_truss():

    joints = ((0, 0), (100, 0), (0, 100), (100, 100), (0, 200), (100, 200))
    bars = ('AB', 'CD', 'EF', 'AC', 'BD', 'CE', 'DF', 'AD', 'BC')
    loads = [('C', 100, 50)]
    supports = (('A', 'pin'), ('B', 'roller'))

    with pytest.raises(np.linalg.LinAlgError, match='Singular matrix'):
        t = init_truss().add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports).solve()

    t = init_truss().add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    with pytest.raises(BadTrussError,
            match=r'The truss contains mechanistic and/or overconstrained components'):
        r = Result(t)
        del r


def test_save_and_load_json_truss():

    t = (
        init_truss(units=(utils.Unit.KILONEWTONS, utils.Unit.MILLIMETRES))
        .add_joints(
            [(0, 0), (290, -90), (815, 127.5), (290, 345), (0, 255), (220.836, 127.5)]
        )
        .add_bars(("AB", "BC", "CD", "DE", "EF", "AF", "DF", "BF"))
        .add_loads([("C", 0, -0.675 * 1)])
        .add_supports([("A", "encastre"), ("E", "pin", -math.pi / 2)])
    )

    r = Result(t)
    plot_diagram(t, r)
    t.dump_truss_to_json(filename="temp_truss.json")
    assert os.path.exists("temp_truss.json")
    _r_new = load_truss_from_json("temp_truss.json", full_screen=False)
    os.unlink("temp_truss.json")
    assert r.truss.results["tensions"] == pytest.approx(
        _r_new.results["tensions"], 1e-9
    )


def test_unloaded():

    joints = ((0, 0), (1, 0.75), (2, 0))
    bars = ('AB', 'BC')
    supports = (('A', 'pin'), ('C', 'pin'))
    t = init_truss().add_joints(joints).add_bars(bars).add_supports(supports)
    r = Result(t)
    plot_diagram(t, r)


def test_bad_inputs():

    joints = ((0, 0), (1, 0.75), (2, 0))
    bars = ('AB', 'BC')
    supports = (('A', 'WAHEY!!'), ('C', 'pin'))

    # test invalid support type
    with pytest.raises(ValueError, match=r'Support type must be'):
        t = init_truss().add_joints(joints).add_bars(bars).add_supports(supports)

    # replace lazy named bar with bad name
    bars = ('Bar AB', 'BC')
    with pytest.raises(ValueError, match=r'Lazily evaluated bar names'):
        t = init_truss().add_joints(joints).add_bars(bars)

    bars = (('Bar AB',), ('BC', {'b': 50}))
    with pytest.raises(ValueError, match=r'Lazily evaluated bar names'):
        t = init_truss().add_joints(joints).add_bars(bars)

    # total nonsense
    joints = 'lalala'
    bars = 69420
    loads = lambda _: None  # noqa
    supports = NotImplementedError
    with pytest.raises(ValueError, match=r'The input `list_of_joints` must be one of the following'):
        t = init_truss().add_joints(joints)
    with pytest.raises(ValueError, match=r'The input `list_of_bars` must be one of the following'):
        t = init_truss().add_joints([(0, 0), (1, 0)]).add_bars(bars)
    with pytest.raises(ValueError, match=r'The input `list_of_loads` must be one of the following'):
        t = init_truss().add_joints([(0, 0), (1, 0)]).add_bars(('AB',)).add_loads(loads)
    with pytest.raises(ValueError, match=r'The input `list_of_supports` must be one of the following'):
        t = init_truss().add_joints([(0, 0), (1, 0)]).add_bars(('AB',)).add_loads(
            [('A', 0, 1)]).add_supports(supports)
        del t


def test_autobuild_sdc():
    plot_diagram(*build_sdc_truss())


if __name__ == "__main__":  # pragma: no cover
    test_autobuild_sdc()
