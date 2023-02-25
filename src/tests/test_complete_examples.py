if __name__ == "__main__":
    import __init__  # noqa

from truss import Result, init_truss, plot_diagram, load_truss_from_json
import pytest

import os
import math

import utils_truss


# utility - quick build
def build_sdc_truss():

    my_truss = init_truss("SDC: Steel Cantilever")
    my_truss.add_joints(
        [(0, 0), (290, -90), (815, 127.5), (290, 345), (0, 255), (220.836, 127.5)]
    )
    my_truss.add_bars(["AB", "BC", "CD", "DE", "EF", "AF", "DF", "BF"])
    my_truss.add_loads([("W", "C", 0, -0.675)])
    my_truss.add_supports([("A", "encastre"), ("E", "pin", -math.pi / 2)])

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

    plot_diagram(truss, results, full_screen=False, show_reactions=True)

    fields = (
        "tensions",
        "reactions",
        "stresses",
        "strains",
        "buckling_ratios",
        "safety_factors",
    )

    # check determinate
    assert truss.check_determinacy_type() == "determinate"

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
        "AB": 48.8359201,
        "BC": 91.39637078,
        "CD": 91.3963708,
        "DE": 48.8359201,
        "EF": 41.0122305,
        "AF": 41.0122305,
        "DF": 36.7071539,
        "BF": 36.7071539,
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
    assert all([x > 3 for x in truss.results["safety_factors"].values()])


def test_truss_semi_lazy_creation():

    thin = {
        "b": 50,
        "t": 9,
        "D": 6,
        "E": 180,  # GPa
        "strength_max": 0.4,  # GPa
    }
    thick = {
        "b": 60,
        "t": 20,
        "D": 12,
        "E": 400,  # GPa
        "strength_max": 0.6,  # GPa
    }

    joints = (("P", 0, 0), ("Q", 100, 0), ("R", 200, 0), ("S", 100, 100))  # set names
    bars = (
        ("Left Span", "P", "Q", thin),
        ("Right Span", "Q", "R", thin),
        ("Left Mast", "P", "S", thick),
        ("Right Mast", "R", "S", thick),
        ("Pillar", "Q", "S", thin),
    )
    loads = [
        ("Truck", "Q", 0, -100),
    ]
    supports = (("Enbankment", "P", "encastre"), ("Rail", "R", "roller", 0.3))

    t = init_truss("BTEC Bridge")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)
    t.solve_and_plot()


def test_truss_very_lazy_creation_with_multiple_steps():

    joints = (
        (0, 0),
        (100, 0),
        (200, 0),
        (300, 0),
        (400, 0),
        (100, 100),
        (200, 100),
        (300, 100),
    )
    bars = (
        "AB",
        "BC",
        "CD",
        "DE",
        "AF",
        "BF",
        "CF",
        "CG",
        "CH",
        "DH",
        "EH",
        "FG",
        "GH",
    )
    loads = [
        ("A", 0, -100),
        ("B", 0, -200),
        ("C", 0, -200),
        ("D", 0, -200),
        ("E", 0, -100),
    ]
    supports = (("A", "pin"), ("E", "roller"))

    t = init_truss("Big Bridge")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    bar_params = {
        "b": 0.018,
        "t": 0.005,
        "D": 0.024,
        "E": 1.0e11,
        "strength_max": 1.3e9,
    }

    extra_joints = ((200, 200), (150, 50), (250, 50))
    extra_bars = (
        ("F", "I", bar_params),
        ("I", "H", bar_params),
        (
            "I",
            "J",
        ),
        (
            "I",
            "K",
        ),
    )

    t.add_joints(extra_joints).add_bars(extra_bars)

    extra_bars = (("JB", bar_params), ("KD",))
    t.add_bars(extra_bars)

    t.solve_and_plot()


def test_build_large_bridge_with_angled_roller():

    joints = (
        (0, 0),
        (100, 0),
        (200, 0),
        (300, 0),
        (400, 0),
        (100, 100),
        (200, 100),
        (300, 100),
    )
    bars = (
        "AB",
        "BC",
        "CD",
        "DE",
        "AF",
        "BF",
        "CF",
        "CG",
        "CH",
        "DH",
        "EH",
        "FG",
        "GH",
    )
    loads = [
        ("A", 0, -1),
        ("B", 0, -2),
        ("C", 0, -2),
        ("D", 0, -2),
        ("E", 0, -1),
    ]
    supports = (("A", "pin"), ("E", "roller", math.pi / 4))

    t = init_truss("Big Bridge")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)
    t.solve_and_plot()


def test_small_arch():

    joints = ((0, 0), (100, 100 * math.sqrt(3)), (200, 0))
    bars = ("AB", "BC", "AC")
    loads = (("B", 0, -100),)
    supports = (("A", "pin"), ("C", "roller", math.pi / 4))

    t = init_truss("Triangle")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)
    t.solve_and_plot()


def test_irregular_truss():

    joints = ((0, 0), (7, 4), (10, 0), (6, 2))
    bars = ("AB", "BC", "CD", "AD", "BD")
    loads = (("B", 20, -60), ("D", 0, -50))
    supports = (("A", "pin"), ("C", "roller"))

    t = init_truss("Ex2 Q5", units="kN m")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    r = Result(t)
    assert pytest.approx(r.tensions["BD"]) == 174.4133


def test_crane_truss():

    joints = (
        ("A", 0, 0),
        ("B", 3, 0),
        ("C", 0, 5),
        ("D", 3, 5),
        ("E", 0, 10),
        ("F", 3, 10),
        ("G", 0, 15),
        ("H", 3, 15),
        ("I", -3, 15),
        ("J", -6, 15),
        ("K", -3, 17.5),
        ("L", 0, 20),
        ("M", 3, 20),
        ("N", 6, 19),
        ("O", 9, 18),
        ("P", 12, 17),
        ("Q", 15, 16),
        ("R", 18, 15),
        ("S", 15, 15),
        ("T", 12, 15),
        ("U", 9, 15),
        ("V", 6, 15),
    )
    bars = (
        "AD",
        "AC",
        "BD",
        "CD",
        "CE",
        "CF",
        "DF",
        "EF",
        "EG",
        "EH",
        "FH",
        "GH",
        "IJ",
        "JK",
        "IK",
        "KL",
        "IL",
        "IG",
        "LG",
        "ML",
        "GM",
        "MH",
        "MV",
        "MN",
        "NV",
        "NO",
        "NU",
        "OU",
        "OP",
        "OT",
        "PT",
        "PQ",
        "PS",
        "QS",
        "QR",
        "RS",
        "ST",
        "UT",
        "UV",
        "HV",
    )
    loads = (("U", 0, -5), ("R", 0, -10))
    supports = (("A", "pin"), ("B", "pin"))

    t = init_truss("Crane", units="kN m")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)
    r = Result(t)
    assert pytest.approx(r.tensions["MV"]) == 4.373214


def test_save_and_load_json_truss():

    t = (
        init_truss(units=(utils_truss.Unit.KILONEWTONS, utils_truss.Unit.MILLIMETRES))
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
