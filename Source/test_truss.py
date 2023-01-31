from truss import Result, init_truss, plot_diagram, load_truss_from_json
import pytest
import os

import utils

import math


# test cases to check general functionality


def test_build_standard_SDC_truss():

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

    results = Result(truss, solution_method=utils.SolveMethod.NUMPY_STD)

    plot_diagram(truss, results, full_screen=False, show_reactions=True)

    fields = ("tensions", "reactions", "stresses", "strains", "buckling_ratios")

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


# test cases for coverage

if __name__ == "__main__":
    test_save_and_load_json_truss()
