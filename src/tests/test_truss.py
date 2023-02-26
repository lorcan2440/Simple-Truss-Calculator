if __name__ == "__main__":
    import __init__  # noqa

from truss import Result, init_truss, BadTrussError, Joint, Bar
import pytest
import numpy as np

import math


def test_underconstrained_arch():

    joints = ((0, 0), (100, 100 * math.sqrt(3)), (200, 0))
    bars = ("AB", "BC")
    loads = (("B", 0, -100), ("C", 0, -100))
    supports = (("A", "pin"), ("C", "roller", math.pi / 4))

    t = init_truss("Simple Arch")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    assert not t.is_statically_determinate()
    assert t.check_determinacy_type() == "underconstrained"

    with pytest.raises(
        BadTrussError,
        match=r".*\(underconstrained; mechanistic\). "
        r"Bars = \d+, Forces = \d+, Joints = \d+ .*",
    ):
        r = Result(t)
        del r


def test_overconstrained_truss():

    joints = ((0, 0), (100, 0), (200, 0), (100, 100))
    bars = ("AB", "BC", "AD", "CD", "BD")
    loads = [
        ("B", 0, -100),
    ]
    supports = (("A", "pin"), ("C", "pin"))

    t = init_truss("Small bridge")
    t.add_joints(joints).add_bars(bars).add_loads(loads).add_supports(supports)

    assert not t.is_statically_determinate()
    assert t.check_determinacy_type() == "overconstrained"

    with pytest.raises(
        BadTrussError,
        match=r".*\(overconstrained\). " r"Bars = \d+, Forces = \d+, Joints = \d+ .*",
    ):
        r = Result(t)
        del r


def __test_determinate_but_internally_singular_truss():

    # FIXME: this is not working correctly. It is computing a solution when there should be none.

    joints = ((0, 0), (100, 0), (0, 100), (100, 100), (0, 200), (100, 200))
    bars = ("AB", "CD", "EF", "AC", "BD", "CE", "DF", "AD", "BC")
    loads = [("E", 100, 50)]
    supports = (("A", "pin"), ("B", "roller"))

    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        t = (
            init_truss()
            .add_joints(joints)
            .add_bars(bars)
            .add_loads(loads)
            .add_supports(supports)
        )
        t.solve_and_plot()

    t = (
        init_truss()
        .add_joints(joints)
        .add_bars(bars)
        .add_loads(loads)
        .add_supports(supports)
    )

    with pytest.raises(
        BadTrussError,
        match=r"The truss contains mechanistic and/or overconstrained components",
    ):
        r = Result(t)
        del r


def test_unloaded():

    # build and solve a valid unloaded truss
    joints = ((0, 0), (1, 0.75), (2, 0))
    bars = ("AB", "BC")
    supports = (("A", "pin"), ("C", "pin"))
    t = init_truss().add_joints(joints).add_bars(bars).add_supports(supports)
    t.solve_and_plot()

    # simple test string representation
    r = Result(t)
    repr = (
        r.__repr__()
    )  # could not get capsys fixture to work to check print() to STDOUT
    assert len(repr) > 0


def test_bad_inputs():

    joints = ((0, 0), (1, 0.75), (2, 0))
    bars = ("AB", "BC")
    supports = (("A", "WAHEY!!"), ("C", "pin"))

    # test invalid support type
    with pytest.raises(ValueError, match=r"Support type must be"):
        t = init_truss().add_joints(joints).add_bars(bars).add_supports(supports)

    # replace lazy named bar with bad name
    bars = ("Bar AB", "BC")
    with pytest.raises(ValueError, match=r"Lazily evaluated bar names"):
        t = init_truss().add_joints(joints).add_bars(bars)

    bars = (("Bar AB",), ("BC", {"b": 50}))
    with pytest.raises(ValueError, match=r"Lazily evaluated bar names"):
        t = init_truss().add_joints(joints).add_bars(bars)

    # connect two joints in different trusses - cannot do this using builder functions
    t_A = init_truss("Truss A")
    t_B = init_truss("Truss B")
    j1 = Joint(t_A, "A", 10, 0)
    j2 = Joint(t_B, "B", 0, 10)
    with pytest.raises(
        BadTrussError, match=r"Bars must connect two joints in the same truss"
    ):
        t_A.bars.update({"AB": Bar("AB", j1, j2)})

    # total nonsense
    joints = "lalala"
    bars = 69420
    loads = lambda _: None  # noqa
    supports = NotImplementedError
    with pytest.raises(
        ValueError, match=r"The input `list_of_joints` must be one of the following"
    ):
        t = init_truss().add_joints(joints)
    with pytest.raises(
        ValueError, match=r"The input `list_of_bars` must be one of the following"
    ):
        t = init_truss().add_joints([(0, 0), (1, 0)]).add_bars(bars)
    with pytest.raises(
        ValueError, match=r"The input `list_of_loads` must be one of the following"
    ):
        t = init_truss().add_joints([(0, 0), (1, 0)]).add_bars(("AB",)).add_loads(loads)
    with pytest.raises(
        ValueError, match=r"The input `list_of_supports` must be one of the following"
    ):
        t = (
            init_truss()
            .add_joints([(0, 0), (1, 0)])
            .add_bars(("AB",))
            .add_loads([("A", 0, 1)])
            .add_supports(supports)
        )
        del t
