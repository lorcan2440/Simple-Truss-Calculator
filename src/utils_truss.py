import numpy as np
import itertools
import string
import os
import math
import warnings
from matplotlib import pyplot as plt
import matplotlib as mpl
import sigfig
from typing import Optional
from enum import Enum, unique


# some default values. symbols defined on databook pg. 8
DEFAULT_BAR_PARAMS = {
    "b": 0.016,  # 16 mm
    "t": 0.004,  # 4 mm
    "D": 0.020,  # 2 cm
    "E": 210e9,  # 210 GPa
    "strength_max": 240e6,  # 240 MPa
}  # in (N, m)


@unique
class Unit(Enum):

    # units of force
    NEWTONS = "N"
    KILONEWTONS = "kN"
    POUND_FORCE = "lbf"
    # units of length
    METRES = "m"
    CENTIMETRES = "cm"
    MILLIMETRES = "mm"
    INCHES = "in"

    # conversion lookup table, all units are converted to metric N-m-Pa internally.
    # value in N-m-Pa = value given * _CONV[unit given]
    _CONV = {
        NEWTONS: 1,
        KILONEWTONS: 1e3,
        POUND_FORCE: 0.224809,
        METRES: 1,
        CENTIMETRES: 1e-2,
        MILLIMETRES: 1e-3,
        INCHES: 0.0254,
    }


def iter_all_strings(start: int = 0):
    for size in itertools.count(1):
        for s in itertools.islice(
            itertools.product(string.ascii_uppercase, repeat=size), start, None
        ):
            yield "".join(s)


def set_matplotlib_fullscreen() -> None:  # pragma: no cover

    """
    Automatically set the matplotlib output to fullscreen.
    """

    # this cannot be tested - the test environment does not have any of these backends.

    from matplotlib import pyplot as plt

    backend = str(plt.get_backend())
    mgr = plt.get_current_fig_manager()
    if backend == "TkAgg":  # used if PyQt5 is not installed
        if os.name == "nt":
            mgr.window.state("zoomed")
        else:
            mgr.resize(*mgr.window.maxsize())
    elif backend == "wxAgg":
        mgr.frame.Maximize(True)
    elif "Qt" in backend:  # used if PyQt5 is installed
        mgr.window.showMaximized()
    else:
        raise RuntimeWarning(
            f"The backend in use, {backend}, is not supported in fullscreen mode."
        )


def find_free_space_around_joint(
    joint: object,
    results: object = None,
    truss: Optional[object] = None,
    show_reactions: bool = True,
    as_degrees: bool = False,
) -> float:

    """
    Helper function to find a place to label text around a joint. Finds a location
    at a fixed small distance from the joint, such that the surrounding bars, loads
    and supports/reaction arrows are as far away as possible.
    """

    support = truss.get_support_by_joint(joint)

    # find the angles occupied due to bars being there
    used_angles = [
        bar.get_direction(origin_joint=joint)
        for bar in truss.get_all_bars_connected_to_joint(joint)
    ]

    # find the angles occupied due to load arrows being there
    used_angles += [load.direction for load in truss.get_all_loads_at_joint(joint)]

    # find the angles occupied due to support icons and/or reaction arrows being there
    # TODO: don't add if the reaction force is zero
    if support is not None:
        if show_reactions:
            if support.support_type == "roller":
                used_angles.append(math.pi + support.normal_direction)
            used_angles.append(math.atan2(*reversed(results.reactions[support.name])))

        else:
            if support.support_type == "pin":
                used_angles.append(math.pi / 2 - support.pin_rotation)

    # sort ascending from 0 to 360 (through 2 * pi)
    used_angles = sorted([i % (2 * math.pi) for i in used_angles])

    # find the angular sizes of the gaps
    differences = [
        (used_angles[i] - used_angles[i - 1]) % (2 * math.pi)
        for i in range(len(used_angles))
    ]

    # determine at what angle is the most free
    max_i = differences.index(max(differences))
    most_free_angle = np.average([used_angles[max_i], used_angles[max_i - 1]])
    if used_angles[max_i] < used_angles[max_i - 1]:
        most_free_angle -= math.pi

    return math.degrees(most_free_angle) if as_degrees else most_free_angle


def rotate_coords(p: tuple[float], x: float, y: float, a: float) -> tuple[float]:
    """
    Rotate a coordinate `p = (_x, _y)` about a centre `(x, y)` by `a` radians counterclockwise.
    """

    return (
        x + (p[0] - x) * math.cos(a) - (p[1] - y) * math.sin(a),
        y + (p[0] - x) * math.sin(a) + (p[1] - y) * math.cos(a),
    )


def draw_support(
    x: float,
    y: float,
    size: float,
    support_type: str = "pin",
    pin_rotation: float = 0,
    roller_normal: np.array = None,
) -> None:

    """
    Draw a particular type of support, using the standard conventional symbols, on
    the matplotlib truss diagram. If roller is chosen, its direction is
    shown by rotating the drawing. Optional pin rotation in clockwise degrees from vertical.
    """

    a = pin_rotation

    if support_type == "encastre":

        # Encastre symbol: solid line and hashed lines representing ground
        plt.plot(
            (x - size / 2, x + size / 2),
            (y, y),  # horizontal line
            linewidth=1,
            color="black",
            zorder=0,
        )
        for x_pos in np.linspace(x - 0.3 * size, x + 0.5 * size, 5):
            plt.plot(
                (x_pos, x_pos - size / 5),
                (y, y - size / 5),  # hashed lines
                linewidth=1,
                color="black",
                zorder=0,
            )

    if (support_type == "pin" and pin_rotation != 0) or support_type == "roller":
        # NOTE: element indices are
        # 0: triangle top left, 1: triangle bottom left, 2: triangle bottom right, 3: triangle top right
        # 4,5,6,7,8: ground top right diagonal points, 9,10,11,12,13: ground bottom left diagonal points
        # 14: ground left point, 15: ground right point
        _old_pts = (
            [
                (x - size / 20, y - math.sqrt(3) * size / 20),
                (x - (1 / (3 * math.sqrt(3))) * size, y - size / 3),
                (x + (1 / (3 * math.sqrt(3))) * size, y - size / 3),
                (x + size / 20, y - math.sqrt(3) * size / 20),
            ]
            + [
                (x_pos, y - (size / 3 if support_type == "pin" else 8 / 15 * size))
                for x_pos, y_pos in zip(
                    list(np.linspace(x - 0.3 * size, x + 0.5 * size, 5)), [y] * 5
                )
            ]
            + [
                (
                    x_pos - size / 5,
                    y - (8 / 15 * size if support_type == "pin" else 11 / 15 * size),
                )  # noqa \
                for x_pos, y_pos in zip(
                    list(np.linspace(x - 0.3 * size, x + 0.5 * size, 5)), [y] * 5
                )
            ]
            + [
                (
                    x - size / 2,
                    y - (size / 3 if support_type == "pin" else 8 / 15 * size),
                ),  # noqa
                (
                    x + size / 2,
                    y - (size / 3 if support_type == "pin" else 8 / 15 * size),
                ),
            ]
        )

    if support_type == "pin":
        if pin_rotation == 0:
            # Pin symbol: triangle resting on ground
            plt.plot(
                (
                    x - size / 20,
                    x - (1 / (3 * math.sqrt(3))) * size,  # equilateral triangle
                    x + (1 / (3 * math.sqrt(3))) * size,
                    x + size / 20,
                ),
                (
                    y - math.sqrt(3) * size / 20,
                    y - size / 3,
                    y - size / 3,
                    y - math.sqrt(3) * size / 20,
                ),
                linewidth=1,
                color="black",
                zorder=0,
            )

            plt.gca().add_patch(  # circle pin
                plt.Circle((x, y), size / 10, color="black", linewidth=1, zorder=1)
            )
            plt.gca().add_patch(
                plt.Circle((x, y), size / 14, color="white", linewidth=1, zorder=1)
            )

            plt.plot(
                (x - size / 2, x + size / 2),
                (y - size / 3, y - size / 3),  # ground
                linewidth=1,
                color="black",
                zorder=0,
            )
            for x_pos in np.linspace(x - 0.3 * size, x + 0.5 * size, 5):
                plt.plot(
                    (x_pos, x_pos - size / 5),
                    (y - size / 3, y - 8 / 15 * size),
                    linewidth=1,
                    color="black",
                    zorder=0,
                )
        else:
            # Transform the important points to be plotted
            rot = lambda _p: rotate_coords(_p, x, y, a)  # noqa
            _new_pts = list(map(rot, _old_pts))
            xtl, ytl = map(list, zip(*_new_pts))

            plt.plot(
                xtl[0:4], ytl[0:4], linewidth=1, color="black", zorder=0
            )  # triangle

            plt.gca().add_patch(  # circle pin
                plt.Circle((x, y), size / 10, linewidth=1, zorder=1, color="black")
            )
            plt.gca().add_patch(
                plt.Circle((x, y), size / 14, linewidth=1, zorder=1, color="white")
            )

            plt.plot(xtl[14:], ytl[14:], linewidth=1, color="black", zorder=0)  # ground
            for i, (x_tr, y_tr) in enumerate(_new_pts[4:9]):
                n = i + 4
                plt.plot(
                    [x_tr, _new_pts[n + 5][0]],
                    [y_tr, _new_pts[n + 5][1]],
                    linewidth=1,
                    color="black",
                    zorder=0,
                )

    if support_type == "roller":
        # Roller symbol: pin with wheels, rotated about pin circle to show direction
        # Transform the important points to be plotted
        # NOTE: element indices are (0-15 unchanged) from pin
        # 16: wheel left centre point, 17: wheel right centre point

        _old_pts += [
            (x - (0.7 / (3 * math.sqrt(3))) * size, y - 13 / 30 * size),
            (x + (0.7 / (3 * math.sqrt(3))) * size, y - 13 / 30 * size),
        ]

        rot = lambda _p: rotate_coords(_p, x, y, a)  # noqa
        _new_pts = list(map(rot, _old_pts))
        xtl, ytl = map(list, zip(*_new_pts))

        plt.plot(xtl[0:4], ytl[0:4], linewidth=1, color="black", zorder=0)  # triangle

        plt.gca().add_patch(  # circle pin
            plt.Circle((x, y), size / 10, linewidth=1, zorder=1, color="black")
        )
        plt.gca().add_patch(
            plt.Circle((x, y), size / 14, linewidth=1, zorder=1, color="white")
        )

        plt.plot(xtl[14:16], ytl[14:16], linewidth=1, color="black", zorder=0)  # ground
        for i, (x_tr, y_tr) in enumerate(_new_pts[4:9]):
            n = i + 4
            plt.plot(
                [x_tr, _new_pts[n + 5][0]],
                [y_tr, _new_pts[n + 5][1]],
                linewidth=1,
                color="black",
                zorder=0,
            )

        plt.gca().add_patch(  # wheels
            plt.Circle(
                (xtl[16], ytl[16]), size / 10, color="black", linewidth=1, zorder=1
            )
        )
        plt.gca().add_patch(
            plt.Circle(
                (xtl[16], ytl[16]), size / 14, color="white", linewidth=1, zorder=1
            )
        )
        plt.gca().add_patch(
            plt.Circle(
                (xtl[17], ytl[17]), size / 10, color="black", linewidth=1, zorder=1
            )
        )
        plt.gca().add_patch(
            plt.Circle(
                (xtl[17], ytl[17]), size / 14, color="white", linewidth=1, zorder=1
            )
        )


def get_safety_factor(bar: object, bar_force: float, **kwargs) -> tuple[float, str]:

    bar_area = bar.effective_area  # mm^2
    bar_stress = bar_force / bar_area  # kN mm^-2 = GPa
    bar_length = bar.length
    yield_limit_stress = bar.params["strength_max"]

    if bar_force > 0:
        # tension: check for yielding
        if bar_stress >= yield_limit_stress:
            warnings.warn(
                f"Bar {bar.name} is expected to have yielded in tension: "
                f"bar stress {bar_stress}, yield stress {yield_limit_stress}",
                UserWarning,
            )

    elif bar_force < 0:
        # compression: check for crushing or buckling
        eta = 0.003 * bar.buckling_ratio  # Robertson, 1925, EN 1993

        joint_1_type = getattr(
            bar.truss.get_support_by_joint(bar.first_joint), "support_type", "pin"
        )
        joint_2_type = getattr(
            bar.truss.get_support_by_joint(bar.second_joint), "support_type", "pin"
        )

        match [joint_1_type, joint_2_type]:
            case ["encastre", "encastre"]:
                l_eff = 0.5 * bar_length
            case ["encastre", "pin"] | ["pin", "encastre"]:
                l_eff = 0.7 * bar_length
            case ["encastre", "roller"] | ["roller", "encastre"]:
                l_eff = 2.0 * bar_length
            case _:
                l_eff = 1.0 * bar_length

        euler_stress = (
            np.pi**2 * bar.params["E"] / (l_eff / bar.radius_of_gyration) ** 2
        )
        yield_limit_stress = min(
            bar.params["strength_max"],
            (1 + eta) * euler_stress
            + yield_limit_stress
            - np.sqrt(
                ((1 + eta) * euler_stress + yield_limit_stress) ** 2
                - 4 * euler_stress * yield_limit_stress
            )
            / 2,
        )

        if abs(bar_stress) >= yield_limit_stress:
            if yield_limit_stress == bar.params["strength_max"]:
                warnings.warn(
                    f"Bar {bar.name} is expected to have yielded in compression (crushing): "
                    f"bar stress {bar_stress}, yield stress {-1 * yield_limit_stress}",
                    UserWarning,
                )
            elif yield_limit_stress < bar.params["strength_max"]:
                warnings.warn(
                    f"Bar {bar.name} is expected to have failed by buckling: "
                    f"bar stress {bar_stress}, critical stress {-1 * yield_limit_stress}",
                    UserWarning,
                )

    try:
        safety_factor = sigfig.sigfig.round(
            yield_limit_stress / abs(bar_stress), **kwargs
        )
    except ZeroDivisionError:
        safety_factor = float(np.inf)

    return safety_factor


def get_colour_from_sf(sf: float):
    if sf <= 1:
        x = 0
    elif sf >= 2:
        x = 1
    else:
        x = sf - 1
    return mpl.colors.hsv_to_rgb((0.3 * x, 0.75, 0.75))


def round_sigfig(num: float | list[float], sig_figs: int):
    if num in (float("inf"), float("-inf")):
        return num
    else:
        if sig_figs is not None:
            if isinstance(num, (list, tuple)):
                return type(num)([sigfig.round(x, sigfigs=sig_figs) for x in num])
            else:
                return float(sigfig.round(num, sigfigs=sig_figs))
        else:
            return num
