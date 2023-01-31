import numpy as np
import itertools
import string
import re
import os
import math
import warnings
from matplotlib import pyplot as plt
from typing import Hashable, Optional
from enum import Enum, auto, unique


#  Fix issue with warning appearing when run from .exe
if os.path.basename(__file__).endswith('.exe'):
    warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*",
                            category=UserWarning)  # deprecation warning inherits from UserWarning


@unique
class SolveMethod(Enum):

    """
    A class to contain the different methods for solving the truss force balance equation
    Ax = B. Can see the different methods using get_constants(SolveMethod).
    """

    NUMPY_STD = auto()
    NUMPY_SOLVE = auto()
    SCIPY = auto()


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
        NEWTONS: 1, KILONEWTONS: 1e3, POUND_FORCE: 0.224809,
        METRES: 1, CENTIMETRES: 1e-2, MILLIMETRES: 1e-3, INCHES: 0.0254,
    }


def iter_all_strings():
    for size in itertools.count(1):
        for s in itertools.product(string.ascii_uppercase, repeat=size):
            yield "".join(s)


def convert_to_valid_var_name(name: str, cls: object, allow_existing_vars=True) -> str:

    """
    Given a user-defined name, converts it to a similar looking valid variable name.
    e.g. `convert_to_valid_var_name("My First Truss")` -> "my_first_truss"
    If this already exists and `allow_existing_vars = False`, a number is appended to the name
    to make it distinct, e.g. "my_first_truss_2", "my_first_truss_3", etc.
    """

    if name in {'', None}:
        name = ''

    # remove trailing whitespace, convert to lowercase and replace spaces with underscores
    new_name = name.strip().upper().replace(' ', '_')

    # remove non-alphanumeric characters except underscores
    pattern = re.compile(r'[\W]+', re.UNICODE)
    new_name = pattern.sub('', new_name)

    if not allow_existing_vars:
        if new_name == '':  # if given name is blank, iterate through alphabet
            for s in iter_all_strings():
                if s not in cls.keys():
                    new_name = s
        elif not allow_existing_vars and new_name in cls.keys():  # if not, add _number to end
            suffix = 2
            while (new_name + '_' + str(suffix)) in cls.keys():
                suffix += 1

    return new_name


def get_constants(cls: type) -> dict[str, Hashable]:

    """
    Used to get a dict of constants {const_name: const_value}
    from the utility classes.
    """

    # get a list of the names of the constants
    names = list(filter(
        lambda a: not callable(getattr(cls(), a)) and not a.startswith('_') and a == a.upper(), dir(cls())))

    # get a list of the values of these constants
    vals = [getattr(cls(), a) for a in names]

    # return in dict {'name': value} form
    return dict(zip(names, vals))


def set_matplotlib_fullscreen() -> None:

    """
    Automatically set the matplotlib output to fullscreen.
    """

    import os
    from matplotlib import pyplot as plt

    backend = str(plt.get_backend())
    mgr = plt.get_current_fig_manager()
    if backend == 'TkAgg':  # used if PyQt5 is not installed
        if os.name == 'nt':
            mgr.window.state('zoomed')
        else:
            mgr.resize(*mgr.window.maxsize())
    elif backend == 'wxAgg':
        mgr.frame.Maximize(True)
    elif backend in ['Qt4Agg', 'Qt5Agg', 'QtAgg']:  # used if PyQt5 is installed
        mgr.window.showMaximized()
    else:
        raise RuntimeWarning(f'The backend in use, {backend}, is not supported in fullscreen mode.')


def find_free_space_around_joint(joint: object, results: object = None,
                                 truss: Optional[object] = None, show_reactions: bool = True,
                                 as_degrees: bool = False) -> float:

    """
    Helper function to find a place to label text around a joint. Finds a location
    at a fixed small distance from the joint, such that the surrounding bars, loads
    and supports/reaction arrows are as far away as possible.
    """

    support = truss.get_support_by_joint(joint)

    # find the angles occupied due to bars being there
    used_angles = [bar.get_direction(origin_joint=joint)
        for bar in truss.get_all_bars_connected_to_joint(joint)]

    # find the angles occupied due to load arrows being there
    used_angles += [load.direction for load in truss.get_all_loads_at_joint(joint)]

    # find the angles occupied due to support icons and/or reaction arrows being there
    # TODO: don't add if the reaction force is zero
    if support is not None:
        if show_reactions:
            if support.support_type == 'roller':
                used_angles.append(math.pi + support.reaction_direction)
            used_angles.append(math.atan2(*reversed(results.reactions[support.name])))

        else:
            if support.support_type == 'pin':
                used_angles.append(math.pi / 2 - support.pin_rotation)

    # sort ascending from 0 to 360 (through 2 * pi)
    used_angles = sorted([i % (2 * math.pi) for i in used_angles])

    # find the angular sizes of the gaps
    differences = [(used_angles[i] - used_angles[i - 1]) % (2 * math.pi) for i in range(len(used_angles))]

    # determine at what angle is the most free
    max_i = differences.index(max(differences))
    most_free_angle = np.average([used_angles[max_i], used_angles[max_i - 1]])
    if used_angles[max_i] < used_angles[max_i - 1]:
        most_free_angle -= math.pi

    return math.degrees(most_free_angle) if as_degrees else most_free_angle


def draw_support(x: float, y: float, size: float, support_type: str = 'pin', pin_rotation: float = 0,
                 roller_normal: np.array = None) -> None:

    """
    Draw a particular type of support, using the standard conventional symbols, on
    the matplotlib truss diagram. If roller is chosen, its direction is
    shown by rotating the drawing. Optional pin rotation in clockwise degrees from vertical.
    """

    # Helper function to rotate the drawing
    if pin_rotation != 0:  # either but not both: cannot be encastre
        if support_type == 'roller':
            a = math.pi / 2 - math.atan2(*reversed(roller_normal))

        elif support_type == 'pin':
            a = math.radians(pin_rotation)

        else:
            raise TypeError(f'''
            'The combination of supplied information: support type ({support_type}), pin rotation angle'
            '({pin_rotation}) and roller direction ({roller_normal}) is invalid.''')

        # function for rotating a given coordinate tuple _p = (_x, _y) by a radians clockwise about (x, y)
        rot = lambda _p: (x + (_p[0] - x) * math.cos(a) + (_p[1] - y) * math.sin(a),  # noqa
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
        ] + [(x_pos, y - (size / 3 if support_type == 'pin' else 8 / 15 * size))
            for x_pos, y_pos in zip(list(np.linspace(x - 0.3 * size, x + 0.5 * size, 5)), [y] * 5)
        ] + [(x_pos - size / 5, y - (8/15 * size if support_type == 'pin' else 11/15 * size))  # noqa \     
            for x_pos, y_pos in zip(list(np.linspace(x - 0.3 * size, x + 0.5 * size, 5)), [y] * 5)
        ] + [(x - size / 2, y - (size / 3 if support_type == 'pin' else 8 / 15 * size)),  # noqa
             (x + size / 2, y - (size / 3 if support_type == 'pin' else 8 / 15 * size))]

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
                plt.plot((x_pos, x_pos - size / 5), (y - size / 3, y - 8 / 15 * size),
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
                plt.plot([x_tr, _new_pts[n + 5][0]], [y_tr, _new_pts[n + 5][1]],
                    linewidth=1, color='black', zorder=0)

    if support_type == 'roller':
        # Roller symbol: pin with wheels, rotated about pin circle to show direction
        # Transform the important points to be plotted
        # NOTE: element indices are (0-15 unchanged) from pin
        # 16: wheel left centre point, 17: wheel right centre point

        _old_pts += [(x - (0.7 / (3 * math.sqrt(3))) * size, y - 13 / 30 * size),
            (x + (0.7 / (3 * math.sqrt(3))) * size, y - 13 / 30 * size)]

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
            plt.plot([x_tr, _new_pts[n + 5][0]], [y_tr, _new_pts[n + 5][1]],
                linewidth=1, color='black', zorder=0)

        plt.gca().add_patch(                                                        # wheels
            plt.Circle((xtl[16], ytl[16]), size / 10, color='black', linewidth=1, zorder=1))
        plt.gca().add_patch(
            plt.Circle((xtl[16], ytl[16]), size / 14, color='white', linewidth=1, zorder=1))
        plt.gca().add_patch(
            plt.Circle((xtl[17], ytl[17]), size / 10, color='black', linewidth=1, zorder=1))
        plt.gca().add_patch(
            plt.Circle((xtl[17], ytl[17]), size / 14, color='white', linewidth=1, zorder=1))
