import Truss_Calculator as tc       # local file import
import timeit, unittest


class TrussTests(unittest.TestCase):

    """
    Unit testing framework for the truss calculator.
    Values verified with https://skyciv.com/free-truss-calculator/.

    TODO:   add some more edge cases

    TODO:   if there are multiple loads/supports, label their names at
            the arrow middle or support normal direction instead of using `find_free_space_around_joint()`.

    TODO:   if a reaction force is zero, do not show its arrow even if `show_reactions` is `True`.
    """

    global weak, medium_1, medium_2, strong

    weak        = {"b": 12.5,   "t": 0.7,   "D": 5,     "E": 210,   "strength_max": 0.216}
    medium_1    = {"b": 16,     "t": 0.9,   "D": 5,     "E": 210,   "strength_max": 0.216}
    medium_2    = {"b": 16,     "t": 1.1,   "D": 5,     "E": 210,   "strength_max": 0.216}
    strong      = {"b": 19,     "t": 1.1,   "D": 5,     "E": 210,   "strength_max": 0.216}


    def test_SDC_truss(self):

        """
        Case 1: standard well-built truss. Uses the module factory functions.
        Represents the truss built in the SDC project.
        """

        tc.create_truss('SDC: Steel Cantilever')
        tc.create_joint('Joint A', 0, 0)
        tc.create_joint('Joint B', 290, -90)
        tc.create_joint('Joint C', 815, 127.5)
        tc.create_joint('Joint D', 290, 345)
        tc.create_joint('Joint E', 0, 255)
        tc.create_joint('Joint F', 220.836, 127.5)

        tc.create_bar('Bar AB', 'Joint A', 'Joint B', medium_2)
        tc.create_bar('Bar BC', 'Joint B', 'Joint C', strong)
        tc.create_bar('Bar CD', 'Joint C', 'Joint D', medium_1)
        tc.create_bar('Bar DE', 'Joint D', 'Joint E', medium_1)
        tc.create_bar('Bar EF', 'Joint E', 'Joint F', medium_1)
        tc.create_bar('Bar AF', 'Joint F', 'Joint A', medium_2)
        tc.create_bar('Bar DF', 'Joint F', 'Joint D', medium_1)
        tc.create_bar('Bar BF', 'Joint F', 'Joint B', weak)

        tc.create_load('W', 'Joint C', 0, -0.675 * 1)

        tc.create_support('Support A', 'Joint A', support_type='encastre')
        tc.create_support('Support E', 'Joint E', support_type='pin', pin_rotation=90)

        try:  # Get the results of the truss calculation and display graphic
            my_results = tc.active_truss.Result(tc.active_truss, sig_figs=3, 
                                                solution_method=tc.SolveMethod.NUMPY_STD)
            print(my_results)
        except tc.np.linalg.LinAlgError as e:  # The truss was badly made, so could not be solved
            tc.active_truss.classify_error_in_truss(e)

        tc.plot_diagram(tc.active_truss, my_results, show_reactions=True)


    def test_multiple_loads(self):

        """
        Case 2: A truss with multiple loads on different joints.
        """

        joints = ((0, 0), (5, 0), (0, 4), (2, 6))
        bars = (('AC', strong), ('BC', strong), ('CD', strong), ('BD', strong))
        loads = [('C', 1000, 0), ('D', 0, -750)]
        supports = (('A', {'support_type': 'pin'}), ('B', {'support_type': 'pin'}))

        tc.create_truss('Multiple loads')
        results = build_from_lists(joints, bars, loads, supports,
                                   sig_figs=3, solution_method=tc.SolveMethod.NUMPY_STD)

        tc.plot_diagram(tc.active_truss, results, show_reactions=True)


    def test_with_angled_roller(self):

        """
        Case 3: A truss with a roller support at an inclined angle.
        """

        joints = ((0, 1), (1, 0), (1, 1))
        bars = (('AB', strong), ('BC', strong), ('AC', strong))
        loads = [('C', 1, 2)]
        supports = (('A', {'support_type': 'pin', 'pin_rotation': 90}), 
                    ('B', {'support_type': 'roller', 'roller_normal_vector': (-1, 2)})
                    )

        tc.create_truss('Angled roller support')
        results = build_from_lists(joints, bars, loads, supports,
                                   sig_figs=3, solution_method=tc.SolveMethod.NUMPY_STD)

        tc.plot_diagram(tc.active_truss, results, show_reactions=True)


    def test_unloaded_truss(self):

        """
        Case 4: A truss without any applied external loads.
        """

        joints = ((0, 1), (1, 0), (1, 1))
        bars = (('AB', strong), ('BC', strong), ('AC', strong))
        loads = []
        supports = (('A', {'support_type': 'pin', 'pin_rotation': 90}), 
                    ('B', {'support_type': 'roller', 'roller_normal_vector': (-1, 2)})
                    )

        tc.create_truss('Completely unloaded truss')
        results = build_from_lists(joints, bars, loads, supports,
                                   sig_figs=3, solution_method=tc.SolveMethod.NUMPY_STD)

        tc.plot_diagram(tc.active_truss, results, show_reactions=True)


    def test_multiple_loads_on_same_joint(self):

        """
        Case 5: A truss with mutliple loads on the same joint which do not cancel out.
        """

        joints = ((0, 1), (1, 0), (1, 1))
        bars = (('AB', strong), ('BC', strong), ('AC', strong))
        loads = [('C', 1, 2), ('C', 0, -1)]
        supports = (('A', {'support_type': 'pin', 'pin_rotation': 90}), 
                    ('B', {'support_type': 'roller', 'roller_normal_vector': (-1, 2)})
                    )

        tc.create_truss('Multiple loads on the same joint')
        results = build_from_lists(joints, bars, loads, supports,
                                   sig_figs=3, solution_method=tc.SolveMethod.NUMPY_STD)

        tc.plot_diagram(tc.active_truss, results, show_reactions=True)


    def test_with_fully_cancelling_loads(self):

        """
        Case 6: A truss with multiple loads on the same joint which do cancel out 
        giving an effectively unloaded truss.
        """

        joints = ((0, 1), (1, 0), (1, 1))
        bars = (('AB', strong), ('BC', strong), ('AC', strong))
        loads = [('C', 1, 2), ('C', -1, -2)]
        supports = (('A', {'support_type': 'pin', 'pin_rotation': 90}), 
                    ('B', {'support_type': 'roller', 'roller_normal_vector': (-1, 2)})
                    )

        tc.create_truss('All external loads cancel out')
        results = build_from_lists(joints, bars, loads, supports,
                                   sig_figs=3, solution_method=tc.SolveMethod.NUMPY_STD)

        tc.plot_diagram(tc.active_truss, results, show_reactions=True)


def build_from_lists(joints: tuple[tuple[float]], bars: tuple[tuple[str, dict]], 
                     loads: list[tuple[str, float, float]], supports: tuple[tuple[str, dict]], **res_kwargs):

    """
    Allows quick construction of full trusses given lists in an appropriate format:

    joints: `((x1, y1), (x2, y2), ...)` named in order by default A, B, C, ...

    bars: `(('AB', strong), ('BC', weak), ...)` bar_name is a two-char string, 
    using the letters from the joints to indicate which ones it goes between

    loads: `[('A', x1, y1), ('C', x2, y2), ...]` joint_name is a one-char string, indicating the loaded joint

    supports: `(('A', kwargs1), ('B', kwargs2), ...)`
    joint_name is a one-char string, indicating the supported joint kwargs can be a dict which fills any of 
    the following:
    `{'support_type': 'pin'/'roller'/'encastre', 'pin_rotation': angle_in_degrees, 'roller_normal_vector': (x, y)}`
    """

    import string

    _alpha = string.ascii_uppercase
    _nums = string.digits
    
    for i, (x, y) in enumerate(joints):
        tc.create_joint('Joint ' + _alpha[i], x, y)
    for i, (c, bar_type) in enumerate(bars):
        tc.create_bar('Bar ' + c, 'Joint ' + c[0], 'Joint ' + c[1], bar_type)
    for i, (j, x, y) in enumerate(loads):
        tc.create_load(r'$W_{' + _nums[i] + r'}$', 'Joint ' + j, x, y)
    for i, (j, kwargs) in enumerate(supports):
        tc.create_support('Support ' + j, 'Joint ' + j, **kwargs)

    try:  # Get the results of the truss calculation and display graphic
        my_results = tc.active_truss.Result(tc.active_truss, **res_kwargs)
        print(my_results)
        return my_results

    except tc.np.linalg.LinAlgError as e:  # The truss was badly made, so could not be solved
        tc.active_truss.classify_error_in_truss(e)


if __name__ == '__main__':

    unittest.main(verbosity=2)
