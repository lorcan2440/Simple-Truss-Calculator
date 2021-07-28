import Truss_Calculator as tc       # local file import
import timeit, unittest             # builtins


class TrussTests(unittest.TestCase):

    """
    Unit testing framework for the truss calculator.
    Values verified with https://skyciv.com/free-truss-calculator/.
    """

    def test_SDC_truss(self):

        """
        This test represents the truss built in the SDC project.
        """

        weak        = {"b": 12.5,   "t": 0.7,   "D": 5,     "E": 210,   "strength_max": 0.216}
        medium_1    = {"b": 16,     "t": 0.9,   "D": 5,     "E": 210,   "strength_max": 0.216}
        medium_2    = {"b": 16,     "t": 1.1,   "D": 5,     "E": 210,   "strength_max": 0.216}
        strong      = {"b": 19,     "t": 1.1,   "D": 5,     "E": 210,   "strength_max": 0.216}

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
        A truss with multiple loads on different joints.
        """

        # TODO: fix load label location 
        # (determining whether to add pi to angle or not) as one looks weird

        strong      = {"b": 19,     "t": 1.1,   "D": 5,     "E": 210,   "strength_max": 0.216}

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
        A truss with a roller support at an angle.
        """

        strong      = {"b": 19,     "t": 1.1,   "D": 5,     "E": 210,   "strength_max": 0.216}

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
                

def build_from_lists(joints, bars, loads, supports, **res_kwargs):

    """
    Allows quick construction of full trusses given lists in an appropriate format:

    joints:     tuple[tuple[x, y]]                      named in order by default A, B, C, ...

    bars:       tuple[tuple[bar_name, bar_params]]      bar_name is a two-char string, using the letters from 
                                                        the joints to indicate which ones it goes between

    loads:      list[tuple[joint_name, x, y]]           joint_name is a one-char string, indicating the loaded joint

    supports:   tuple[tuple[joint_name, kwargs]]        joint_name is a one-char string, indicating the supported joint
                                                        kwargs can be a dict which fills any of the following:
                                                        {'support_type': pin/roller/encastre, 'pin_rotation': angle,
                                                         'roller_normal_vector': tuple[x, y]}
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

    unittest.main()
