# Simple Truss Calculator

![Example Truss used in SDC](Media/GitHub/example_screenshot.png)

Calculator with display for finding internal/reaction forces, stresses and strains of a pin-jointed, straight-membered, plane truss. Assumes linear elasticity and small deflections.

100% Python. See `src/truss.py` for the program.


![Truss Calculator CI](https://github.com/lorcan2440/Simple-Truss-Calculator/actions/workflows/main.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/lorcan2440/Simple-Truss-Calculator/badge.svg)](https://coveralls.io/github/lorcan2440/Simple-Truss-Calculator?branch=master)

## How to use

### Installation

Copy the `src/truss.py` and `src/utils.py` files into your working directory.

### Usage example

```python
from truss import Result, init_truss, plot_diagram

# initialise Truss object
my_truss = init_truss('My first truss')

# set the (x, y) locations of the joints - places where bars, loads or supports can be placed
my_truss.add_joints([(0, 0), (290, -90), (815, 127.5), (290, 345), (0, 255), (220.836, 127.5)])

# join some joints together with bars - joints are named 
# 'A', 'B', 'C', ... automatically in the order they were listed
my_truss.add_bars(['AB', 'BC', 'CD', 'DE', 'EF', 'AF', 'DF', 'BF'])

# add a load at the named joint
my_truss.add_loads([('W', 'C', 0, -0.675)])

# add two supports at the named joints
my_truss.add_supports([('A', 'encastre'), ('E', 'pin', -math.pi / 2)])

# solve and plot
my_truss.solve_and_plot()

# get results in text form
results = Result(my_truss)
print(results)
```

For more detailed examples, see the complete test cases (`src/tests/test_complete_examples.py`).

## To-do List and Future Aims

### Create a basic program

  Goal: to make a simple once-per-use calculator.

  * [x] Create the core program using an object-oriented approach
  * [x] Add factory functions to build the objects
  * [x] Make it show the result on a static diagram with `matplotlib`
  * [x] Add unit testing with `PyTest`

### Make a Flask app for a GUI

  Goal: to make an interactive app

### Implement truss optimisation



## Why I made this

The idea for creating this program came from the Structural Design Course (SDC) long lab project in Part IA (1st year) of the Engineering course at the University of Cambridge. I originally wrote about 30 lines of code just to check some calculations of my own, then thought about making it work for any truss so I could improve it. After the lab ended I continued working on it in my spare time just for fun. Maybe someone else can get some use out of it too..?


## Donations

If you found this especially useful, you could consider giving me a cuppa :coffee: :smile:

[<img src="Media/GitHub/paypal_donate_button_transparent.png" width=84, height=50>](https://www.paypal.me/lorcan2440)
&emsp;&emsp;
[<img src="Media/GitHub/buy_me_a_coffee.png" width=99, height=50>](https://www.buymeacoffee.com/lorcan)
