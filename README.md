# Simple Truss Calculator
Calculator with display for finding internal/reaction forces, stresses and strains of a pin-jointed, straight-membered, plane truss.
Intended for personal use only but documented as a module.
Soon I hope to make it more user-friendly and interactive.

## TODOs

* [x] Create a basic program - v0

  Goal: to make a simple once-per-use calculator.

  * [x] Create the core program using an object-oriented approach
  * [x] Add factory functions to build the objects
  * [x] Make it show the result on a static diagram with `matplotlib`
  * [x] Add unit testing
 
* [ ] Make the program more generalised - v1

  Goal: to make it easier to work in a more modular form.
  
  * [x] Allow creation of multiple trusses
  * [x] Allow joints/bars/etc to be given user-friendly names
  * [x] Allow autogenerated names
  * [ ] Allow exporting and loading of complete truss objects in JSON file format

* [ ] Convert to a PyQt5 application - v2

  Goal: to make a simple editable truss calculator.

  * [ ] Set up Qt5 basics
  * [ ] Make the joints draggable
  * [ ] Make the loads/supports rotatable
  * [ ] Improve the speed of the calculator program if necessary (want 60 fps animations)

* [ ] Design a GUI

  Goal: to give all the functionality of the code using buttons and text fields.
  
  * [ ] Add property manager menus for each object when clicked on


* [ ] Implement truss optimisation

  Goal: to find a truss which minimises/maximises a given objective (max failure load, min deflection, min fabrication cost, etc.) subject to certain constraints (number of joints, positions of supports, etc.) and display it given a starting truss from which to make    adjustments to.


## Donations

If you found this especially useful, you could consider giving me a cuppa :coffee: :smile:

[<img src="media/paypal_donate_button_transparent.png" width=84, height=50>](https://www.paypal.me/lorcan2440)