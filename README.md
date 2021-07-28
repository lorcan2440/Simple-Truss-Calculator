# Simple Truss Calculator
Calculator and interactive program for finding internal/reaction forces, stresses and strains of a pin-jointed, straight-membered, plane truss.
Intended for personal use only; documented but not in a module form. 
Soon I hope to make it more user-friendly and interactive.

## TODOs

* [x] Create a basic program

  Goal: to make a simple once-per-use calculator.

  * [x] Create the core program using an object-oriented approach
  * [x] Add factory functions to build the objects
  * [x] Make it show the result on a static diagram with `matplotlib`
  * [x] Add unit testing
 
* [ ] Make the diagram interactive

  Goal: to make a simple editable truss calculator.
  
  * [x] Allow creation of multiple trusses
  * [x] Allow joints/bars/etc to be given user-friendly names
  * [ ] or autogenerated names
  * [ ] Make the joints draggable
  * [ ] Make the loads/supports rotatable
  * [ ] Improve the speed of the program if necessary (want 60 fps animations)

* [ ] Design a GUI

  Goal: to give all the functionality of the code using buttons and text fields.
  
  * [ ] Add property manager menus for each object when clicked on


* [ ] Implement truss optimisation

  Goal: to find a truss which minimises/maximises a given constraint (max failure load, min deflection, etc.) and display it given a starting truss from which to make    adjustments to.
