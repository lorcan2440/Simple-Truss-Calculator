# Simple Truss Calculator

Calculator with display for finding internal/reaction forces, stresses and strains of a pin-jointed, straight-membered, plane truss.

100% Python; intended for personal use only but documented as a module.
Soon I hope to make it more user-friendly and interactive.

Assumes linear elasticity: I will *not* be implementing plastic behaviour or FEA to keep it *simple*.


## Video

[<iframe width="560" height="315" src="https://www.youtube.com/embed/pQN-pnXPaVg" 
title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; 
clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen>
</iframe>]


## To-do List and Future Aims

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
  * [ ] Add examples of module use
  * [ ] Add CI/CD and switch unit testing framework to pytest
  * [ ] Publish as a Python package


* [ ] Convert to a PyQt5 application - v2

  Goal: to make a simple editable truss calculator.

  * [ ] Set up `PyQt5` basics
  * [ ] Make the joints draggable
  * [ ] Make the loads/supports rotatable
  * [ ] Improve the speed of the calculator program if necessary (want 60 fps animations)
  
  
* [ ] Extend the calculations further (no FEA)

  Goal: to provide additional functionality to the results of the calculator.
  
  * [ ] Calculate displacements using virtual work and display the deformed truss
  * [ ] For bars in compression, show their buckled shapes with the option of taking into account an initial imperfection
  * [ ] Add the option of using double-members to avoid buckling
  
  
* [ ] Design a GUI

  Goal: to give all the functionality of the code using buttons and text fields.
  
  * [ ] Add property manager menus for each object when clicked on
  * [ ] Allow the option of either 2D or 3D trusses, using `mpl_toolkits.mplot3d` for 3D display.


* [ ] Implement truss optimisation

  Goal: to find a truss which minimises/maximises a given objective (max failure load, min deflection, min fabrication cost, etc.) subject to certain constraints (number of joints, positions of supports, etc.) and display it given a starting truss from which to make adjustments to.


## Why I made this

The idea for creating this program came from the Structural Design Course (SDC) long lab project in Part IA (1st year) of the Engineering course at the University of Cambridge. I originally wrote about 30 lines of code just to check some calculations of my own, then thought about making it work for any truss so I could improve it. After the lab ended I continued working on it in my spare time just for fun. Maybe someone else can get some use out of it too..?


## Donations

If you found this especially useful, you could consider giving me a cuppa :coffee: :smile:

[<img src="media/paypal_donate_button_transparent.png" width=84, height=50>](https://www.paypal.me/lorcan2440)
&emsp;&emsp;
[<img src="media/buy_me_a_coffee.png" width=99, height=50>](https://www.buymeacoffee.com/lorcan)
