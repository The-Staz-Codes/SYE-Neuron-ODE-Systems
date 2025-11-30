Title:
SYE-Neuron-ODE-Systems

Description:
This repository contains a series of interactive tools designed to support bifurcation detection and facilitate exploration of the FitzHugh-Nagumo and Hodgkin-Huxley models, systems of ordinary differential equations that model action potentials. It is intended to teach users with an undergraduate-level familiarity with mathematics about how neurons communicate and can be represented mathematically. It includes visualizations for understanding membrane potential, phase portrait, and trace-determinant plane behavior under user-specified parameters and an external current amount that the user can control. While holding other parameters constant, it can determine under which external current values bifurcations occur by constructing a trace-determinant plane and detecting when system classifications change. For the FitzHugh-Nagumo model, bifurcations can be predicted for user-specified parameters using a surrogate model that was trained on a synthetically generated dataset. 

Installation Requirements (Dependencies):
- Python 3.13+
- numpy
- scipy 
- matplotlib
- random
- pandas
- R 4.4.1+
- tidyverse
- here
- mgcv

Repository Contents:
- FHN_2Plot_Visualization.py: Visualization tool that generates a phase portrait (left) and trace-determinant plane (right) side-by-side based on user-specified alpha and beta parameters for the FitzHugh-Nagumo model. Upon running, the user can also specify if they would like to include nullclines, vector arrows, or both on the phase portrait. A slider at the bottom of the program allows users to vary the external current parameter, with the plots updating dynamically.
- FHN_3Plot_Visualization.py: Visualization tool that generates a phase portrait (top left), trace-determinant plane (top right), and membrane potential versus time plot (bottom) based on user-specified alpha and beta parameters for the FitzHugh-Nagumo model. The upper slider allows the user to adjust the time parameter to determine the specific membrane potential at a given time. The lower slider allows users to vary the external current parameter, with the plots updating dynamically.
- FHN_Analysis.qmd: Outlines statistical analysis of a synthetically generated dataset. It includes histograms to understand the spread of external current values where bifurcations occur under randomized parameters for the FitzHugh-Nagumo model. It also contains the surrogate model capable of predicting the second bifurcation for user-specified alpha and beta parameters.
- FHN_Bifurcations.py: Program that determines under which external current values bifurcations in the FitzHugh-Nagumo model occur for user-specified alpha and beta parameters. It primarily accomplishes this by linearizing about equilibrium points with the Jacobian matrix, then extracting eigenvalues for each tested external current value and assigning a system classification at that point. After testing every external current value in its set range, the program identifies where the system changes its classification and labels that point as a bifurcation. 
- FHN_HRUMC_Visualization.py: A condensed version of the FHN_3Plot_Visualization.py program that includes a smaller range of external current values that can be adjusted. It includes a special feature that adjusts the external current to cycle over every possible value once from left to right by pressing the “1” key. It eliminates the need for the user to manually adjust external current values and is useful in presentation settings, such as the Hudson River Undergraduate Mathematics Conference.
- FHN_Simulation.py: This program generates a synthetic dataset for the FitzHugh-Nagumo model by randomizing its parameters, determining bifurcation values, and then storing this information in a dataframe, repeating the process 1000 times. The dataframe is written as a CSV for external export.
- FHN_bifurcation_data.csv: CSV containing the data written by the FHN_Simulation.py program. It includes information on alpha, beta, first bifurcation, second bifurcation, third bifurcation, fourth bifurcation, first Hopf bifurcation, and second Hopf bifurcation for 1000 observations. 
- FitzHugh-Nagumo_Model.py: Visualization tool that generates a phase portrait based on user-specified alpha and beta parameters for the FitzHugh-Nagumo model. A slider at the bottom of the program allows users to vary the external current parameter, with the plot updating dynamically.
- HH_Bifurcations.py: Program that roughly determines which external current values bifurcations may occur for the Hodgkin-Huxley model based on constant starting parameters. It primarily accomplishes this by linearizing about equilibrium points with the Jacobian matrix, then extracting eigenvalues for each tested external current value and assigning a system classification at that point. After testing every external current value in its set range, the program identifies where the system changes its classification and labels that point as a bifurcation. It also displays a membrane potential versus time plot (top) and trace-determinant plane (bottom). The upper slider allows the user to adjust the time parameter to determine the specific membrane potential at a given time. The lower slider allows users to vary the external current parameter injected into the system from 10-40 ms, with plots updating dynamically.
- Hodgkin-Huxley_Model.py: Visualization tool that generates membrane potential versus time (top) and probabilities of ion channel subunits being open versus time (bottom) plots based on an initial voltage of -65 mV and constant rate and capacitance parameters for the Hodgkin-Huxley model. The upper slider allows the user to adjust the time parameter to determine the specific membrane potential and probabilities of open ion channel subunits at a given time. The lower slider allows users to vary the external current parameter injected into the system from 10-40 ms, with plots updating dynamically. 

License:

Copyright (c) 2025 Jason Stasio

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
