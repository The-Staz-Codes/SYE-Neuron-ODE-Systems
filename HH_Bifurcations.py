from scipy.integrate import odeint
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define constants
Capacitance = 1  # Membrane capacitance (µF/cm²)
G_Na = 120  # Maximum sodium conductance (mS/cm²)
E_Na = 61  # Sodium Nernst potential (mV)
G_K = 36  # Maximum potassium conductance (mS/cm²)
E_K = -95  # Potassium Nernst potential (mV)
G_Leak = 0.3  # Leak conductance (mS/cm²)
E_Leak = -65  # Leak Nernst potential (mV)

# Global cache for storing membrane potential
MP_new = None

# Define functions that create sliders to view specific solutions
def update_I(val):
    global MP_new

    # Update the value of injected current
    I_new = I_slider.val

    # Solve the system using our function and odeint
    new_solution = odeint(hh_system, x0, t, args=(I_new,))

    # Extract the solution for membrane potential from matrix
    MP_new = new_solution[:, 0]

    # Extract specific solutions to display on plot
    new_x_intersect = t_slider.val  # Position of the time slider
    new_specific_MP = np.interp(new_x_intersect, t, MP_new)  # Specific membrane potential (mV)

    # Update the membrane potential
    MP_line.set_data(t, MP_new)
    MP_dot.set_data([new_x_intersect], [new_specific_MP])
    MP_dot.set_label(f'Intersection ({new_x_intersect: .2f}, {new_specific_MP: .2f})')

    # Create mapping (0 to 150 in increments of 0.01 -> 0 to 14999)
    position = np.interp(I_slider.val, [5, 150], [0, 14999])
    position = int(round(position))

    # Set coordinates for TD position (for each equilibrium point)
    x_cor_1 = np.round(TD_path_X_1[position], 3)
    y_cor_1 = np.round(TD_path_Y_1[position], 3)
    dot_TD_1.set_data([x_cor_1], [y_cor_1])
    dot_TD_1.set_label(f'TD Coordinates ({x_cor_1: .2f}, {y_cor_1: .2f})')

    x_cor_2 = np.round(TD_path_X_2[position], 3)
    y_cor_2 = np.round(TD_path_Y_2[position], 3)
    dot_TD_2.set_data([x_cor_2], [y_cor_2])
    dot_TD_2.set_label(f'TD Coordinates ({x_cor_2: .2f}, {y_cor_2: .2f})')

    x_cor_3 = np.round(TD_path_X_3[position], 3)
    y_cor_3 = np.round(TD_path_Y_3[position], 3)
    dot_TD_3.set_data([x_cor_3], [y_cor_3])
    dot_TD_3.set_label(f'TD Coordinates ({x_cor_3: .2f}, {y_cor_3: .2f})')

    # Update legends and bounds
    ax1.legend()
    ax1.set_ylim(np.min(MP_new) - 5, np.max(MP_new) + 5)
    ax2.legend()
    fig.canvas.draw_idle()

def update_t(val):
    # Find new coordinates
    x_new = t_slider.val
    if I_slider.val == 5:
        y_new_MP = np.interp(x_new, t, A)
    else:
        y_new_MP = np.interp(x_new, t, MP_new)

    # Update coordinates
    MP_dot.set_data([x_new], [y_new_MP])

    # Update labels
    MP_dot.set_label(f'Intersection ({x_new: .2f}, {y_new_MP: .2f})')

    # Update legends
    ax1.legend()
    fig.canvas.draw_idle()

# Define function for classifying phase portrait
def classify_state_space(J):
    # Determine eigenvalues
    eigenvalues = np.linalg.eigvals(J)

    # Extract real parts
    real_parts = np.real(eigenvalues)
    real_parts[np.isclose(real_parts, 0, atol=1e-10)] = 0

    # Extract imaginary parts
    imaginary_parts = np.imag(eigenvalues)

    # Determine classification
    if np.all(real_parts > 0) and np.unique(real_parts).size == real_parts.size and np.all(imaginary_parts == 0):
        return 1  # Source
    elif np.all(real_parts < 0) and np.unique(real_parts).size == real_parts.size and np.all(imaginary_parts == 0):
        return 2  # Sink
    elif np.any(real_parts > 0) and np.any(real_parts < 0) and np.all(imaginary_parts == 0):
        return 3  # Saddle
    elif np.any(real_parts > 0) and np.any(imaginary_parts != 0):
        return 4 # Spiral Source
    elif np.any(real_parts < 0) and np.any(imaginary_parts != 0):
        return 5  # Spiral Sink
    elif np.all(real_parts > 0) and np.unique(real_parts).size != real_parts.size and np.all(imaginary_parts == 0):
        return 6 # Degenerate Source
    elif np.all(real_parts < 0) and np.unique(real_parts).size != real_parts.size and np.all(imaginary_parts == 0):
        return 7 # Degenerate Sink
    else:
        return 0 # Other classification

# Define voltage dependent rate parameters as lambda functions
def a_n(V): return (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))
def b_n(V): return 0.125 * np.exp(-0.0125 * (V + 65))

def a_m(V): return (0.1 * (V + 40)) / (1 - np.exp(-0.1 * (V + 40)))
def b_m(V): return 4.0 * np.exp(-0.0556 * (V + 65))

def a_h(V): return 0.07 * np.exp(-0.05 * (V + 65))
def b_h(V): return 1 / (1 + np.exp(-0.1 * (V + 35)))

# Define derivatives of rate parameters as lambda functions
def ddva_n(V): return (
        (0.01 * (1 - np.exp(-0.1 * (V+55))) - 0.001 * (V+55) * np.exp(-0.1 * (V+55))) / ((1-np.exp(-0.1 * (V+55))) ** 2)
)
def ddvb_n(V): return -0.0015625 * np.exp(-(V + 65) / 80)

def ddva_m(V): return (
        (0.1 * (1 - np.exp(-0.1 * (V+40))) - 0.01 * (V+40) * np.exp(-0.1 * (V+40))) / ((1 - np.exp(-0.1 * (V+40))) ** 2)
)
def ddvb_m(V): return -(2 / 9) * np.exp(-(V+65) / 18)

def ddva_h(V): return -0.0035 * np.exp(-(V+65) / 20)
def ddvb_h(V): return (0.1 * np.exp(-0.1 * (V+35))) / ((1 + np.exp(-0.1 * (V+35))) ** 2)

# Define function for solving system algebraically, not over time
def fixed_point_equations(vars, I):
    V, n, m, h = vars
    f = (I - (G_Na * m ** 3 * h * (V - E_Na) +
                     G_K * n ** 4 * (V - E_K) +
                     G_Leak * (V - E_Leak))) / Capacitance
    g = a_n(V) * (1 - n) - b_n(V) * n
    k = a_m(V) * (1 - m) - b_m(V) * m
    s = a_h(V) * (1 - h) - b_h(V) * h
    return [f, g, k, s]

# Define function to obtain fixed point(s) to linearize about
def find_fixed_points(x0, I):
    # Unpack information from vector input
    n0 = x0[1]
    m0 = x0[2]
    h0 = x0[3]

    guesses = [[-70, n0, m0, h0], [-65, n0, m0, h0], [-60, n0, m0, h0]]
    fixed_points = []

    for guess in guesses:
        sol = fsolve(fixed_point_equations, guess, args=(I,))

        # Ensure uniqueness of solutions
        if not any(np.allclose(sol, fp, atol=1e-5) for fp in fixed_points):
            fixed_points.append(sol)

    return fixed_points

# Define function for solving the ODE system
def hh_system(x, t, I):
    # External current value
    if 10 < t < 40:
        I_ext = I
    else:
        I_ext = 0

    # Unpack information from vector input
    V = x[0]
    n = x[1]
    m = x[2]
    h = x[3]

    # Define each ODE in the system
    dVdt = (I_ext - (G_Na * m ** 3 * h * (V - E_Na) +
            G_K * n ** 4 * (V - E_K) +
            G_Leak * (V - E_Leak))) / Capacitance
    dndt = a_n(V) * (1 - n) - b_n(V) * n
    dmdt = a_m(V) * (1 - m) - b_m(V) * m
    dhdt = a_h(V) * (1 - h) - b_h(V) * h

    # Return matrix output
    return [dVdt, dndt, dmdt, dhdt]

# Establish Initial conditions (begin in steady state)
V0 = -65
n0 = a_n(V0) / (a_n(V0) + b_n(V0))
m0 = a_m(V0) / (a_m(V0) + b_m(V0))
h0 = a_h(V0) / (a_h(V0) + b_h(V0))

# Pack information into a vector
x0 = [V0, n0, m0, h0]

# Define initial external current
Ic = 0

# Define time vector
t = np.linspace(0, 50, 5000)

# Create list system to determine classifications for each equilibrium point
classifications_1 = []
classifications_2 = []
classifications_3 = []

# Create list to store tested I-values
tested_values = []

# Create list to record bifurcations
bifurcations = []

# Define lists to draw the path taken in TD plane for each possible equilibrium point
TD_path_X_1 = []
TD_path_Y_1 = []
TD_path_X_2 = []
TD_path_Y_2 = []
TD_path_X_3 = []
TD_path_Y_3 = []

# Keep testing different I values
for I in np.arange(0, 150, 0.01):
    # Begin counter
    counter = 0

    # Record each tested I value
    tested_values.append(I)

    # Find all possible fixed points
    fixed_points = find_fixed_points(x0, I)

    for V, n, m, h in fixed_points:
        # Increase the counter
        counter += 1

        # Linearize about this fixed point
        J = np.array([
            [-G_Na * m ** 3 * h - G_K * n ** 4 - G_Leak, -G_K * 4 * n ** 3 * (V - E_K),
             -G_Na * 3 * m ** 2 * h * (V - E_Na), -G_Na * m ** 3 * (V - E_Na)],
            [ddva_n(V) * (1 - n) - ddvb_n(V) * n, -a_n(V) - b_n(V), 0, 0],
            [ddva_m(V) * (1 - m) - ddvb_m(V) * m, 0, -a_m(V) - b_m(V), 0],
            [ddva_h(V) * (1 - h) - ddvb_h(V) * h, 0, 0, -a_h(V) - b_h(V)]
        ])

        # Calculate Trace
        Trace = -(G_Na * m ** 3 * h + G_K * n ** 4 + G_Leak + a_n(V) + b_n(V) + a_m(V) + b_m(V) + a_h(V) + b_h(V))
        if counter == 1:
            TD_path_X_1.append(Trace)
        elif counter == 2:
            TD_path_X_2.append(Trace)
        elif counter == 3:
            TD_path_X_3.append(Trace)

        # Calculate Determinant
        Determinant = Determinant = (
                ((G_Na * m ** 3 * (V - E_Na)) * (a_n(V) + b_n(V)) * (a_m(V) + b_m(V))
                * (ddva_h(V) * (1 - h) - ddvb_h(V) * h))
                + ((-a_h(V) - b_h(V))
                * ((-G_Na * 3 * m ** 2 * h * (V - E_Na)) * (a_n(V) + b_n(V)) * (ddva_m(V) * (1 - m) - ddvb_m(V) * m)
                + (-a_m(V) - b_m(V))
                * ((G_Na * m ** 3 * h + G_K * n ** 4 + G_Leak) * (a_n(V) + b_n(V))
                + (G_K * 4 * n ** 3 * (V - E_K)) * (ddva_n(V) * (1 - n) - ddvb_n(V) * n))))
        )
        if counter == 1:
            TD_path_Y_1.append(Determinant)
        elif counter == 2:
            TD_path_Y_2.append(Determinant)
        elif counter == 3:
            TD_path_Y_3.append(Determinant)

        # Determine classification
        classification_value = classify_state_space(J)

        # Record each classification
        if counter == 1:
            classifications_1.append(classification_value)
        elif counter == 2:
            classifications_2.append(classification_value)
        elif counter == 3:
            classifications_2.append(classification_value)

    # Ensure classifications and trace-determinant path lists will all be the same size
    if counter == 1:
        TD_path_X_2.append(np.nan)
        TD_path_Y_2.append(np.nan)
        TD_path_X_3.append(np.nan)
        TD_path_Y_3.append(np.nan)
        classifications_2.append(np.nan)
        classifications_3.append(np.nan)
    elif counter == 2:
        TD_path_X_3.append(np.nan)
        TD_path_Y_3.append(np.nan)
        classifications_3.append(np.nan)

# Look for bifurcation points by comparing the values in the classification lists
for i in range(1, len(classifications_1)):
    if not np.isnan(classifications_1[i]) and not np.isnan(classifications_1[i - 1]):
        if classifications_1[i] != classifications_1[i - 1]:
            bifurcations.append(str(np.round(tested_values[i], 4)))
    if not np.isnan(classifications_2[i]) and not np.isnan(classifications_2[i - 1]):
        if classifications_2[i] != classifications_2[i - 1]:
            bifurcations.append(str(np.round(tested_values[i], 4)))
    if not np.isnan(classifications_3[i]) and not np.isnan(classifications_3[i - 1]):
        if classifications_3[i] != classifications_3[i - 1]:
            bifurcations.append(str(np.round(tested_values[i], 4)))
print("Bifurcations at: ")
for i in bifurcations:
    print("I = " + i)

# Solve the system using our function and odeint
solution = odeint(hh_system, x0, t, args=(Ic,))

# Extract the solutions from matrix
A = solution[:,0] # Membrane Potential (mV)
B = solution[:,1]
C = solution[:,2]
D = solution[:,3]

# Extract specific solutions to display on plot
x_intersect = 0 # Initial position of the slider (time in ms)
specific_MP = np.interp(x_intersect, t, A) # Specific membrane potential (mV)

# Generate the base plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)  # Adjust space for slider
plt.gcf().canvas.manager.set_window_title("Hodgkin-Huxley Model")

# Plot the results for change in membrane potential
MP_line, = ax1.plot(t, A, label="Membrane Potential (mV)")
MP_dot, = ax1.plot(x_intersect, specific_MP, 'ro', markersize=5,
               label = f'Intersection ({x_intersect: .2f}, {specific_MP: .2f})', zorder =3)  # Dot at intersection
ax1.set_ylabel("Membrane Potential (mV)")
ax1.set_title("Hodgkin-Huxley Model")
ax1.set_ylim(np.min(A) - 5, np.max(A) + 5)
ax1.legend()
ax1.grid()

# Plot the Trace-Determinant Plane
# Define trace (tr) and determinant (det) ranges
TD_path_X_all = TD_path_X_1 + TD_path_X_2 + TD_path_X_3
TD_path_Y_all = TD_path_Y_1 + TD_path_Y_2 + TD_path_Y_3

trace_values = np.linspace(min(TD_path_X_all) - 0.5, max(TD_path_X_all) + 0.5, 1001)
determinant_values = np.linspace(min(TD_path_X_all) - 0.5, max(TD_path_X_all) + 0.5, 1001)

# Define the curve where eigenvalues transition from real to complex (parabolic boundary)
det_boundary = (trace_values**2) / 4

# Generate the plot
ax2.plot(trace_values, det_boundary, 'k')

# Highlight regions
ax2.fill_between(trace_values, det_boundary, max(TD_path_Y_all) + 0.5, color='lightblue', alpha=0.5)
ax2.fill_between(trace_values, 0, det_boundary, color='yellow', alpha=0.5)
ax2.fill_between(trace_values, det_boundary, min(TD_path_Y_all) - 0.5, color='pink', alpha=0.5)

# Draw the paths
ax2.plot(TD_path_X_1, TD_path_Y_1, linewidth = 1, color = 'blue', alpha = 1)
ax2.plot(TD_path_X_2, TD_path_Y_2, linewidth = 1, color = 'darkgreen', alpha =0.5)
ax2.plot(TD_path_X_3, TD_path_Y_3, linewidth = 1, color = 'red', alpha = 0.5)

# Create dots to traverse the path for equilibrium points
dot_TD_1, = ax2.plot(TD_path_X_1[0], TD_path_Y_1[0], 'ro', markersize=5, alpha = 0.5,
               label = f'TD Coordinates ({TD_path_X_1[0]: .2f}, {TD_path_Y_1[0]: .2f})', zorder = 3)
dot_TD_2, = ax2.plot(TD_path_X_1[0], TD_path_Y_1[0], 'ko', markersize=5, alpha = 0.5,
               label = f'TD Coordinates ({TD_path_X_2[0]: .2f}, {TD_path_Y_2[0]: .2f})', zorder = 3)
dot_TD_3, = ax2.plot(TD_path_X_1[0], TD_path_Y_1[0], 'bo', markersize=5, alpha = 0.5,
               label = f'TD Coordinates ({TD_path_X_3[0]: .2f}, {TD_path_Y_3[0]: .2f})', zorder = 3)

# Style the plot
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel("Trace")
ax2.set_ylabel("Determinant")
ax2.grid()
ax2.legend()
ax2.set_xlim(min(TD_path_X_all) - 0.5, max(TD_path_X_all) + 0.5)
ax2.set_ylim(min(TD_path_Y_all) - 0.5, max(TD_path_Y_all) + 0.5)

# Add a slider to adjust t-value
ax_t_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of slider (left, bottom, width, height)
t_slider = Slider(ax_t_slider, 'Time', t[0], t[-1], valinit=x_intersect)
t_slider.label.set_fontsize(7)

# Add a slider to adjust I-value
ax_I_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position of slider (left, bottom, width, height)
I_slider = Slider(ax_I_slider, 'External Current \n Injected Between 10-40 ms', 0, 150, valinit=0)
I_slider.label.set_fontsize(7)

# Connect sliders to update functions
t_slider.on_changed(update_t)
I_slider.on_changed(update_I)
plt.show()