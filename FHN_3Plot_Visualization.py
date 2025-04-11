import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

# Define function for simulating the FHN system (constant current)
def fhn_system(vector, t, external_current, alpha, beta):
    # Unpack information from vector input
    x = vector[0]
    y = vector[1]

    # Define each ODE in the system
    dXdt = y + x -(x ** 3/3) + external_current
    dYdt = -x + alpha - beta * y

    # Return matrix output
    return [dXdt, dYdt]

# Define function for solving system algebraically, not over time
def fixed_point_equations(vars, I, alpha, beta):
    x, y = vars
    dxdt = y + x - (x ** 3) / 3 + I
    dydt = -x + alpha - beta * y
    return [dxdt, dydt]

# Define function to obtain fixed point(s) to linearize about
def find_fixed_points(I, alpha, beta):
    guesses = [[-2, -2], [-1, -1], [0, 0], [1, 1], [2, 2]]
    fixed_points = []

    for guess in guesses:
        sol = fsolve(fixed_point_equations, guess, args=(I, alpha, beta))

        # Ensure uniqueness of solutions
        if not any(np.allclose(sol, fp, atol=1e-5) for fp in fixed_points):
            fixed_points.append(sol)

    return fixed_points

# Define function for updating I and making associated plot adjustments
def update_I(val):
    global MP_new

    # PLOT 1
    I_new = I_slider.val

    # Solving for the first position
    positions_1 = odeint(fhn_system, position_1, t, args=(I_new, alpha, beta))
    x_sol_1, y_sol_1 = positions_1[:, 0], positions_1[:, 1]

    # Solving for second position
    positions_2 = odeint(fhn_system, position_2, t, args=(I_new, alpha, beta))
    x_sol_2, y_sol_2 = positions_2[:, 0], positions_2[:, 1]

    # Solving for third position
    positions_3 = odeint(fhn_system, position_3, t, args=(I_new, alpha, beta))
    x_sol_3, y_sol_3 = positions_3[:, 0], positions_3[:, 1]

    # Clear the plot
    ax1.clear()

    # Plot new solution curves
    ax1.plot(x_sol_1, y_sol_1, 'green')
    ax1.plot(x_sol_2, y_sol_2, 'blue')
    ax1.plot(x_sol_3, y_sol_3, 'orange')

    # Re-add title and grid
    ax1.set_title("Phase Portrait")
    ax1.grid()

    # PLOT 2
    # Create mapping (-3 to 3 in increments of 0.01 -> 0 to 599)
    position = np.interp(I_slider.val, [-3, 3], [0, 599])
    position = int(round(position))

    # Set coordinates for TD position
    x_cor = np.round(TD_path_X[position], 3)
    y_cor = np.round(TD_path_Y[position], 3)
    dot_TD.set_data([x_cor], [y_cor])
    dot_TD.set_label(f'TD Coordinates ({x_cor: .2f}, {y_cor: .2f})')

    # Display information for the real eigenvalues
    real_x = np.round(Real_X[position], 3)
    real_y = np.round(Real_Y[position], 3)
    dot_real.set_label(f'Real Eigenvalues ({real_x: .2f}, {real_y: .2f})')

    # Display information for the imaginary eigenvalues
    imag_x = np.round(Imaginary_X[position], 3)
    imag_y = np.round(Imaginary_Y[position], 3)
    dot_imag.set_label(f'Imag. Eigenvalues ({imag_x: .2f}, {imag_y: .2f})')

    # Update the legend and refresh the plot
    ax2.legend()
    fig.canvas.draw_idle()

    # PLOT 3
    # Solve the system for the initial position
    solutions_new = odeint(fhn_system, initial_position, t, args=(I_new, alpha, beta))
    MP_new = solutions_new[:, 0]

    # Update MP_line
    MP_line.set_data(t, MP_new)

    # Recalculate dot coordinates at current time slider value
    x_new = t_slider.val
    y_new = np.interp(x_new, t, MP_new)
    MP_dot.set_data([x_new], [y_new])
    MP_dot.set_label(f'Intersection ({x_new: .2f}, {y_new: .2f})')

    # Update legend and refresh the plot
    ax3.set_ylim(min(MP_new) - 0.5, max(MP_new) + 0.5)
    ax3.legend()
    fig.canvas.draw_idle()

# Define function for updating t and making plot 3 adjustments
def update_t(val):
    # Find new coordinates
    x_new = t_slider.val
    if I_slider.val == -3:
        y_new = np.interp(x_new, t, MP)
    else:
        y_new = np.interp(x_new, t, MP_new)
    MP_dot.set_data([x_new], [y_new])

    # Update the label with the new intersection coordinates
    MP_dot.set_label(f'Intersection ({x_new: .2f}, {y_new: .2f})')
    ax3.legend()
    fig.canvas.draw_idle()

#Define parameters of FitzHugh-Nagumo model
alpha = float(input("Enter a-value between 0.33 and 0.99: "))
if not (0.33 <= alpha <= 0.99):
    raise ValueError("a-value must be in specified bounds")
beta = float(input("Enter b-value between a and 1.00: "))
if not (alpha < beta <= 1.0):
    raise ValueError("b-value must be in specified bounds")
print("-------------")

# Global cache for storing membrane potential as it updates (reduces computational load)
MP_new = None

# Specify initial current
I = -3
# Specify initial position
initial_position = [-2.5, 7]

# Initial starting positions for solution curves
position_1 = [0.0, 1.0]
position_2 = [1.0, 1.0]
position_3 = [2.0, 1.0]

# Define time vector
t = np.linspace(0, 100, 5000)

# Define lists to draw the path taken in TD plane
TD_path_X = []
TD_path_Y = []

# Define lists to store real eigenvalues
Real_X = []
Real_Y = []

# Define lists to store imaginary eigenvalues
Imaginary_X = []
Imaginary_Y = []

# Solving for the first solution curve
positions_1 = odeint(fhn_system, position_1, t, args=(I, alpha, beta))
x_sol_1, y_sol_1 = positions_1[:, 0], positions_1[:, 1]

# Solving for second solution curve
positions_2 = odeint(fhn_system, position_2, t, args=(I, alpha, beta))
x_sol_2, y_sol_2 = positions_2[:, 0], positions_2[:, 1]

# Solving for third solution curve
positions_3 = odeint(fhn_system, position_3, t, args=(I, alpha, beta))
x_sol_3, y_sol_3 = positions_3[:, 0], positions_3[:, 1]

# Solve the system for the initial position
solutions = odeint(fhn_system, initial_position, t, args=(I, alpha, beta))
MP = solutions[:,0] # Membrane Potential (mV)

# Extract specific solutions to display on third plot
x_intersect = 0 # Initial position of the slider (time in ms)
specific_MP = np.interp(x_intersect, t, MP) # Specific membrane potential (mV)

# Gather data for different I values
for I in np.arange(-3, 3, 0.01):

    # Find all possible fixed points
    fixed_points = find_fixed_points(I, alpha, beta)

    # Calculate Jacobian matrix
    for X, Y in fixed_points:
        # Linearize about point
        J = np.array([
            [1 - X ** 2, 1],
            [-1, -beta]
        ])

        # Determine eigenvalues
        eigenvalues = np.linalg.eigvals(J)

        # Extract real parts
        real_parts = np.real(eigenvalues)
        Real_X.append(np.round(real_parts[1], 5))
        Real_Y.append(np.round(real_parts[0], 5))

        # Extract imaginary parts
        imaginary_parts = np.imag(eigenvalues)
        Imaginary_X.append(np.round(imaginary_parts[1], 5))
        Imaginary_Y.append(np.round(imaginary_parts[0], 5))

        # Calculate Trace
        Trace = 1 - X ** 2 - beta
        TD_path_X.append(Trace)

        # Calculate Determinant
        Determinant = 1 - beta + beta * X ** 2
        TD_path_Y.append(Determinant)

# Generate the base plot
fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, height_ratios=[1, 1])
plt.gcf().canvas.manager.set_window_title("FitzHugh-Nagumo Model")

# Assign subplots
ax1 = fig.add_subplot(gs[0, 0])  # Top-left
ax2 = fig.add_subplot(gs[0, 1])  # Top-right
ax3 = fig.add_subplot(gs[1, :])  # Bottom row, spans both columns
plt.subplots_adjust(bottom=0.25)

# FIRST PLOT
FHN_plt_1, = ax1.plot(x_sol_1, y_sol_1, 'green')
FHN_plt_2, = ax1.plot(x_sol_2, y_sol_2, 'blue')
FHN_plt_3, = ax1.plot(x_sol_3, y_sol_3, 'orange')
ax1.set_title("Phase Portrait")
ax1.grid()

# SECOND PLOT
# Define trace (tr) and determinant (det) ranges
trace_values = np.linspace(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5, 600)
determinant_values = np.linspace(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5, 600)

# Define the curve where eigenvalues transition from real to complex (parabolic boundary)
det_boundary = (trace_values**2) / 4
ax2.plot(trace_values, det_boundary, 'k')

# Highlight regions
ax2.fill_between(trace_values, det_boundary, max(TD_path_Y) + 0.5, color='lightblue', alpha=0.5)
ax2.fill_between(trace_values, 0, det_boundary, color='yellow', alpha=0.5)
ax2.fill_between(trace_values, det_boundary, min(TD_path_Y) - 0.5, color='pink', alpha=0.5)

# Draw the path
ax2.plot(TD_path_X, TD_path_Y, linewidth = 1)

# Create a dots to traverse the path and track eigenvalues
dot_TD, = ax2.plot(TD_path_X[0], TD_path_Y[0], 'ro', markersize=5,
               label = f'TD Coordinates ({TD_path_X[0]: .2f}, {TD_path_Y[0]: .2f})', zorder = 3)
dot_real, = ax2.plot(Real_X[0], Real_Y[0], 'o', color = 'white', markersize=5,
               label = f'Real Eigenvalues ({Real_X[0]: .2f}, {Real_Y[0]: .2f})', alpha = 0)
dot_imag, = ax2.plot(Imaginary_X[0], Imaginary_Y[0],'o', color='white', markersize=5,
               label = f'Imag. Eigenvalues ({Imaginary_X[0]: .2f}, {Imaginary_Y[0]: .2f})', alpha = 0)

# Style the plot
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)
ax2.set_xlabel("Trace")
ax2.set_ylabel("Determinant")
ax2.set_title("Trace-Determinant Plane")
ax2.grid()
ax2.legend()
ax2.set_xlim(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5)
ax2.set_ylim(min(TD_path_Y) - 0.5, max(TD_path_Y) + 0.5)

# THIRD PLOT
MP_line, = ax3.plot(t, MP, label="Membrane Potential (mV)")
MP_dot, = ax3.plot(x_intersect, specific_MP, 'ko', markersize=5,
               label = f'Intersection ({x_intersect: .2f}, {specific_MP: .2f})', zorder =3)
ax3.set_xlabel("Time (ms)")
ax3.set_ylabel("Membrane Potential (mV)")
ax3.set_title("MP vs t")
ax3.grid()
ax3.legend()
ax3.set_ylim(min(MP) - 0.5, max(MP) + 0.5)

# Add a slider to adjust I-value
ax_I_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
I_slider = Slider(ax_I_slider, "I (External Current)", -3, 3, valinit=-3, valstep=0.01)

# Add a slider to adjust t-value
ax_t_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
t_slider = Slider(ax_t_slider, 'Time (ms)', t[0], t[-1], valinit=x_intersect)

# Connect sliders to update functions
I_slider.on_changed(update_I)
t_slider.on_changed(update_t)
plt.show()