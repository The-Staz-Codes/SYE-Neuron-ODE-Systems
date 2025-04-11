import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define function for solving the system of ODEs
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

# Define function for drawing vector field
def vector_field(v, w, I):
    dv = w + v - (v **3 / 3) + I
    dw = -v + alpha - beta * w
    return dv, dw

# Define function for determining nullclines
def nullclines_finder(I, x):
    nullcline_1 = (x ** 3) / 3 - x - I
    nullcline_2 = (alpha - x) / beta
    return[nullcline_1, nullcline_2]

# Define function for updating I and making associated plot adjustments
def update_I(val):
    # PLOT 1
    I_new = slider.val

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

    # Concatenate the solution arrarys
    x_all = np.concatenate((x_sol_1, x_sol_2, x_sol_3))
    y_all = np.concatenate((y_sol_1, y_sol_2, y_sol_3))

    # Obtain the new range of x values for nullclines
    x_new = np.linspace(np.min(x_all) - 0.5, np.max(x_all) + 0.5)

    # Obtain the new nullcline positions
    nullcline_1_new = nullclines_finder(I_new, x_new)[0]
    nullcline_2_new = nullclines_finder(I_new, x_new)[1]
    nullcline_all_new = np.concatenate([np.array(nullcline_1_new), np.array(nullcline_2_new)])

    if grid_lines == True:
        if nullclines == True:
            # Define the vector field
            v_vals = np.linspace(np.min(x_all) - 0.5, np.max(x_all) + 0.5, 30)
            w_vals = np.linspace(np.min(nullcline_all_new) - 0.5, np.max(nullcline_all_new) + 0.5, 30)
        else:
            # Define the vector field
            v_vals = np.linspace(np.min(x_all) - 0.5, np.max(x_all) + 0.5, 20)
            w_vals = np.linspace(np.min(y_all) - 0.5, np.max(y_all) + 0.5, 20)

        # Compute vector field at current I
        V, W = np.meshgrid(v_vals, w_vals)
        DV, DW = vector_field(V, W, I_new)
        magnitude = np.sqrt(DV ** 2 + DW ** 2)

        # Normalize
        magnitude[magnitude == 0] = 1
        DV_norm = DV / magnitude
        DW_norm = DW / magnitude

        # Plot vector field
        scale = 0.5
        ax1.quiver(V, W, DV_norm * scale, DW_norm, color='gray', alpha=0.5)

    if nullclines == True:
        # Update nullcline displays
        ax1.plot(x_new, nullcline_1_new, color="grey")
        ax1.plot(x_new, nullcline_2_new, color="grey")

    # PLOT 2
    # Create mapping (-4 to 3 in increments of 0.01 -> 0 to 699)
    position = np.interp(slider.val, [-4, 3], [0, 699])
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

#Define parameters of FitzHugh-Nagumo model
alpha = float(input("Enter a-value between 0.33 and 0.99: "))
if not (0.33 <= alpha <= 0.99):
    raise ValueError("a-value must be in specified bounds")
beta = float(input("Enter b-value between a and 1.00: "))
if not (alpha < beta <= 1.0):
    raise ValueError("b-value must be in specified bounds")
grid = input("Turn vector arrows on? (Y or N): ")
nulls = input("Turn nullclines on? (Y or N): ")
print("-------------")

# Decide if vector arrows will be included
if grid in ["Y", "y", "Yes", "yes", "0"]:
    grid_lines = True
else:
    grid_lines = False

# Decide if nullclines will be included
if nulls in ["Y", "y", "Yes", "yes", "0"]:
    nullclines = True
else:
    nullclines = False

# Specify initial current
Ic = -4
# Initial starting positions for solution curves
position_1 = [0.0, 1.0]
position_2 = [1.0, 1.0]
position_3 = [2.0, 1.0]

# Define time vector
t = np.linspace(0, 100, 1001)

# Define lists to draw the path taken in TD plane
TD_path_X = []
TD_path_Y = []

# Define lists to store real eigenvalues
Real_X = []
Real_Y = []

# Define lists to store imaginary eigenvalues
Imaginary_X = []
Imaginary_Y = []

# Solving for the first position
positions_1 = odeint(fhn_system, position_1, t, args=(Ic, alpha, beta))
x_sol_1, y_sol_1 = positions_1[:, 0], positions_1[:, 1]

# Solving for second position
positions_2 = odeint(fhn_system, position_2, t, args=(Ic, alpha, beta))
x_sol_2, y_sol_2 = positions_2[:, 0], positions_2[:, 1]

# Solving for third position
positions_3 = odeint(fhn_system, position_3, t, args=(Ic, alpha, beta))
x_sol_3, y_sol_3 = positions_3[:, 0], positions_3[:, 1]

# Concatenate the solution arrarys
x_all = np.concatenate((x_sol_1, x_sol_2, x_sol_3))
y_all = np.concatenate((y_sol_1, y_sol_2, y_sol_3))

# Gather data for different I values
for I in np.arange(-4, 3, 0.01):

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
plt.subplots_adjust(bottom=0.25)
plt.gcf().canvas.manager.set_window_title("FitzHugh-Nagumo Model")

# FIRST PLOT
FHN_plt_1, = ax1.plot(x_sol_1, y_sol_1, 'green')
FHN_plt_2, = ax1.plot(x_sol_2, y_sol_2, 'blue')
FHN_plt_3, = ax1.plot(x_sol_3, y_sol_3, 'orange')
ax1.set_title("Phase Portrait")
ax1.grid()

# Define range of x values for nullclines
x = np.linspace(np.min(x_all) - 0.5, np.max(x_all) + 0.5, 800)

# Obtain nullcline positions
nullcline_1 = nullclines_finder(Ic, x)[0]
nullcline_2 = nullclines_finder(Ic, x)[1]
nullcline_all = np.concatenate([np.array(nullcline_1), np.array(nullcline_2)])

if grid_lines == True:
    if nullclines == True:
        # Define the vector field
        v_vals = np.linspace(np.min(x_all) - 0.5, np.max(x_all) + 0.5, 30)
        w_vals = np.linspace(np.min(nullcline_all) - 0.5, np.max(nullcline_all) + 0.5, 30)
    else:
        # Define the vector field
        v_vals = np.linspace(np.min(x_all) - 0.5, np.max(x_all) + 0.5, 20)
        w_vals = np.linspace(np.min(y_all) - 0.5, np.max(y_all) + 0.5, 20)

    # Compute vector field at current I
    V, W = np.meshgrid(v_vals, w_vals)
    DV, DW = vector_field(V, W, Ic)
    magnitude = np.sqrt(DV**2 + DW**2)

    # Normalize
    DV_norm = DV / magnitude
    DW_norm = DW / magnitude

    # Draw vector field
    quiver = ax1.quiver(V, W, DV_norm, DW_norm, color='gray', alpha=0.5)

if nullclines == True:
    # Draw nullclines
    ax1.plot(x, nullcline_1, color = "grey")
    ax1.plot(x, nullclines_finder(Ic, x)[1], color = "grey")

# SECOND PLOT
# Define trace (tr) and determinant (det) ranges
trace_values = np.linspace(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5, 600)
determinant_values = np.linspace(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5, 600)

# Define the curve where eigenvalues transition from real to complex (parabolic boundary)
det_boundary = (trace_values**2) / 4
ax2.plot(trace_values, det_boundary, 'k')

# Highlight regions
plt.fill_between(trace_values, det_boundary, max(TD_path_Y) + 0.5, color='lightblue', alpha=0.5)
plt.fill_between(trace_values, 0, det_boundary, color='yellow', alpha=0.5)
plt.fill_between(trace_values, det_boundary, min(TD_path_Y) - 0.5, color='pink', alpha=0.5)

# Draw the path
ax2.plot(TD_path_X, TD_path_Y, linewidth = 1)

# Create a dots to traverse the path and track eigenvalues
dot_TD, = plt.plot(TD_path_X[0], TD_path_Y[0], 'ro', markersize=5,
               label = f'TD Coordinates ({TD_path_X[0]: .2f}, {TD_path_Y[0]: .2f})', zorder = 3)
dot_real, = plt.plot(Real_X[0], Real_Y[0], 'o', color = 'white', markersize=5,
               label = f'Real Eigenvalues ({Real_X[0]: .2f}, {Real_Y[0]: .2f})', alpha = 0)
dot_imag, = plt.plot(Imaginary_X[0], Imaginary_Y[0],'o', color='white', markersize=5,
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

# Add a slider to adjust I-value
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of slider (left, bottom, width, height)
slider = Slider(ax_slider, "", -4, 3, valinit=-4, valstep=0.01)
slider_label = fig.text(0.5, 0.05, "I (External Current)", ha="center", fontsize=12)

# Connect slider to update functions
slider.on_changed(update_I)
plt.show()