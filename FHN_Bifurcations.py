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

# Define function for rigorous test of convergence
def trajectory_convergence(fhn_system, t, I, alpha, beta):
    # Set tolerance level
    tolerance = 1e-6

    # Define multiple initial points in phase space
    points = [
        [-2.0, -1.0], [-1.0, -2.0], [-1.0, -1.0], [-1.0, 0.0],
        [0.0, -1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 2.0]
    ]

    # Solve the system for each initial condition
    solutions = [odeint(fhn_system, p, t, args=(I, alpha, beta)) for p in points]

    # Extract final points of each trajectory
    final_points = np.array([sol[-1] for sol in solutions])

    # Assume divergence if all solutions failed
    if len(final_points) == 0:
        return False

    # Compute pairwise Euclidean distances between all final points
    max_distance = np.max(np.linalg.norm(final_points[:, None, :] - final_points[None, :, :], axis=2))

    # Check if the last 50 time steps oscillate (suggests a limit cycle)
    last_values = np.array([sol[-50:] for sol in solutions])
    std_dev = np.mean(np.std(last_values, axis=1))

    # Check if all final points are within the convergence tolerance
    if max_distance < tolerance and std_dev < tolerance:
        return True
    else:
        return False

# Define function for updating I and viewing associated real eigenvalues on the TD-graph
def update_I(val):
    # Create mapping (-7.5 to 5.5 in increments of 0.01 -> 0 to 1299)
    position = np.interp(slider.val, [-7.5, 5.5], [0, 1299])
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
    ax.legend()
    fig.canvas.draw_idle()

# Define function for classifying phase portrait
def classify_phase_portrait(J):
    # Determine eigenvalues
    eigenvalues = np.linalg.eigvals(J)

    # Extract real parts
    real_parts = np.real(eigenvalues)
    real_parts[np.isclose(real_parts, 0, atol=1e-10)] = 0

    # Extract imaginary parts
    imaginary_parts = np.imag(eigenvalues)

    # Add real parts to the real values lists
    Real_X.append(np.round(real_parts[1], 5))
    Real_Y.append(np.round(real_parts[0], 5))

    # Add imaginary parts to the Imaginary values lists
    Imaginary_X.append(np.round(imaginary_parts[1], 5))
    Imaginary_Y.append(np.round(imaginary_parts[0], 5))

    # Determine classification
    if np.all(real_parts > 0) and real_parts[0] != real_parts[1] and np.all(imaginary_parts == 0):
        return 1  # Source
    elif np.all(real_parts < 0) and real_parts[0] != real_parts[1] and np.all(imaginary_parts == 0):
        return 2  # Sink
    elif real_parts[0] * real_parts[1] < -1e-5 and np.all(imaginary_parts == 0):
        return 3  # Saddle
    elif np.any(real_parts > 0) and np.any(imaginary_parts != 0):
        return 4 # Spiral Source
    elif np.any(real_parts < 0) and np.any(imaginary_parts != 0):
        return 5  # Spiral Sink
    elif np.all(real_parts > 0) and real_parts[0] == real_parts[1] and np.all(imaginary_parts == 0):
        return 6 # Degenerate Source
    elif np.all(real_parts < 0) and real_parts[0] == real_parts[1] and np.all(imaginary_parts == 0):
        return 7 # Degenerate Sink
    else:
        return 0 # Other classification

#Define parameters of FitzHugh-Nagumo model
alpha = float(input("Enter a-value between 0.33 and 0.99: "))
if not (0.33 <= alpha <= 0.99):
    raise ValueError("a-value must be in specified bounds")
beta = float(input("Enter b-value between a and 1.00: "))
if not (alpha < beta <= 1.0):
    raise ValueError("b-value must be in specified bounds")
print("-------------")

# Specify initial conditions
I = -5 # starting I (external current) value
vector = [0.0, 0.0] # Starting position

# Define time vector
t = np.linspace(0, 100, 1001)

# Create list system to determine classifications and store tested I-values
classifications = []
tested_values = []

# Create list to record bifurcations
bifurcations = []

# Create list to record Hopf_bifurcations
Hopf_bifurcations = []

# Define lists to draw the path taken in TD plane
TD_path_X = []
TD_path_Y = []

# Define lists to store real eigenvalues
Real_X = []
Real_Y = []

# Define lists to store imaginary eigenvalues
Imaginary_X = []
Imaginary_Y = []

# Keep testing different I values
for I in np.arange(-7.5, 5.5, 0.01):

    # Record each tested I value
    tested_values.append(I)

    # Find all possible fixed points
    fixed_points = find_fixed_points(I, alpha, beta)

    # Default classification if no valid fixed point is found
    # (Assume system is unstable or non-convergent/divergent)
    classification_value = 0

    for X, Y in fixed_points:
        # Linearize about point
        J = np.array([
            [1 - X ** 2, 1],
            [-1, -beta]
        ])

        # Calculate Trace
        Trace = 1 - X ** 2 - beta
        TD_path_X.append(Trace)

        # Calculate Determinant
        Determinant = 1 - beta + beta * X ** 2
        TD_path_Y.append(Determinant)

        # Check if this fixed point is stable (convergence test)
        if trajectory_convergence(fhn_system, t, I, alpha, beta):
            Hopf_bifurcations.append(1)
        else:
            Hopf_bifurcations.append(0)

        # Determine classification
        classification_value = classify_phase_portrait(J)

        # Record each classification
        classifications.append(classification_value)

# Look for bifurcation points by comparing the values in the classification list and Hopf bifurcation list
for i in range(1, len(classifications)):
    if (classifications[i] - classifications[i - 1] != 0):
        bifurcations.append(str(np.round(tested_values[i], 4)))
    if (Hopf_bifurcations[i] - Hopf_bifurcations[i - 1] != 0):
        bifurcations.append(str(np.round(tested_values[i], 4)) + ' *')
print("Bifurcations at: ")
for i in bifurcations:
    print("I = " + i)

# Convey information about classifications
classification_count = 0
print("-------------")
for i in range(1, len(classifications)):
    if (classifications[i] - classifications[i - 1] == 0):
        classification_count += 1
    else:
        if classifications[i - 1] == 1:
            print("Source: " + str(classification_count + 1) + " ticks")
        elif classifications[i - 1] == 2:
            print("Sink: " + str(classification_count + 1) + " ticks")
        elif classifications[i - 1] == 3:
            print("Saddle: " + str(classification_count + 1) + " ticks")
        elif classifications[i - 1] == 4:
            print("Spiral Source: " + str(classification_count + 1) + " ticks")
        elif classifications[i - 1] == 5:
            print("Spiral Sink: " + str(classification_count + 1) + " ticks")
        elif classifications[i - 1] == 6:
            print("Degenerate Source: " + str(classification_count + 1) + " ticks")
        elif classifications[i - 1] == 7:
            print("Degenerate Sink: " + str(classification_count + 1) + " ticks")
        else:
            print("Unstable: " + str(classification_count + 1) + " ticks")
        # Reset count
        classification_count = 0
#Ensure the final unchanged position is added
if classifications[-1] == 1:
    print("Source: " + str(classification_count + 1) + " ticks")
elif classifications[-1] == 2:
    print("Sink: " + str(classification_count + 1) + " ticks")
elif classifications[-1] == 3:
    print("Saddle: " + str(classification_count + 1) + " ticks")
elif classifications[-1] == 4:
    print("Spiral Source: " + str(classification_count + 1) + " ticks")
elif classifications[-1] == 5:
    print("Spiral Sink: " + str(classification_count + 1) + " ticks")
elif classifications[-1] == 6:
    print("Degenerate Source: " + str(classification_count + 1) + " ticks")
elif classifications[-1] == 7:
    print("Degenerate Sink: " + str(classification_count + 1) + " ticks")
else:
    print("Unstable: " + str(classification_count + 1) + " ticks")

# Draw the trace-determinant plane and plot the specific path our system travels
# Define trace (tr) and determinant (det) ranges
trace_values = np.linspace(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5, 1001)
determinant_values = np.linspace(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5, 1001)

# Define the curve where eigenvalues transition from real to complex (parabolic boundary)
det_boundary = (trace_values**2) / 4

# Generate the plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25)
ax.plot(trace_values, det_boundary, 'k')

# Highlight regions
plt.fill_between(trace_values, det_boundary, max(TD_path_Y) + 0.5, color='lightblue', alpha=0.5)
plt.fill_between(trace_values, 0, det_boundary, color='yellow', alpha=0.5)
plt.fill_between(trace_values, det_boundary, min(TD_path_Y) - 0.5, color='pink', alpha=0.5)

# Draw the path
ax.plot(TD_path_X, TD_path_Y, linewidth = 1)

# Create a dot to traverse the path
dot_TD, = plt.plot(TD_path_X[0], TD_path_Y[0], 'ro', markersize=5,
               label = f'TD Coordinates ({TD_path_X[0]: .2f}, {TD_path_Y[0]: .2f})', zorder = 3)
dot_real, = plt.plot(TD_path_X[0], TD_path_Y[0], 'o', color = 'white', markersize=5,
               label = f'Real Eigenvalues ({Real_X[0]: .2f}, {Real_Y[0]: .2f})', alpha = 0)
dot_imag, = plt.plot(Imaginary_X[0], Imaginary_Y[0],'o', color='white', markersize=5,
               label = f'Imag. Eigenvalues ({Imaginary_X[0]: .2f}, {Imaginary_Y[0]: .2f})', alpha = 0)

# Style the plot
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.set_xlabel("Trace")
ax.set_ylabel("Determinant")
ax.set_title("Trace-Determinant Plane")
ax.grid()
ax.legend()
ax.set_xlim(min(TD_path_X) - 0.5, max(TD_path_X) + 0.5)
ax.set_ylim(min(TD_path_Y) - 0.5, max(TD_path_Y) + 0.5)

# Add a slider to adjust I-value
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of slider (left, bottom, width, height)
slider = Slider(ax_slider, "I (External Current)", -7.5, 5.5, valinit=-7.5, valstep=0.01)

# Connect slider to update functions
slider.on_changed(update_I)
plt.show(block = True)