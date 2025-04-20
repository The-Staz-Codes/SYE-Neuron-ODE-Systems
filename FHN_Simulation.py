import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
from random import randrange
import pandas as pd

# Create dictionary that we will add to, then use to create a dataframe based off the simulation
data = {
    'alpha':[],
    'beta':[],
    'first_bifurcation':[],
    'second_bifurcation':[],
    'third_bifurcation':[],
    'fourth_bifurcation':[],
    'first_hopf_bifurcation':[],
    'second_hopf_bifurcation':[]
}

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

# Define function to obtain fixed point to linearize about
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

# Define function for classifying phase portrait
def classify_phase_portrait(J):
    # Determine eigenvalues
    eigenvalues = np.linalg.eigvals(J)

    # Extract real parts
    real_parts = np.real(eigenvalues)
    real_parts[np.isclose(real_parts, 0, atol=1e-10)] = 0

    # Extract imaginary parts
    imaginary_parts = np.imag(eigenvalues)

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

# Helper function to append up to N bifurcations
def safe_append_bifurcations(bif_list, target_key_list, default='N/A'):
    for i, key in enumerate(target_key_list):
        if i < len(bif_list):
            data[key].append(bif_list[i])
        else:
            data[key].append(default)

# Run the simulation
for i in range(1000):
    # Define parameters of FitzHugh-Nagumo model
    alpha = randrange(33, 100) / 100
    lower_beta_bound = int(alpha * 100) + 1
    beta = randrange(lower_beta_bound, 101) / 100

    # Add alpha and beta values to data dictionary
    data['alpha'].append(alpha)
    data['beta'].append(beta)

    # Define time vector
    t = np.linspace(0, 100, 1001)

    # Create list system to determine classifications and store tested I-values
    classifications = []
    Hopf_classification = []
    tested_values = []

    # Create list to record bifurcations
    bifurcations = []

    # Create list to record Hopf_bifurcations
    Hopf_bifurcations = []

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

            # Calculate Determinant
            Determinant = 1 - beta + beta * X ** 2

            # Check if this fixed point is stable (convergence test)
            if trajectory_convergence(fhn_system, t, I, alpha, beta):
                Hopf_classification.append(1)
            else:
                Hopf_classification.append(0)

        # Determine classification
        classification_value = classify_phase_portrait(J)

        # Record each classification
        classifications.append(classification_value)

    # Look for bifurcation points by comparing the values in the classification list and Hopf bifurcation list
    for i in range(1, len(classifications)):
        if (classifications[i] - classifications[i - 1] != 0):
            bifurcations.append(str(np.round(tested_values[i], 4)))
        if (Hopf_classification[i] - Hopf_classification[i - 1] != 0):
            Hopf_bifurcations.append(str(np.round(tested_values[i], 4)))

    # Add bifurcations to the data dictionary
    safe_append_bifurcations(bifurcations, [
        'first_bifurcation', 'second_bifurcation',
        'third_bifurcation', 'fourth_bifurcation'
    ])

    safe_append_bifurcations(Hopf_bifurcations, [
        'first_hopf_bifurcation', 'second_hopf_bifurcation'
    ])

# Write the csv file
df = pd.DataFrame(data)
df.to_csv('FHN_bifurcation_data.csv', index=False)