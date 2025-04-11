import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Define function for solving the system of ODEs
def system_of_odes(vector, t, external_current, alpha, beta):
    x,y = vector

    d_vector = [
        y+x-(x ** 3/3) + external_current,
        -x + alpha - beta * y
    ]
    return d_vector

# Define function to animate the plot
def update(frame):
    x_current_1 = x_sol_1[0:frame + 1]
    y_current_1 = y_sol_1[0:frame + 1]

    x_current_2 = x_sol_2[0:frame + 1]
    y_current_2 = y_sol_2[0:frame + 1]

    FHN_plt_1.set_data(x_current_1, y_current_1)
    FHN_plt_2.set_data(x_current_2, y_current_2)

    return FHN_plt_1, FHN_plt_2

# Define parameters of FitzHugh-Nagumo model
alpha = float(input("Enter a-value between 0.33 and 0.99: ")) #0.65
beta = float(input("Enter b-value between a and 1.00: ")) #0.85
print("-------------")

# Specify starting I (external current) value
I = -3

# Initial starting positions for solution curves
position_0 = [0.0, 1.0]
position_1 = [1.0, 1.0]
time_points = np.linspace(0, 100, 1001)

# Create list system to determine spiral sink versus limit cycle and store tested I values
classification = []
tested_values = []

# Create list to record Hopf bifurcations
bifurcations = []

# Keep testing different I values
while I < 3:

    # Define I-value
    external_current = I

    # Solving for the first position
    positions_1 = odeint(system_of_odes, position_0, time_points, args=(external_current,alpha,beta))
    x_sol_1, y_sol_1 = positions_1[:,0], positions_1[:,1]

    # Solving for second position
    positions_2 = odeint(system_of_odes, position_1, time_points, args=(external_current,alpha,beta))
    x_sol_2, y_sol_2 = positions_2[:,0], positions_2[:,1]

    # Determine convergence (spiral sink) or non-convergence (limit cycle)
    tolerance = 1e-8
    diff_x = np.abs(x_sol_1[-1] - x_sol_2[-1])
    diff_y = np.abs(y_sol_1[-1] - y_sol_2[-1])

    # Check if the final distance is below the tolerance (1 = converges, 0 = doesn't converge)
    if diff_x < tolerance and diff_y < tolerance:
        classification.append(int(1))
        I += 0.01
    else:
        classification.append(int(0))
        I += 0.01

    # Record each tested I value
    tested_values.append(I)

# Look for bifurcation points by comparing the values in the classification list
for i in range(len(classification)):
    if classification[i] - classification[i - 1] != 0:
        print("Bifurcation at I = " + str(tested_values[i]))
        bifurcations.append(tested_values[i])

# Plot our solutions at bifurcations
for i in range(len(bifurcations)):

    # Define I value
    external_current = bifurcations[i]

    # Solving for the first system
    positions_1 = odeint(system_of_odes, position_0, time_points, args=(external_current, alpha, beta))
    x_sol_1, y_sol_1 = positions_1[:, 0], positions_1[:, 1]

    # Solving for second system
    positions_2 = odeint(system_of_odes, position_1, time_points, args=(external_current, alpha, beta))
    x_sol_2, y_sol_2 = positions_2[:, 0], positions_2[:, 1]

    # Generate the plot
    fig, ax = plt.subplots()
    FHN_plt_1, = ax.plot(x_sol_1, y_sol_1, 'green')
    FHN_plt_2, = ax.plot(x_sol_2, y_sol_2, 'blue')

    # Display the animation
    animation = FuncAnimation(fig, update, frames=len(time_points), interval=10)
    plt.show()

# Plot the midpoint between bifurcations
# Define I value
external_current = (bifurcations[-1] + bifurcations[0]) / 2

# Solving for the first system
positions_1 = odeint(system_of_odes, position_0, time_points, args=(external_current, alpha, beta))
x_sol_1, y_sol_1 = positions_1[:, 0], positions_1[:, 1]

# Solving for second system
positions_2 = odeint(system_of_odes, position_1, time_points, args=(external_current, alpha, beta))
x_sol_2, y_sol_2 = positions_2[:, 0], positions_2[:, 1]

# Generate the plot
fig, ax = plt.subplots()
FHN_plt_1, = ax.plot(x_sol_1, y_sol_1, 'green')
FHN_plt_2, = ax.plot(x_sol_2, y_sol_2, 'blue')

# Display the animation
animation = FuncAnimation(fig, update, frames=len(time_points), interval=10)
plt.show()