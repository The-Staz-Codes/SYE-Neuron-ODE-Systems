import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import odeint

# Define function for solving the system of ODEs
def fhn_system(vector, t, external_current, alpha, beta):
    x,y = vector

    d_vector = [
        y+x-(x ** 3/3) + external_current,
        -x + alpha - beta * y
    ]
    return d_vector

# Define a function for update the I-value
def update_I(val):
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
    ax.clear()

    # Plot new solution curves
    ax.plot(x_sol_1, y_sol_1, 'green')
    ax.plot(x_sol_2, y_sol_2, 'blue')
    ax.plot(x_sol_3, y_sol_3, 'orange')

    # Re-add title and grid
    ax.set_title("FitzHugh-Nagumo Model")
    ax.grid()

#Define parameters of FitzHugh-Nagumo model
alpha = float(input("Enter a-value between 0.33 and 0.99: "))
beta = float(input("Enter b-value between a and 1.00: "))
print("-------------")

# Specify starting I (external current) value
I = -3

# Initial starting positions for solution curves
position_1 = [0.0, 1.0]
position_2 = [1.0, 1.0]
position_3 = [2.0, 1.0]

# Define time vector
t = np.linspace(0, 100, 1001)

# Solving for the first position
positions_1 = odeint(fhn_system, position_1, t, args=(I,alpha,beta))
x_sol_1, y_sol_1 = positions_1[:,0], positions_1[:,1]

# Solving for second position
positions_2 = odeint(fhn_system, position_2, t, args=(I,alpha,beta))
x_sol_2, y_sol_2 = positions_2[:,0], positions_2[:,1]

# Solving for third position
positions_3 = odeint(fhn_system, position_3, t, args=(I,alpha,beta))
x_sol_3, y_sol_3 = positions_3[:,0], positions_3[:,1]

# Generate the plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
FHN_plt_1, = ax.plot(x_sol_1, y_sol_1, 'green')
FHN_plt_2, = ax.plot(x_sol_2, y_sol_2, 'blue')
FHN_plt_3, = ax.plot(x_sol_3, y_sol_3, 'orange')
ax.set_title("FitzHugh-Nagumo Model")
ax.grid()

# Add a slider to adjust I-value
ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of slider (left, bottom, width, height)
slider = Slider(ax_slider, "", -3, 3, valinit=-3, valstep=0.01)
slider_label = fig.text(0.5, 0.05, "I (External Current)", ha="center", fontsize=12)

# Connect slider to update functions
slider.on_changed(update_I)
plt.show()