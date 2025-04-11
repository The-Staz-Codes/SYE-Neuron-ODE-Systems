from scipy.integrate import odeint
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

# Global cache for storing membrane potential and probabilities as they update (reduces computational load)
A_new = None
B_new = None
C_new = None
D_new = None

# Define functions that create sliders to view specific solutions
def update_I(val):
    global A_new, B_new, C_new, D_new

    # Update the value of injected current
    I_new = I_slider.val

    # Solve the system using our function and odeint
    new_solution = odeint(hh_system, x0, t, args=(I_new,))

    # Extract the solutions from matrix
    A_new = new_solution[:, 0]
    B_new = new_solution[:, 1]
    C_new = new_solution[:, 2]
    D_new = new_solution[:, 3]

    # Extract specific solutions to display on plot
    new_x_intersect = t_slider.val  # Position of the time slider
    new_specific_MP = np.interp(new_x_intersect, t, A_new)  # Specific membrane potential (mV)
    new_specific_n = np.interp(new_x_intersect, t, B_new)
    new_specific_m = np.interp(new_x_intersect, t, C_new)
    new_specific_h = np.interp(new_x_intersect, t, D_new)

    # Update the plots
    # Curves
    MP_line.set_data(t, A_new)
    n_line.set_data(t, B_new)
    m_line.set_data(t, C_new)
    h_line.set_data(t, D_new)

    # Dots
    dot.set_data([new_x_intersect], [new_specific_MP])
    n_dot.set_data([new_x_intersect], [new_specific_n])
    m_dot.set_data([new_x_intersect], [new_specific_m])
    h_dot.set_data([new_x_intersect], [new_specific_h])
    dot.set_label(f'Intersection ({new_x_intersect: .2f}, {new_specific_MP: .2f})')
    n_dot.set_label(f'Intersection ({new_x_intersect: .2f}, {new_specific_n: .2f})')
    m_dot.set_label(f'Intersection ({new_x_intersect: .2f}, {new_specific_m: .2f})')
    h_dot.set_label(f'Intersection ({new_x_intersect: .2f}, {new_specific_h: .2f})')

    # Update legends
    ax1.legend()
    ax2.legend(handles=[n_dot, m_dot, h_dot], loc='lower right')
    fig.canvas.draw_idle()

def update_t(val):
    # Find new coordinates
    x_new = t_slider.val
    if I_slider.val == 5:
        y_new_MP = np.interp(x_new, t, A)
        y_new_n = np.interp(x_new, t, B)
        y_new_m = np.interp(x_new, t, C)
        y_new_h = np.interp(x_new, t, D)
    else:
        y_new_MP = np.interp(x_new, t, A_new)
        y_new_n = np.interp(x_new, t, B_new)
        y_new_m = np.interp(x_new, t, C_new)
        y_new_h = np.interp(x_new, t, D_new)

    # Set x-positions of the vertical lines
    vline1.set_xdata([x_new])
    vline2.set_xdata([x_new])

    # Update coordinates
    dot.set_data([x_new], [y_new_MP])
    n_dot.set_data([x_new], [y_new_n])
    m_dot.set_data([x_new], [y_new_m])
    h_dot.set_data([x_new], [y_new_h])

    # Update labels
    dot.set_label(f'Intersection ({x_new: .2f}, {y_new_MP: .2f})')
    n_dot.set_label(f'n Intersection ({x_new: .2f}, {y_new_n: .2f})')
    m_dot.set_label(f'm Intersection ({x_new: .2f}, {y_new_m: .2f})')
    h_dot.set_label(f'h Intersection ({x_new: .2f}, {y_new_h: .2f})')

    # Update legends
    ax1.legend()
    ax2.legend(handles=[n_dot, m_dot, h_dot], loc='lower right')
    fig.canvas.draw_idle()

# Define voltage dependent rate parameters as lambda functions
def a_n(V): return (0.01 * (V + 55)) / (1 - np.exp(-0.1 * (V + 55)))
def b_n(V): return 0.125 * np.exp(-0.0125 * (V + 65))

def a_m(V): return (0.1 * (V + 40)) / (1 - np.exp(-0.1 * (V + 40)))
def b_m(V): return 4.0 * np.exp(-0.0556 * (V + 65))

def a_h(V): return 0.07 * np.exp(-0.05 * (V + 65))
def b_h(V): return 1 / (1 + np.exp(-0.1 * (V + 35)))

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

# Define time vector
t = np.linspace(0, 50, 5000)

# Define initial external current
Ic = 5

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

# Do the same for the sub-unit probabilities
specific_n = np.interp(x_intersect, t, B)
specific_m = np.interp(x_intersect, t, C)
specific_h = np.interp(x_intersect, t, D)

# Generate the base plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)  # Adjust space for slider
plt.gcf().canvas.manager.set_window_title("Hodgkin-Huxley Model")

# Plot the results for change in membrane potential
MP_line, = ax1.plot(t, A, label="Membrane Potential (mV)")
vline1 = ax1.axvline(x=x_intersect, color='black', linestyle='--', linewidth=1)
dot, = ax1.plot(x_intersect, specific_MP, 'ro', markersize=5,
               label = f'Intersection ({x_intersect: .2f}, {specific_MP: .2f})', zorder =3)  # Dot at intersection
ax1.set_ylabel("Membrane Potential (mV)")
ax1.set_title("Hodgkin-Huxley Model")
ax1.legend()
ax1.grid()

# Plot the results for change in probability of open subunits
n_line, = ax2.plot(t, B, label = 'n subunit', color = 'purple')
m_line, = ax2.plot(t, C, label = 'm subunit', color = 'orange')
h_line, = ax2.plot(t, D, label = 'h subunit', color = 'green')
vline2 = ax2.axvline(x=x_intersect, color='black', linestyle='--', linewidth=1)
n_dot, = ax2.plot(x_intersect, specific_n, 'ro', markersize=5,
               label = f'n Intersection ({x_intersect: .2f}, {specific_n: .2f})', zorder =3)
m_dot, = ax2.plot(x_intersect, specific_m, 'ko', markersize=5,
               label = f'm Intersection ({x_intersect: .2f}, {specific_m: .2f})', zorder =3)
h_dot, = ax2.plot(x_intersect, specific_h, 'bo', markersize=5,
               label = f'h Intersection ({x_intersect: .2f}, {specific_h: .2f})', zorder =3)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel('Probability')

# Add separate legends, the legend for subunit curves will remain the same and can be placed once
legend1 = ax2.legend(handles=[ax2.get_lines()[0], ax2.get_lines()[1], ax2.get_lines()[2]],
                    loc='upper right') # First legend for subunit curves
ax2.add_artist(legend1)
ax2.legend(handles=[n_dot, m_dot, h_dot], loc='lower right') # Second legend for intersection points
ax2.grid()

# Add a slider to adjust t-value
ax_t_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # Position of slider (left, bottom, width, height)
t_slider = Slider(ax_t_slider, 'Time', t[0], t[-1], valinit=x_intersect)
t_slider.label.set_fontsize(7)

# Add a slider to adjust I-value
ax_I_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Position of slider (left, bottom, width, height)
I_slider = Slider(ax_I_slider, 'External Current \n Injected Between 10-40 ms', 5, 100, valinit=5)
I_slider.label.set_fontsize(7)

# Connect sliders to update functions
t_slider.on_changed(update_t)
I_slider.on_changed(update_I)
plt.show()