import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants for the pendulum
g = 9.81  # gravitational acceleration (m/s^2)
L = 1.0   # length of the pendulum (m)

# Differential equation for the simple pendulum
def pendulum_ode(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Initial conditions: theta = π/4 radians, omega = 0 rad/s
initial_conditions = [np.pi / 4, 0]

# Time span for the simulation
t_span = (0, 10)  # 10 seconds
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solving the ODE
solution = solve_ivp(pendulum_ode, t_span, initial_conditions, t_eval=t_eval)

# Extracting results
theta = solution.y[0]
omega = solution.y[1]
time = solution.t

# Plotting the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(time, theta, label='Theta (rad)')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.title('Pendulum Angle')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time, omega, label='Omega (rad/s)')
plt.xlabel('Time (s)')
plt.ylabel('Omega (rad/s)')
plt.title('Angular Velocity')
plt.legend()

plt.tight_layout()
plt.show()

# Output data for pysindy
time, theta, omega


import numpy as np
from pysindy import SINDy

# Prepare the data for sindy
time_series_data = np.vstack([theta, omega]).T  # theta and omega from your simulation

# Instantiate and configure the SINDy model
model = SINDy()
model.fit(time_series_data, t=time)

# Print the discovered differential equations
discovered_equations = model.equations()
print(discovered_equations)
