import numpy as np
import matplotlib.pyplot as plt
from observer import Observer
from dc_model import SysDyn
               
# Motor Parameters
J = 0.01      # Inertia (kg*m^2)
b = 0.1       # Friction coefficient (N*m*s)
K_t = 1    # Motor torque constant (N*m/A)
K_e = 0.01    # Back EMF constant (V*s/rad)
R_a = 1.0     # Armature resistance (Ohm)
L_a = 0.001   # Armature inductance (H)

# Desired Eigenvalues for Observer
lambda_1 = -11
lambda_2 = -20

# Simulation Parameters
t_start = 0.0
t_end = 0.05
dt = 0.00001  # Smaller time step for Euler integration
time = np.arange(t_start, t_end, dt)
num_steps = len(time)

# Initial Conditions for the System [omega, I_a]
x_init = np.array([0.0, 0.0])  # True system state [omega, I_a]
motor_model = SysDyn(J, b, K_t, K_e, R_a, L_a,dt, x_init)

# Initial Conditions for the Observer [omega_hat, I_a_hat]
x_hat_init = np.array([0.0, 0.0])  # Initial guess for the observer state [omega_hat, I_a_hat]
observer = Observer(motor_model.A, motor_model.B, motor_model.C,dt, x_hat_init)

# Compute the observer gain L
# Place the eigenvalues of (A - L C) at desired locations
observer.ComputeObserverGains(lambda_1, lambda_2)


# Input Voltage (Armature voltage V_a)
V_a = 12.0  # Volts

# Preallocate arrays for storing results
omega = np.zeros(num_steps)
I_a = np.zeros(num_steps)
hat_omega = np.zeros(num_steps)
hat_I_a = np.zeros(num_steps)
T_m_true = np.zeros(num_steps)
T_m_estimated = np.zeros(num_steps)
V_terminal = np.zeros(num_steps)
V_terminal_hat = np.zeros(num_steps)

# Simulation loop using Euler integration
for k in range(num_steps):
    # Time stamp
    t = time[k]
    
    cur_y = motor_model.step(V_a)
    x_cur = motor_model.getCurrentState()
    
    # Output measurement (Terminal Voltage)
    V_terminal[k] = cur_y
    
    x_hat_cur,y_hat_cur = observer.update(V_a, cur_y)

    # Store results
    omega[k] = x_cur[0]
    I_a[k] = x_cur[1]
    hat_omega[k] = x_hat_cur[0]
    hat_I_a[k] = x_hat_cur[1]
    T_m_true[k] = K_t * I_a[k]
    T_m_estimated[k] = K_t * hat_I_a[k]
    V_terminal_hat[k] = y_hat_cur

# Plotting the results
plt.figure(figsize=(12, 10))

# Angular velocity
plt.subplot(4, 1, 1)
plt.plot(time, omega, label='True $\omega$ (rad/s)')
plt.plot(time, hat_omega, '--', label='Estimated $\hat{\omega}$ (rad/s)')
plt.title('Angular Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# Armature current
plt.subplot(4, 1, 2)
plt.plot(time, I_a, label='True $I_a$ (A)')
plt.plot(time, hat_I_a, '--', label='Estimated $\hat{I}_a$ (A)')
plt.title('Armature Current')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.legend()
plt.grid(True)

# Torque
plt.subplot(4, 1, 3)
plt.plot(time, T_m_true, label='True $T_m$ (N*m)')
plt.plot(time, T_m_estimated, '--', label='Estimated $\hat{T}_m$ (N*m)')
plt.title('Motor Torque')
plt.xlabel('Time (s)')
plt.ylabel('Torque (N*m)')
plt.legend()
plt.grid(True)

# Terminal Voltage
plt.subplot(4, 1, 4)
plt.plot(time, V_terminal, label='Measured $V_{terminal}$ (V)')
plt.plot(time, V_terminal_hat, '--', label='Estimated $\hat{V}_{terminal}$ (V)')
plt.title('Terminal Voltage')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
