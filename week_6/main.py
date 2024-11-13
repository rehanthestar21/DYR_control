import numpy as np
import time
import os
import matplotlib.pyplot as plt
from simulation_and_control import pb, MotorCommands, PinWrapper,dyn_cancel, feedback_lin_ctrl, SinusoidalReference, CartesianDiffKin, differential_drive_controller_adjusting_bearing
from simulation_and_control import differential_drive_regulation_controller,regulation_polar_coordinates,regulation_polar_coordinate_quat,wrap_angle,velocity_to_wheel_angular_velocity
import pinocchio as pin
from regulator_model import RegulatorModel




def init_simulator(conf_file_name):
    """Initialize simulation and dynamic model."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir)
    
    ext_names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    source_names = ["pybullet"]
    
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names, source_names, False, 0, cur_dir)
    num_joints = dyn_model.getNumberofActuatedJoints()
    
    return sim, dyn_model, num_joints


def main():
    # Configuration for the simulation
    conf_file_name = "pandaconfig.json"  # Configuration file for the robot
    sim,dyn_model,num_joints=init_simulator(conf_file_name)

    # getting time step
    time_step = sim.GetTimeStep()
    current_time = 0

    # initializing MPC
     # Define the matrices
    num_states = num_joints * 2
    num_controls = num_joints
   
    # Horizon length
    N_mpc = 10
    # Initialize the regulator model
    regulator = RegulatorModel(N_mpc, num_states, num_controls, num_states)
    # define system matrices
    regulator.setSystemMatrices(time_step)
    # Define the cost matrices
   
    Qcoeff_joint_pos = [1000] * num_controls
    Qcoeff_joint_vel = [0] * num_controls
    # making one vectro Qcoeff
    Qcoeff = np.hstack((Qcoeff_joint_pos, Qcoeff_joint_vel))
    Rcoeff = [0.0]*num_controls
    
    R = 0.1 * np.eye(num_controls)  # Control input cost matrix
    regulator.setCostMatrices(Qcoeff,Rcoeff)
    
    # data storage
    q_mes_all, qd_mes_all, q_d_all, qd_d_all = [], [], [], []

    # command object
    cmd = MotorCommands()


    while True:


        # Measured state 
        q_mes = sim.GetMotorAngles(0)
        qd_mes = sim.GetMotorVelocities(0)
        qdd_est = sim.ComputeMotorAccelerationTMinusOne(0)
        

        # Figure out what the controller should do next
        # MPC section/ low level controller section ##################################################################
        S_bar, T_bar, Q_bar, R_bar = regulator.propagation_model_regulator_fixed_std()
        H,F = regulator.compute_H_and_F(S_bar, T_bar, Q_bar, R_bar)
        x0_mpc = np.vstack((q_mes, qd_mes))
        x0_mpc = x0_mpc.flatten()
        u_mpc = regulator.compute_solution(x0_mpc, F, H)
        
        # Return the optimal control sequence
        u_mpc = u_mpc[0:num_controls] 
        tau_cmd = dyn_cancel(dyn_model, q_mes, qd_mes, u_mpc)
        cmd.SetControlCmd(tau_cmd, ["torque"]*7)
        sim.Step(cmd, "torque")

        ##### advance simulation ##################################################################
        sim.Step(cmd, "torque")
        time_step = sim.GetTimeStep()

        # Exit logic with 'q' key (unchanged)
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break
        

        # Update current time
        current_time += time_step

    # Plotting 
    #add visualization of final x, y, trajectory and theta
    
    
    
if __name__ == '__main__':
    main()