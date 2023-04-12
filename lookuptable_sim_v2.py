import numpy as np
from scipy.integrate import odeint
import math
import pacejka_regress
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib 
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import Axes3D
import pickle
import rospy

def pacejka_model(alpha, mu, Fz, B, C, D, E):
    Fy_hat = mu * Fz * D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))
    return Fy_hat

def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """
    Acceleration constraints, adjusts the acceleration based on constraints

        Args:
            vel (float): current velocity of the vehicle
            accl (float): unconstraint desired acceleration
            v_switch (float): switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max (float): maximum allowed acceleration
            v_min (float): minimum allowed velocity
            v_max (float): maximum allowed velocity

        Returns:
            accl (float): adjusted acceleration
    """

    # positive accl limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # accl limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit
    # print("accel cmd: ", accl)
    return accl

def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """
    Steering constraints, adjusts the steering velocity based on constraints

        Args:
            steering_angle (float): current steering_angle of the vehicle
            steering_velocity (float): unconstraint desired steering_velocity
            s_min (float): minimum steering angle
            s_max (float): maximum steering angle
            sv_min (float): minimum steering velocity
            sv_max (float): maximum steering velocity

        Returns:
            steering_velocity (float): adjusted steering velocity
    """

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max
    # print("steering_velocity cmd: ", steering_velocity)
    return steering_velocity

def vehicle_dynamics_st(x, u, mu, lf, lr, h, m, I, Bf ,Cf ,Df ,Ef, Br, Cr, Dr ,Er, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max,des_vel,des_steer):
    """
    Single Track Dynamic Vehicle Dynamics.

        Args:
            x (numpy.ndarray (3, )): vehicle state vector (x1, x2, x3, x4, x5, x6, x7)
                x1: x position in global coordinates
                x2: y position in global coordinates
                x3: yaw angle
                x4: velocity in x direction
                x5: velocity in y direction
                x6: yaw_rate
                x7: steering angle

            u (numpy.ndarray (2, )): control input vector (u1, u2)
                u1: steering angle velocity of front wheels
                u2: longitudinal acceleration

        Returns:
            f (numpy.ndarray): right hand side of differential equations
    """

    # gravity constant m/s^2
    g = 9.81
    Px=x[0]
    Py=x[1]
    # if x[2] > 360:
    #     x[2] = x[2]-360
    # if x[2] < -360:
    #     x[2]= x[2]+360
    psi=x[2]
    Vx=x[3]
    Vy=x[4]
    psidot=x[5]
    steer =x[6]
    steerdot=u[0]
    Ax=u[1]

    # arr = x
    # for element in arr:
    #     print(element)
    # print()

    # system dynamics
    # alphaf = math.atan2(Vy + lf * psidot, Vx) - steer
    alphaf = -1*(math.atan2(Vy + lf * psidot, Vx) - steer)
    # alphar = math.atan2(Vy - lr * psidot, Vx)
    alphar = -1*math.atan2(Vy - lr * psidot, Vx)
    Fzf = (m*g*lr - m*Ax*h) / (lr+lf)
    Fzr = (m*g*lf + m*Ax*h) / (lr+lf)
    Fyf = pacejka_model(alphaf, mu, Fzf, Bf, Cf, Df, Ef) 
    Fyr = pacejka_model(alphar, mu, Fzr, Br, Cr, Dr, Er)
    # print("Fyf: ", Fyf, "Fyr: ", Fyr)
    # print(alphaf,Fzf, Fyf)
    # alpha = math.atan2(math.tan(u[0])*lr/(lr+lf))
    alpha = math.atan2(Vy,Vx)
    f = np.array(
        [Vx*math.cos(psi)- Vy*math.sin(psi), #0
        Vx*math.sin(psi)+ Vy*math.cos(psi), #1
        psidot, #2
        Ax+(1/m)*( -1*Fyf*math.sin(steer) + m*Vy*psidot), # Ax, #3
        (1/m)*(Fyr+Fyf*math.cos(steer)-m*Vx*psidot), # (1/m)*(Fyr+Fyf)-Vx*psidot,#4
        (1/I)*(Fyf*lf*math.cos(steer)-Fyr*lr), #5
        steerdot #6
        ])
    
    return f

def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """
    Basic controller for speed/steer -> accl./steer vel.

        Args:
            speed (float): desired input speed
            steer (float): desired input steering angle

        Returns:
            accl (float): desired input acceleration
            sv (float): desired input steering velocity
    """
    # print("speed, steer, current_speed, current_steer: ",speed, steer, current_speed, current_steer)
    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    # accl
    vel_diff = speed - current_speed
    # currently forward
    if current_speed > 0.:
        if (vel_diff > 0):
            # accelerate
            kp = 120.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 120.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # currently backwards
    else:
        if (vel_diff > 0):
            # braking
            kp = 122.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 122.0 * max_a / (-min_v)
            accl = kp * vel_diff
    # print("sv, accel from pid: ", sv, accl)
    return sv, accl

def func_ST(x, t, u, mu, lf, lr, h, m, I, Bf ,Cf ,Df ,Ef, Br, Cr, Dr ,Er, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max,des_vel,des_steer):
    u1 = pid(des_vel, des_steer, x[3], x[6], sv_max, a_max, v_max, v_min)
    # print("u1: ",u1[0], u1[1])
    u1 = np.array([steering_constraint(x[6], u1[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u1[1], v_switch, a_max, v_min, v_max)])
    # print("u1 constraned: ",u1[0], u1[1])
    f = vehicle_dynamics_st(x, u1, mu, lf, lr, h, m, I, Bf ,Cf ,Df ,Ef, Br, Cr, Dr ,Er, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max,des_vel,des_steer)
    return f

def update(frame, ax, trajectory):
    x = frame[0]
    y = frame[1]
    heading = frame[2]
    steer_angle = frame[3]

    # Add the current position to the trajectory
    trajectory.append([x, y])

    # Clear the previous plot
    ax.cla()

    # Draw the car
    car_width = 0.05
    car_height = 0.1
    wheel_radius = 0.015
    wheel_width = 0.01

   # Define the rotation matrix
    rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    rotation_matrix_wheel = np.array([[np.cos(heading+steer_angle), -np.sin(heading+steer_angle)], [np.sin(heading+steer_angle), np.cos(heading+steer_angle)]])
    # print(heading, steer_angle)
    # Define the translation matrix
    translation_matrix = np.array([[x], [y]])
    translation_matrix_wheel = np.array([[x+car_height/2], [y+car_width/2]])
    # Define the car rectangle vertices
    car_rect_vertices = np.array([[-car_height/2, -car_width/2], [car_height/2, -car_width/2], [car_height/2, car_width/2], [-car_height/2, car_width/2]])
    wheel_rect_vertices = np.array([[-wheel_radius/2, -wheel_width/2], [wheel_radius/2, -wheel_width/2], [wheel_radius/2, wheel_width/2], [-wheel_radius/2, wheel_width/2]])
    # Apply the rotation and translation
    car_rect_vertices = np.dot(rotation_matrix, car_rect_vertices.T).T + translation_matrix.T
    wheel_rect_vertices = np.dot(rotation_matrix_wheel, wheel_rect_vertices.T).T + translation_matrix_wheel.T
    # Draw the rotated car rectangle
    rect = plt.Polygon(car_rect_vertices, facecolor='black', edgecolor='black')
    rect_wheel = plt.Polygon(wheel_rect_vertices, facecolor='gray', edgecolor='gray')
    ax.add_patch(rect)
    ax.add_patch(rect_wheel)

    # Plot the trajectory
    if len(trajectory) > 1:
        trajectory_array = np.array(trajectory)
        ax.plot(trajectory_array[:, 0], trajectory_array[:, 1], color='blue')

    # Set the x and y limits of the plot
    figs=5
    ax.set_xlim([-figs, +figs])
    ax.set_ylim([-figs, +figs])

    # Set the aspect ratio of the plot to equal
    ax.set_aspect('equal')

def animate(x, y, heading ,steer):
    data = np.array([x, y, heading,steer]).T
    fig, ax = plt.subplots()
    trajectory = []
    ani = FuncAnimation(fig, update, frames=data, fargs=(ax, trajectory), interval=1 )
    plt.show()

def find_converged_value(data, start_idx, threshold, window_width):
    """
    Finds the converged value of a 1D array of data.
    If the data did not converge, returns NaN.
    """
    # Initialize the starting and ending indices of the window
    window_start = start_idx # ignore start part of sim
    window_end = window_start + window_width

    # Iterate through the data and check for convergence over the window
    while window_end < len(data):
        # Get the window of data
        window_data = data[window_start:window_end]

        # Check if the absolute difference between the maximum and minimum values in the window is below the threshold
        if abs(np.max(window_data) - np.min(window_data)) < threshold:
            # If the values are converging, return the converged value
            # print("total lengt: ",len(data))
            # print("where converge: ",window_start)
            return np.mean(window_data)

        # Move the window by one step
        window_start += 1
        window_end += 1

    # If the loop completes without finding convergence, return NaN
    return np.nan

def simulate_car(u, mu, lf, lr, h, m, I, Bf ,Cf ,Df ,Ef, Br, Cr, Dr ,Er, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max, des_vel,des_steer):
    # simulate single-track model
    x_st = odeint(func_ST, x0_ST, t, args=(u, mu, lf, lr, h, m, I, Bf ,Cf ,Df ,Ef, Br, Cr, Dr ,Er, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max, des_vel,des_steer))
    return x_st

def interpolate_array(array):
    """
    Interpolates 0 or NaN values in a 2D array using 4 nearest neighbors
    
    Args:
    array (numpy array): 2D array to interpolate
    
    Returns:
    numpy array: interpolated 2D array
    """
    # create a copy of the array to modify
    interpolated_array = array.copy()
    
    # loop over each element in the array
    for i in range(array.shape[0]):
        for j in range(1,array.shape[1]):
            # if the value is 0 or NaN, interpolate using 4 nearest neighbors
            # if interpolated_array[i,j] == 0 or np.isnan(interpolated_array[i,j]):
            if interpolated_array[i,j] == 0 or interpolated_array[i,j] == np.nan:
                neighbors = []
                # check the top neighbor
                if i > 0 :
                    if array[i-1,j]!=0 and array[i-1,j]!=np.nan:
                        neighbors.append(array[i-1,j])
                # check the bottom neighbor
                if i < array.shape[0]-1 :
                    if array[i+1,j]!=0 and array[i+1,j]!=np.nan:
                        neighbors.append(array[i+1,j])
                # check the left neighbor
                if j > 0 :
                    if array[i,j-1]!=0 and array[i,j-1]!=np.nan:
                        neighbors.append(array[i,j-1])
                # check the right neighbor
                if j < array.shape[1]-1 :
                    if  array[i,j+1]!=0 and array[i,j+1]!=np.nan:
                        neighbors.append(array[i,j+1])
                # if there are no valid neighbors, skip interpolation for this value
                if len(neighbors) == 0:
                    continue
                # otherwise, take the average of the neighbors
                else:
                    interpolated_array[i,j] = np.mean(neighbors)
    # if np.any(np.isnan(interpolated_array)):
    #    return interpolate_array(interpolated_array)
    return interpolated_array

def find_steer(vel,Ac,f):
    steer_range = np.arange(0.0,0.5,0.001)
    best_diff=9999
    for cur_steer in steer_range:
        if cur_steer ==0:
            cur_Ac=0
        else:
            cur_Ac = f(np.array([vel, cur_steer]))
        if cur_Ac == np.nan:
            pass
        Ac_diff = abs(Ac-cur_Ac)
        if Ac_diff < best_diff:
            best_diff = Ac_diff
            best_steer = cur_steer
    if best_steer == 9999 or best_diff > 0.1:
        return np.nan
    else:
        return best_steer

if __name__ == '__main__':
    mu = 1.0
    lf = 0.25
    lr = 0.20
    h = 0.15
    m = 3.0
    I = 0.075
    g = 9.81
    #steering constraints
    s_min = -1.066  #minimum steering angle [rad]
    s_max = 1.066  #maximum steering angle [rad]
    sv_min = -0.4  #minimum steering velocity [rad/s]
    sv_max = 0.4  #maximum steering velocity [rad/s]
    #longitudinal constraints
    v_min = -13.6  #minimum velocity [m/s]
    v_max = 50.8  #minimum velocity [m/s]
    v_switch = 7.319  #switching velocity [m/s]
    a_max = 11.5  #maximum absolute acceleration [m/s^2]
    t_start = 0.0
    t_final = 10.0
    x0 = -0.0
    y0 = -0.0
    Psi0 = 0.0
    vx0 = 0.0
    vy0= 0.0
    Psidot0 = 0.0
    delta0 = 0.0
    initial_state = [x0,y0,Psi0,vx0,vy0,Psidot0,delta0]
    x0_ST = np.array(initial_state)
    # time vector
    t = np.arange(t_start, t_final, 1e-2)
    # set input: constant stereing and velocity 
    u = np.array([0.0, 0.0]) # dump value. it does not effect
    # Pacejka tire model parameter
    [Bf, Cf, Df, Ef] = [4, 0.8, 0.8, 0.8]
    [Br, Cr, Dr, Er] = [3, 1.0, 1.0, 1.0]

    # make look up table
    test_vel_range = np.arange(0.1,10.0, 0.2)
    test_steer_range = np.arange(0.0,0.50,0.01)
    # test_vel_range = np.arange(0.1,10.0, 1.0)
    # test_steer_range = np.arange(0.0,0.50,0.05)
    print("test_vel_range: ",test_vel_range)
    print("test_steer_range: ",test_steer_range)
    vel_len = len(test_vel_range)
    steer_len = len(test_steer_range)
    print("vel_len:", vel_len)
    print("steer_len: ",steer_len)
    assert(vel_len == steer_len)
    # Create the 2D array to store the b values
    Ac_list = np.zeros((len(test_vel_range), len(test_steer_range)))

    # create 2D meshgrid of input arrays
    for i, des_vel in enumerate(test_vel_range):
        for j, des_steer in enumerate(test_steer_range):
            x_st = simulate_car(u, mu, lf, lr, h, m, I, Bf ,Cf ,Df ,Ef, Br, Cr, Dr ,Er, s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max, des_vel,des_steer)          
            
            # export Ac, steer, Vx
            steer = x_st[:,6]
            Vx =x_st[:,3]
            Vy =x_st[:,4]
            V = Vx*Vx+Vy*Vy
            V = np.sqrt(V)
            psidot = x_st[:,5]
            Ac=V*psidot

            # determine steady state
            ss_threshold = 0.0000001
            ss_window_width = 10
            ss_start_idx= 500
            ss_Ac = find_converged_value(Ac,ss_start_idx, ss_threshold, ss_window_width)
            Ac_list[i,j]=ss_Ac

    # plot 
    Ac_list = np.array(Ac_list)
    # Ac_list=interpolate_array(Ac_list) # zero value interpolation
    X, Y = np.meshgrid(test_vel_range, test_steer_range)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(X.flatten(), Y.flatten(), Ac_list.flatten(), cmap='viridis')
    ax.set_xlabel('vel')
    ax.set_ylabel('steer')
    ax.set_zlabel('Ac')
    # print("x:",X.flatten())
    # plt.show()

    # regression steer model from the simulation
    f= RegularGridInterpolator((test_vel_range, test_steer_range),Ac_list,method='linear',bounds_error=False,fill_value=np.nan )
    # f= RegularGridInterpolator((test_vel_range, test_steer_range),Ac_list,method='linear' )

    ### ac = f(np.array([cur_vel,steer])) ##
    # print(f(np.array([0.1,0.88])))

    # Transform axis of LUT
    max_Ac = np.max(Ac_list)
    desired_test_Ac_range_size = vel_len
    test_Ac_range = np.arange(0.0,max_Ac,max_Ac/desired_test_Ac_range_size)
    assert(len(test_Ac_range)==desired_test_Ac_range_size)
    steer_list_1D = []
    steer_list_2D =np.zeros((vel_len, len(test_Ac_range)))
    for i,vel in enumerate(test_vel_range):
        for j,Ac in enumerate(test_Ac_range):
            # print("vel, Ac: ",vel,Ac)
            if Ac ==0:
                target_steer = 0.0
            else:
                target_steer = find_steer(vel,Ac,f)
                
            steer_list_1D.append(target_steer)
            steer_list_2D[i,j]=target_steer
            print("vx, ac , steer: ",vel,Ac,target_steer)
 
    # save LUT to txt file.
    with open('loouptb.txt', 'w') as f:
        f.write('vx: ')
        for i,value in enumerate(test_vel_range):
            if i == len(test_vel_range)-1:
                print('{:g}'.format(float(value)), end='\n', file=f)
            else:
                print('{:g}'.format(float(value)), end=', ', file=f)
        f.write('alat: ')
        for i,value in enumerate(test_Ac_range):
            if i == len(test_Ac_range)-1:
                print('{:g}'.format(float(value)), end='\n', file=f)
            else:
                print('{:g}'.format(float(value)), end=', ', file=f)
        f.write('delta: ')
        for i,value in enumerate(steer_list_1D):
            if i == len(steer_list_1D)-1:
                print('{:g}'.format(float(value)), end='\n', file=f)
            else:
                print('{:g}'.format(float(value)), end=', ', file=f)

    X, Y = np.meshgrid(test_vel_range[::-1], test_Ac_range)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    steer_list_2D = np.array(steer_list_2D)
    # steer_list_2D=interpolate_array(steer_list_2D) # zero value interpolation
    ax.plot_trisurf(X.flatten(), Y.flatten(), np.array(steer_list_2D).flatten(), cmap='viridis')
    ax.set_xlabel('vel')
    ax.set_ylabel('Ac')
    ax.set_zlabel('Steer')
    plt.show()
    # # Save the regressed function as an pkl
    # with open('ac_function.pkl', 'wb') as f_out:
    #     pickle.dump(f, f_out)

    
    #####Load the lookup table from the binary file
    #####Load the lookup table from the binary file
    """
    import pickle

    # load the interpolation function from a file
    with open('ac_function.pkl', 'rb') as f_in:
        f = pickle.load(f_in)

    steer = f(np.array([cur_vel,des_ac]))
    """