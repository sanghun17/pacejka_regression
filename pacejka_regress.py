import numpy as np
import rospy
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
import rosbag   
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive

# Function to calculate lateral force using Pacejka model
def pacejka_model(alpha, mu, Fz, B, C, D, E):
    Fy_hat = mu * Fz * D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha))))
    return Fy_hat

# Function to make virtual data for regression test
def virtual_data_making(alpha, mu, Fz,B,C,D,E):
    Fy_virtual = mu * Fz * D * np.sin(C * np.arctan(B * alpha - E * (B * alpha - np.arctan(B * alpha)))) + np.random.normal(0, 1, len(alpha))
    return Fy_virtual

# Function to calculate tire slip angle using ROS data
def calculate_slip_angle_rear(v_y, yaw_rate, l_r, v_x):
    return np.arctan((v_y - yaw_rate * l_r) / v_x)
def calculate_slip_angle_front(v_y, yaw_rate, l_f, v_x,steer_angle):
    return np.arctan((v_y - yaw_rate * l_f) / v_x)-steer_angle

# Function to calculate normal force on tire using ROS data
def calculate_normal_force_rear(mass, g, a_long, h_cg, lr, lf):
    return (mass * g * lf + mass * a_long * h_cg) / (lr + lf)
def calculate_normal_force_front(mass, g, a_long, h_cg, lr, lf):
    return (mass * g * lr - mass * a_long * h_cg) / (lr + lf)

# Function to calculate true lateral force using ROS data
def calculate_true_lateral_force_rear(mass, lr, lf, accel_y):
    return (mass * lf / ((lr + lf))) * accel_y
def calculate_true_lateral_force_front(mass, lr, lf, steer_angle, accel_y):
    return (mass * lr / ((lr + lf) * np.cos(steer_angle))) * accel_y

# Function to run EM algorithm and estimate parameters
def run_em_algorithm(alpha, Fy, mu, Fz, lr, lf, h_cg, mass):
    # Initialize parameters
    B = 800.0
    C = 20.0
    D = 20.0
    E = 200.0

    EM_iter=3 # Run 3 iterations of EM algorithm
    outliers_list = np.zeros(len(alpha), dtype=bool)
    
    for i in range(1, EM_iter+1):
        print("iteration of EM: ",i)
        # Estimate new parameters using least squares
        x0 = np.array([B, C, D, E])
        alpha_inlier = alpha[~outliers_list]
        print("num outlier: ",outliers_list.shape)

        Fy_inlier = Fy[~outliers_list]
        Fz_inlier = Fz[~outliers_list]
        bounds = [(0.1, 12.0), (0.1, 1.5), (0.1, 10.0), (-10.0, 1.1)]  # # Set lower and upper bounds for parameters
        result = minimize(least_squares, x0, args=(alpha_inlier, Fy_inlier, mu, Fz_inlier, lr, lf, h_cg, mass),bounds=bounds)
        B, C, D, E = result.x

        Fy_hat = pacejka_model(alpha, mu, Fz, B, C, D, E)
        residuals = Fy - Fy_hat
        threshold = 10 / pow(2, i) # 10 should be tuned properly? 
        # threshold = 100000
        outliers_list = np.abs(residuals) > threshold

    # Plot results
    plt.figure()
    plt.scatter(alpha[outliers_list], Fy[outliers_list], color='red')
    plt.scatter(alpha[~outliers_list], Fy[~outliers_list], color='green')
    alpha_range = np.linspace(-0.6, 0.6, 100)
    Fy_range = pacejka_model(alpha_range, mu, np.mean(Fz), B, C, D, E)
    plt.plot(alpha_range, Fy_range, color='blue')
    plt.xlabel('Tire Slip Angle')
    plt.ylabel('Lateral Force')
    plt.title('Pacejka Model Regression')

    # Return estimated parameters
    return B, C, D, E

# Function to calculate least squares cost
def least_squares(x, alpha, Fy, mu, Fz, lr, lf, h_cg, mass):
    B, C, D, E = x
    Fy_hat = pacejka_model(alpha, mu, Fz, B, C, D, E)
    residual = Fy - Fy_hat
    return np.sum(residual ** 2)



if __name__ == '__main__':
    # Load ROS data
    bag = rosbag.Bag('/home/shx1/shtest1_2023-04-02-12-55-32.bag')
    topic_1 = '/imu' 
    topic_2 = '/vesc/odom'
    topic_3 = '/vesc/low_level/ackermann_cmd_mux/output' # topic_3 have to contain steering angle.

    # Determine the longer and shorter topics
    length_1 = bag.get_message_count(topic_filters=[topic_1])
    length_2 = bag.get_message_count(topic_filters=[topic_2])

    if length_1 >= length_2:
        longer_topic = topic_1
        shorter_topic = topic_2
    else:
        longer_topic = topic_2
        shorter_topic = topic_1

  # Create lists to store matching messages for different topics
  # since, frequency of each topic is different.
    matching_msgs_topic1 = []
    matching_msgs_topic2 = []
    matching_msgs_topic3 = []

    for short_msg in bag.read_messages(topics=[shorter_topic]):
        short_timestamp = short_msg.timestamp
        closest_diff = float('inf')
        closest_msg = None

        for long_msg in bag.read_messages(topics=[longer_topic],start_time=short_timestamp-rospy.Duration(1)):
            long_timestamp = long_msg.timestamp
            timestamp_diff = abs(short_timestamp.to_sec() - long_timestamp.to_sec())

            if timestamp_diff < closest_diff:
                closest_diff = timestamp_diff
                closest_msg = long_msg

            if timestamp_diff > closest_diff:
                break

        closest_diff2 = float('inf')
        closest_msg2 = None

        for steer_msg in bag.read_messages(topics=[topic_3],start_time=short_timestamp-rospy.Duration(1)):
            steer_timestamp = steer_msg.timestamp
            timestamp_diff = abs(short_timestamp.to_sec() - steer_timestamp.to_sec())

            if timestamp_diff < closest_diff2:
                closest_diff2 = timestamp_diff
                closest_msg2 = steer_msg

            if timestamp_diff > closest_diff2:
                break

        if closest_msg is not None and closest_msg2 is not None:
            if closest_msg.topic == topic_1:
                matching_msgs_topic1.append(closest_msg)
                matching_msgs_topic2.append(short_msg)
                matching_msgs_topic3.append(closest_msg2)
            elif closest_msg.topic == topic_2:
                matching_msgs_topic1.append(short_msg)
                matching_msgs_topic2.append(closest_msg)
                matching_msgs_topic3.append(closest_msg2)

    # Print the number of matching messages found
    print("Found {} matching messages".format(len(matching_msgs_topic1)))
    # Close the bag file
    bag.close()

    accel_x = []
    accel_y = []
    vel_x = []
    vel_y = []
    yaw_rate = []
    steer_angle = []
    for msg in matching_msgs_topic1:
        # Extract the actual ROS message from the BagMessage object
        ros_msg = msg.message
        # Convert ROS message to numpy array and append to list
        accel_x.append([ros_msg.linear_acceleration.x])
        accel_y.append([ros_msg.linear_acceleration.y])
        yaw_rate.append([ros_msg.angular_velocity.z])
    for msg in matching_msgs_topic2:
        # Extract the actual ROS message from the BagMessage object
        ros_msg = msg.message
        # Convert ROS message to numpy array and append to list
        vel_x.append([ros_msg.twist.twist.linear.x])
        vel_y.append([ros_msg.twist.twist.linear.y])
    for msg in matching_msgs_topic3:
        ros_msg=msg.message
        steer_angle.append([ros_msg.drive.steering_angle])

    # Convert list to numpy array and reshape to 1D array
    accel_x = np.array(accel_x)
    accel_y = np.array(accel_y)
    vel_x = np.array(vel_x)
    vel_y = np.array(vel_y)
    yaw_rate = np.array(yaw_rate)
    steer_angle = np.array(steer_angle)
    accel_x=accel_x.flatten()
    accel_y=accel_y.flatten()
    vel_x=vel_x.flatten()
    vel_y=vel_y.flatten()
    yaw_rate=yaw_rate.flatten()
    steer_angle = steer_angle.flatten()

    # If vel_x is 0 or negative, remove corresponding data
    for i in range(len(vel_x)-1, -1, -1):
        if vel_x[i] <= 0:
            accel_x = np.delete(accel_x, i)
            accel_y = np.delete(accel_y, i)
            vel_x = np.delete(vel_x, i)
            vel_y = np.delete(vel_y, i)
            yaw_rate = np.delete(yaw_rate, i)
            steer_angle = np.delete(steer_angle,i)





    # Define vehicle and tire parameters
    lr = 0.125
    lf = 0.125
    h_cg = 0.2
    mass = 3.0
    g = 9.81
    
    # Calculate tire slip angle using ROS data
    slip_angle_rear = calculate_slip_angle_rear(vel_y, yaw_rate, lr, vel_x)
    slip_angle_front = calculate_slip_angle_front(vel_y, yaw_rate, lf, vel_x,steer_angle)

    # Calculate normal force and true lateral force using ROS data
    Fz_rear = calculate_normal_force_rear(mass, g, accel_x, h_cg, lr, lf)
    Fz_front = calculate_normal_force_front(mass, g, accel_x, h_cg, lr, lf)

    # print(Fz.shape)
    Fy_true_rear = calculate_true_lateral_force_rear(mass, lr, lf, accel_y=accel_y)
    Fy_true_front = calculate_true_lateral_force_front(mass, lr, lf, steer_angle, accel_y)

    ### regression test with virtual data start ####
    ### regression test with virtual data start ####
    ### regression test with virtual data start ####
    # alpha_v = np.arange(-0.6, 0.6, 0.01)
    # mu_v = 1.0
    # Fz_v = np.full_like(alpha_v, 3.0)
    # B_vt = 4.0 # vt: virtual true
    # C_vt = 1.0
    # D_vt = 1.0
    # E_vt = 1.0
    # Fy_v = virtual_data_making(alpha_v, mu_v, Fz_v,B_vt,C_vt,D_vt,E_vt)
    # B_ve, C_ve, D_ve, E_ve = run_em_algorithm(alpha=alpha_v, Fy=Fy_v, mu=mu_v, Fz=Fz_v, lr=lr, lf=lf, h_cg=h_cg, mass=mass)
    # print(f"B_vt: {B_vt}, C_vt: {C_vt}, D_vt: {D_vt}, E_vt: {E_vt}")
    # print(f"B_ve: {B_ve}, C_ve: {C_ve}, D_ve: {D_ve}, E_ve: {E_ve}") # ve: virtual estimation
    ### regression test with virtual data end ###
    ### regression test with virtual data end ###
    ### regression test with virtual data end ###

    # Run EM algorithm to estimate tire parameters
    B_r, C_r, D_r, E_r = run_em_algorithm(alpha=slip_angle_rear, Fy=Fy_true_rear, mu=1.0, Fz=Fz_rear, lr=lr, lf=lf, h_cg=h_cg, mass=mass)
    B_f, C_f, D_f, E_f = run_em_algorithm(alpha=slip_angle_front, Fy=Fy_true_front, mu=1.0, Fz=Fz_front, lr=lr, lf=lf, h_cg=h_cg, mass=mass)

    # Print estimated tire parameters
    print(f"B_r: {B_r}, C_r: {C_r}, D_r: {D_r}, E_r: {E_r}") # r: rear
    print(f"B_f: {B_f}, C_f: {C_f}, D_f: {D_f}, E_f: {E_f}") # f: front

    # Show the figures
    plt.show()    