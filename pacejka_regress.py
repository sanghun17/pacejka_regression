
# import sys
# sys.path.insert(0, '/home/shx1/catkin_ws_pacejka/src/rosbag2')
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
# from sensor_msgs.msg import Imu
# from nav_msgs.msg import Odometry
# from ackermann_msgs.msg import AckermannDrive
# from ackermann_msgs.msg import AckermannDriveStamped
import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

class BagFileParser():
    def __init__(self, bag_file):
        self.conn = sqlite3.connect(bag_file)
        self.cursor = self.conn.cursor()

        ## create a message type map
        topics_data = self.cursor.execute("SELECT id, name, type FROM topics").fetchall()
        self.topic_type = {name_of:type_of for id_of,name_of,type_of in topics_data}
        self.topic_id = {name_of:id_of for id_of,name_of,type_of in topics_data}
        self.topic_msg_message = {name_of:get_message(type_of) for id_of,name_of,type_of in topics_data}

    def __del__(self):
        self.conn.close()

    # Return [(timestamp0, message0), (timestamp1, message1), ...]
    def get_messages(self, topic_name):

        topic_id = self.topic_id[topic_name]
        # Get from the db
        rows = self.cursor.execute("SELECT timestamp, data FROM messages WHERE topic_id = {}".format(topic_id)).fetchall()
        # Deserialise all and timestamp them
        return [ (timestamp,deserialize_message(data, self.topic_msg_message[topic_name])) for timestamp,data in rows]



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

    bag_file = '/home/shx1/rosbag/sh_test2/sh_test2_0.db3'
    parser = BagFileParser(bag_file)

    topic_1 = '/odom' # topic_1 have to be odom including vel_x, vel_y
    topic_2 = '/imu/data' # topic_2 have to be imu
    topic_3 = '/ackermann_cmd'  # topic_3 have to contain steering angle.

    odom_msgs = parser.get_messages(topic_1)
    imu_msgs = parser.get_messages(topic_2)
    steer_msgs = parser.get_messages(topic_3)

    matching_msgs_topic1 = []
    matching_msgs_topic2 = []
    matching_msgs_topic3 = []

    for odom_msg in odom_msgs:
        odom_timestamp_sec  = odom_msg[1].header.stamp.sec
        odom_timestamp_nsec = odom_msg[1].header.stamp.nanosec*(0.000000001)
        odom_timestamp      = odom_timestamp_sec+odom_timestamp_nsec

        closest_diff = float('inf')
        closest_imu_msg = None
        for imu_msg in imu_msgs:
            imu_timestamp_sec  =  imu_msg[1].header.stamp.sec
            imu_timestamp_nsec =  imu_msg[1].header.stamp.nanosec*(0.000000001)
            imu_timestamp      =  imu_timestamp_sec+imu_timestamp_nsec
            timestamp_diff     = abs(odom_timestamp - odom_timestamp)

            if timestamp_diff < closest_diff:
                closest_diff = timestamp_diff
                closest_imu_msg = imu_msg

            if timestamp_diff > closest_diff:
                break

        closest_diff = float('inf')
        closest_steer_msg = None

        for steer_msg in steer_msgs:
            steer_timestamp_sec  =  steer_msg[1].header.stamp.sec
            steer_timestamp_nsec =  steer_msg[1].header.stamp.nanosec*(0.000000001)
            steer_timestamp      =  steer_timestamp_sec+imu_timestamp_nsec
            timestamp_diff       = abs(odom_timestamp - steer_timestamp)

            if timestamp_diff < closest_diff:
                closest_diff2 = timestamp_diff
                closest_steer_msg = steer_msg

            if timestamp_diff > closest_diff:
                break

        if closest_imu_msg is not None and closest_steer_msg is not None:
                matching_msgs_topic1.append(odom_msg)
                matching_msgs_topic2.append(closest_imu_msg)
                matching_msgs_topic3.append(closest_steer_msg)

    # Print the number of matching messages found
    print("Found {} matching messages".format(len(matching_msgs_topic1)))
 
    accel_x = []
    accel_y = []
    vel_x = []
    vel_y = []
    yaw_rate = []
    steer_angle = []
    for ros_msg in matching_msgs_topic1:
        vel_x.append([ros_msg[1].twist.twist.linear.x])
        vel_y.append([ros_msg[1].twist.twist.linear.y])
    for ros_msg in matching_msgs_topic2:
        accel_x.append([ros_msg[1].linear_acceleration.x])
        accel_y.append([ros_msg[1].linear_acceleration.y])
        yaw_rate.append([ros_msg[1].angular_velocity.z])
    for ros_msg in matching_msgs_topic3:
        steer_angle.append([ros_msg[1].drive.steering_angle])

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