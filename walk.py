import rclpy
from rclpy.node import Node
import time
import math
import numpy as np 
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

map_dimensions = 16
map_origin_x = map_dimensions / 2
map_origin_y = map_dimensions / 2

occ_grid = -1 * np.ones((map_dimensions, map_dimensions), dtype = int)

robot_pos_set = False 


plt.ion
fig, ax = plt.subplots() 
img = ax.imshow(occ_grid, cmap='gray', origin = 'lower', vmin=0, vmax=255)
plt.ion()
plt.show()

class Tracker(Node):
    def __init__(self):
        super().__init__('Track')
        self.startTime = time.time()
        self.time = 0
        self.subscription = self.create_subscription(
            Odometry,
            '/ground_truth',
            self.listener_callback,
            10
        )
        self.subscription = self.create_subscription( 
            LaserScan, 
            '/base_scan', 
            self.sensor_callback, 
            10
        )
        self.robot_x = 0
        self.robot_y = 0
        self.robot_theta = 0 
    def listener_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = math.atan2(2*(msg.pose.pose.orientation.w*msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y), 
                                      1-2*(msg.pose.pose.orientation.y**2 + msg.pose.pose.orientation.z**2))
        robot_pos_set = True 

    def sensor_callback(self, msg): 
        global occ_grid
        occ_grid = -1 * np.ones((map_dimensions, map_dimensions), dtype = int)
        x, y = rcoord_to_mapcoord((self.robot_x, self.robot_y)) 
        occ_grid[y, x] = 255
        Lx = self.robot_x + 5 * math.cos(3 * math.pi / 4 + self.robot_theta) 
        Ly = self.robot_y + 5 * math.sin(3 * math.pi / 4 + self.robot_theta) 
        Lx, Ly = rcoord_to_mapcoord((Lx, Ly)) 
        mx, my, r = find_furthest_lidar_square(self.robot_x, self.robot_y, self.robot_theta, msg.ranges)
        occ_grid[my, mx] =255 
        

        img.set_data(occ_grid) 
        fig.canvas.draw()
        fig.canvas.flush_events() 
        plt.pause(0.1)

def find_furthest_lidar_square(robot_x, robot_y, robot_theta, ranges, map_dimensions=16, max_range=5):
    # Lidar parameters
    num_readings = 270
    angle_min = -3 * math.pi / 4
    angle_max = 3 * math.pi / 4
    angle_increment = (angle_max - angle_min) / (num_readings - 1)

    furthest = (None, None, -1)
    for i in range(num_readings):
        r = ranges[i]
        lidar_theta = angle_min + i * angle_increment + robot_theta
        Lx = robot_x + r * math.cos(lidar_theta)
        Ly = robot_y + r * math.sin(lidar_theta)
        mx, my = rcoord_to_mapcoord((Lx, Ly))
        if 0 <= mx < map_dimensions and 0 <= my < map_dimensions:
            if r > furthest[2]:
                furthest = (mx, my, r)
    return furthest  
    
def rcoord_to_mapcoord(rcoord):
    rx, ry = rcoord
    mx = int(map_origin_x + int(rx) - (rx < 0)) 
    my = int(map_origin_y + int(ry) - (ry < 0)) 
    return((mx, my))
    
def main(args=None):
    rclpy.init(args=args)
    tracker_node = Tracker()
    rclpy.spin(tracker_node)
    tracker_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
