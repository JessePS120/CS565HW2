import rclpy
from rclpy.node import Node
import time
import math
import numpy as np 
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt 
plt.ion()

map_dimensions = 16

class GoalFinder(): 
    def __init__(self, map_dimensions): 
        self.map_origin_x = map_dimensions / 2
        self.map_origin_y = map_dimensions / 2 
        self.map_dimensions = map_dimensions 
        self.occ_grid = np.ones((map_dimensions, map_dimensions), dtype = int)
        self.goals_reached = []
        self.cur_goal = (None, None) 
    #Public Methods 
    #Calculates the nearest goal using lidar by sweeping from -3pi/4 to 3pi/4. 
    #Returns a pair of coordinates (x, y) that correspond to the furthest coordinates away. 
    def find_goal(self, robot_info, ranges): 
        # Lidar parameters
        num_readings = 270
        angle_min = -3 * math.pi / 4
        angle_max = 3 * math.pi / 4
        angle_increment = (angle_max - angle_min) / (num_readings - 1)
        robot_x, robot_y, robot_theta = robot_info 
        furthest = (None, None, None, None, -1)
        self._clear_occ_grid() 
        for i in range(num_readings):
            r = ranges[i]
            lidar_theta = angle_min + i * angle_increment + robot_theta
            Lx = robot_x + r * math.cos(lidar_theta)
            Ly = robot_y + r * math.sin(lidar_theta)
            mx, my = self.to_grid_coords((Lx, Ly))
            if 0 <= mx < self.map_dimensions and 0 <= my < self.map_dimensions:
                if r > furthest[4] and (furthest[0], furthest[1] not in self.goals_reached):
                    furthest = (mx, my, Lx, Ly, r)
        self.cur_goal = (furthest[0], furthest[1])
        self.occ_grid[furthest[1], furthest[0]] = 127
        return (furthest[2], furthest[3]) 
    #This function should be called when the PID controller is able to navigate to 
    #the goal selected by find_goal. This ensures that the goal will be stored 
    #and not picked again. 
    def set_goal(self): 
        if None not in self.cur_goal: 
            self.occ_grid[self.cur_goal[1], self.cur_goal[0]] = 255
            self.goals_reached.append(self.cur_goal) 
    #Function for converting coordinates into grid square coordinates. 
    def to_grid_coords(self, coords):
        rx, ry = coords
        mx = int(self.map_origin_x + int(rx) - (rx < 0)) 
        my = int(self.map_origin_y + int(ry) - (ry < 0)) 
        return((mx, my))
    #Getters 
    def get_occ_grid(self):  
        return self.occ_grid 
    #Private Methods
    def _clear_occ_grid(self):
        self.occ_grid[self.occ_grid < 255] = 1 

class Tracker(Node):
    def __init__(self):
        super().__init__('Track')
        self.startTime = time.time()
        self.time = 0
        self.GoalFinder = GoalFinder(map_dimensions)
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
        # Create a persistent figure and image for faster updates
        self.fig, self.ax = plt.subplots()
        self.img = None
    def listener_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = math.atan2(2*(msg.pose.pose.orientation.w*msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y), 
                                      1-2*(msg.pose.pose.orientation.y**2 + msg.pose.pose.orientation.z**2))

    def sensor_callback(self, msg): 
        '''global occ_grid
        occ_grid = -1 * np.ones((map_dimensions, map_dimensions), dtype = int)
        x, y = rcoord_to_mapcoord((self.robot_x, self.robot_y)) 
        occ_grid[y, x] = 255
        Lx = self.robot_x + 5 * math.cos(3 * math.pi / 4 + self.robot_theta) 
        Ly = self.robot_y + 5 * math.sin(3 * math.pi / 4 + self.robot_theta) 
        Lx, Ly = rcoord_to_mapcoord((Lx, Ly)) 
        mx, my, r = find_furthest_lidar_square(self.robot_x, self.robot_y, self.robot_theta, msg.ranges)
        occ_grid[my, mx] =255'''

        # Update goal finder and get occupancy grid
        self.GoalFinder.find_goal((self.robot_x, self.robot_y, self.robot_theta), msg.ranges)
        grid = self.GoalFinder.get_occ_grid()

        # Create the image on first callback, then update the data
        if self.img is None:
            self.img = self.ax.imshow(grid, cmap='gray', origin='lower', vmin=0, vmax=255, interpolation='nearest')
            self.ax.set_title('Occupancy Grid')
            plt.show()
        else:
            self.img.set_data(grid)
            # in case value range changed
            self.img.set_clim(0, 255)

        # Draw and flush events so the window updates
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.1)
    
def main(args=None):
    rclpy.init(args=args)
    tracker_node = Tracker()
    rclpy.spin(tracker_node)
    tracker_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
