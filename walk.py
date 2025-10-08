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

map_dimension = 64
map_resolution = 0.25

#Helper Classes. 
class Queue():
    def __init__(self):
        self.items = []
        
    def is_empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        # Return and remove the head item, or None if empty
        if not self.is_empty():
            return self.items.pop(0)
        return None

    def peek(self):
        if not self.is_empty():
            return self.items[0]
        return None 


class Map(): 
    #These values are chosen so that a gray scale map may be created for debugging. 
    #All tiles are initialized to zero. 
    UNKNOWN = 0
    #All tiles that contain walls.  
    WALL = 63 
    #Marks tiles that are free. 
    VIEWED = 127
    #Marks the current goal tile. 
    GOAL = 191
    #Marks tiles that have been visited. 
    VISITED = 255
    #Initializer
    def __init__(self, map_dimension, resolution): 
        self.map_origin_x = map_dimension / 2
        self.map_origin_y = map_dimension/ 2 
        self.map_dimension = map_dimension 
        #What real world length each tile actually represents. 
        self.resolution = resolution 
        self.occ_grid = np.zeros((map_dimension, map_dimension), dtype = int)
    #Setters. Note that this erases the current occupany grid. 
    def set_dimensions(self, map_dimension): 
        self.map_origin_x = map_dimension / 2
        self.map_origin_y = map_dimension / 2 
        self.map_dimension = map_dimension 
        self.occ_grid = np.zeros((map_dimension, map_dimension), dtype = int)
    #Getters 
    def get_dimension(self): 
        return self.map_dimension
    def get_resolution(self): 
        return self.resolution
    def get_occ_grid(self): 
        return self.occ_grid 
    #Public Methods. 
    #Checks if a pair of coordinates will fit within the grid. 
    def are_valid_world(self, coords): 
        grid_coords = self.to_grid_coords(coords)
        if 0 <= grid_coords[0] < self.map_dimension and 0 <= grid_coords[1] < self.map_dimension: 
            return True 
        return False  
    def are_valid_grid(self, grid_coords): 
        if 0 <= grid_coords[0] < self.map_dimension and 0 <= grid_coords[1] < self.map_dimension: 
            return True 
        return False  
    #Converts coordinates into grid coordinates. 
    def to_grid_coords(self, coords):
        # coords are world coordinates (rx, ry) in meters
        rx, ry = coords
        # convert meters -> cells, then offset by origin and floor to get integer cell index
        mx = int(math.floor(self.map_origin_x + (rx / self.resolution)))
        my = int(math.floor(self.map_origin_y + (ry / self.resolution)))
        # clamp to valid range
        mx = max(0, min(self.map_dimension - 1, mx))
        my = max(0, min(self.map_dimension - 1, my))
        return (mx, my)
    def to_world_coords(self, grid_coords):
        gx, gy = grid_coords
        wx = (gx - self.map_origin_x) * self.resolution + self.resolution / 2.0
        wy = (gy - self.map_origin_y) * self.resolution + self.resolution / 2.0
        return (wx, wy)
    def set_occ_grid_wall(self, grid_coords): 
        xcoord, ycoord = grid_coords
        self.occ_grid[ycoord, xcoord] = self.WALL 
    #Sets a square in the grid to the value passed in.  
    def set_occ_grid_viewed(self, grid_coords): 
        xcoord, ycoord = grid_coords
        self.occ_grid[ycoord, xcoord] = self.VIEWED
    def set_occ_grid_goal(self, grid_coords): 
        xcoord, ycoord = grid_coords
        self.occ_grid[ycoord, xcoord] = self.GOAL
    def set_occ_grid_visited(self, grid_coords): 
        xcoord, ycoord = grid_coords
        self.occ_grid[ycoord, xcoord] = self.VISITED
    #Clears the coordinates of the occupancy grid. 
    def clear_occ_grid(self, grid_coords): 
        x, y = grid_coords 
        self.occ_grid[y, x] = 0 
    def update_map(self, robot_info, ranges): 
        #Lidar parameters. 
        max_range = 5 
        num_readings = len(ranges) 
        angle_min = -3 * math.pi / 4
        angle_max = 3 * math.pi / 4
        angle_increment = (angle_max - angle_min) / (num_readings - 1)
        #Robot data. 
        robot_x, robot_y, robot_theta = robot_info 
        #Loop through all of the lidar readings and select the furthest point away. 
        for i in range(num_readings):
            lidar_theta = angle_min + i * angle_increment + robot_theta
            temp_coords = (robot_x + ranges[i] * math.cos(lidar_theta), robot_y + ranges[i] * math.sin(lidar_theta))
            temp_grid_coords = self.to_grid_coords(temp_coords) 
            #Lidar points that go outside of the map are ignored. 
            if self.are_valid_world(temp_coords): 
                #Check if lidar has gone its max range. If not then it has hit a wall. 
                if (ranges[i] != max_range ): 
                    self.set_occ_grid_wall(temp_grid_coords)
                #Find all of the squares that lidar has passed through and set to to "viewed" if they contained no previous state. 
                points = self._bresenham(self.to_grid_coords((robot_x, robot_y)), temp_grid_coords) 
                if points: 
                    for j in points: 
                        if self.get_occ_grid()[j[1], j[0]] == self.UNKNOWN and self.are_valid_grid((j[0], j[1])): 
                            self.set_occ_grid_viewed((j[0], j[1])) 
    #Uses breenham algorithm to calculate and return the number of squares that are 
    #touched by the line between two endpoints. 
    def _bresenham(self, scoords, gcoords):
        x0, y0 = scoords 
        x1, y1 = gcoords 
        points = []
        x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy 
        x, y = x0, y0
        while True:
            yield (x, y)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    #Returns the number of unknown neighbors near a square 
    def unknown_neighbors(self, grid_coords): 
        x, y = grid_coords 
        total = 0
        if self.are_valid_grid((x-1, y-1)) and self.occ_grid[(y-1, x-1)] == self.UNKNOWN:
            total = total+1 
        if self.are_valid_grid((x, y-1)) and self.occ_grid[(y-1, x)] == self.UNKNOWN:
            total = total+1 
        if self.are_valid_grid((x+1, y-1)) and self.occ_grid[(y-1, x+1)] == self.UNKNOWN:
            total = total+1 
        if self.are_valid_grid((x-1, y)) and self.occ_grid[(y, x-1)] == self.UNKNOWN:
            total = total+1 
        if self.are_valid_grid((x+1, y)) and self.occ_grid[(y, x+1)] == self.UNKNOWN:
            total = total+1 
        if self.are_valid_grid((x-1, y+1)) and self.occ_grid[(y+1, x-1)] == self.UNKNOWN:
            total = total+1 
        if self.are_valid_grid((x, y+1)) and self.occ_grid[(y+1, x)] == self.UNKNOWN:
            total = total+1 
        if self.are_valid_grid((x+1, y+1)) and self.occ_grid[(y+1, x+1)] == self.UNKNOWN:
            total = total+1 
        return total 
    
    def wall_count(self, grid_coords): 
        x, y = grid_coords 
        total = 0
        if self.are_valid_grid((x-1, y-1)) and self.occ_grid[(y-1, x-1)] == self.WALL:
            total = total+1 
        if self.are_valid_grid((x, y-1)) and self.occ_grid[(y-1, x)] == self.WALL:
            total = total+1 
        if self.are_valid_grid((x+1, y-1)) and self.occ_grid[(y-1, x+1)] == self.WALL:
            total = total+1 
        if self.are_valid_grid((x-1, y)) and self.occ_grid[(y, x-1)] == self.WALL:
            total = total+1 
        if self.are_valid_grid((x+1, y)) and self.occ_grid[(y, x+1)] == self.WALL:
            total = total+1 
        if self.are_valid_grid((x-1, y+1)) and self.occ_grid[(y+1, x-1)] == self.WALL:
            total = total+1 
        if self.are_valid_grid((x, y+1)) and self.occ_grid[(y+1, x)] == self.WALL:
            total = total+1 
        if self.are_valid_grid((x+1, y+1)) and self.occ_grid[(y+1, x+1)] == self.WALL:
           total = total+1 
        return total 
    
class GoalFinder(): 
    #Initializer 
    def __init__(self, map): 
        self.map = map 
        self.cur_goal = None 

    #Getters 
    def get_map(self):  
        return self.map
    def get_cur_goal(self): 
        return self.cur_goal 
    
    #Public Methods
    #Calculates the nearest goal using lidar by sweeping from -3pi/4 to 3pi/4. 
    #Returns a pair of coordinates (x, y) that correspond to the furthest coordinates away. 
    def find_goal(self, robot_coords): 
        #Set goal to the end of the path unless the path is really long then choose a closer point. 
        #This should prevent the robot from having to travel long distances. 
        # Choose a goal along the BFS-generated path. If the endpoint is
        # farther than max_path_dist, pick the first node along the path
        # (searching from goal back toward start) that lies within
        # max_path_dist (meters) of the robot.
        max_path_dist = 1.5
        if self.cur_goal is not None:
            self.map.set_occ_grid_viewed(self.cur_goal[1])
        robot_x, robot_y = robot_coords
        start_grid = self.map.to_grid_coords((robot_x, robot_y))
        path = self._BFS(start_grid)
        # _BFS returns none if nothing was found. 
        if not path:
            return None
        goal_grid = path[-1]
        goal_world = self.map.to_world_coords(goal_grid)
        # If the final goal is far, try to select a closer node along the path
        for node in reversed(path):
            node_world = self.map.to_world_coords(node)
            dist = math.hypot(node_world[0] - robot_x, node_world[1] - robot_y)
            if dist <= max_path_dist and self.map.get_occ_grid()[node[1], node[0]] != self.map.VISITED:
                goal_grid = node
                goal_world = node_world
                break
        self.cur_goal = (goal_world, goal_grid)
        self.map.set_occ_grid_goal(goal_grid)
        return goal_world
    
    #Returns true if the robot is within a set distance of the goal point. 
    def near_goal(self, robot_coords, max_dist=1): 
        if self.cur_goal == None: 
            return False 
        return ((robot_coords[1] - self.cur_goal[0][1])**2 + (robot_coords[0] - self.cur_goal[0][0])**2) <= (max_dist**2) 
    
    #This function should be called when the PID controller is able to navigate to 
    #the goal selected by find_goal. This ensures that the goal will be stored 
    #and not picked again. 
    def set_goal(self): 
        if self.cur_goal != None: 
           #Sets the goal reached on the occ grid for debugging purposes. 
           self.map.set_occ_grid_visited(self.cur_goal[1])
           self.cur_goal = None 

    #Private Methods 
    def _valid_node(self, visited, coords): 
        #A node may only be added to the queue if
        #it has not already been added 
        #it is not a wall 
        #it was not a previous goal 
        if (self.map.are_valid_grid(coords) 
            and self.map.get_occ_grid()[coords[1], coords[0]] != self.map.WALL 
            and self.map.get_occ_grid()[coords[1], coords[0]] != self.map.UNKNOWN
            and visited.get_occ_grid()[coords[1], coords[0]] != visited.VISITED 
        ): return True 
        return False 
    
    def _BFS(self, rcoords):
        #Needs grid coordinates 
        grid_queue = Queue() 
        #Using another map here to keep track of visited nodes. 
        visited = Map(self.map.get_dimension(), self.map.get_resolution())
        #Place the robot's starting position in queue and in the visited map.  
        grid_queue.push((rcoords, [rcoords]))
        while not grid_queue.is_empty(): 
            temp_coords, temp_path = grid_queue.pop() 
            visited.set_occ_grid_visited(temp_coords)
            if (
                self.map.get_occ_grid()[temp_coords[1], temp_coords[0]] != self.map.VISITED 
                and self.map.get_occ_grid()[temp_coords[1], temp_coords[0]] == self.map.VIEWED
                and self.map.wall_count(temp_coords) == 0 
                and self.map.unknown_neighbors(temp_coords) > 0
            ): return temp_path  
            #Check if any neighbors need to be searched. 
            possible_neighbors = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
            for i, j in possible_neighbors: 
                if self._valid_node(visited, (temp_coords[0] - i, temp_coords[1] - j)): 
                    grid_queue.push(((temp_coords[0]-i, temp_coords[1]-j), temp_path + [(temp_coords[0]-i, temp_coords[1]-j)]))
                    visited.set_occ_grid_visited((temp_coords[0]-i, temp_coords[1]-j))
        return None 

class Tracker(Node):
    def __init__(self):
        super().__init__('Track')
        self.startTime = time.time()
        self.time = 0.1
        #self.timer = self.create_timer(self.time, self.timer_callback)
        self.GoalFinder = GoalFinder(Map(map_dimension, map_resolution))
        self.lidar_msg = None 
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
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.robot_x = 0
        self.robot_y = 0
        self.robot_theta = 0 
        self.robot_set = False 
        self.current_goal = None
        self.goal_reached = True  # Start with no goal
        self.goal_start_time = None  # Track when current goal was set
        self.stuck_counter = 0  # Track consecutive obstacle detections
        self.last_robot_pos = (0, 0)  # Track robot position for stuck detection
        # PID controller state
        self.prev_error_theta = 0.0
        self.integral_theta = 0.0
        self.prev_time = time.time()
        # Create a persistent figure and image for faster updates
        self.fig, self.ax = plt.subplots()
        self.img = None
    
    def listener_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = math.atan2(2*(msg.pose.pose.orientation.w*msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y), 
                                      1-2*(msg.pose.pose.orientation.y**2 + msg.pose.pose.orientation.z**2))
        self.robot_set = True 

    def sensor_callback(self, msg): 
        if self.robot_set: 
            grid = self.GoalFinder.map.get_occ_grid()
            self.GoalFinder.map.update_map((self.robot_x, self.robot_y, self.robot_theta), msg.ranges)
            # Mark robot's current grid square as explored
            robot_grid_x, robot_grid_y = self.GoalFinder.map.to_grid_coords((self.robot_x, self.robot_y))
            # Only find a new goal if we don't have a current goal or if we've reached the current goal
            if self.goal_reached:
                #self.GoalFinder.map.update_map((self.robot_x, self.robot_y, self.robot_theta), msg.ranges)
                print("Looking for new goal") 
                goal_world_coords = self.GoalFinder.find_goal((self.robot_x, self.robot_y))
                if goal_world_coords is not None: 
                    # Check if the goal is in a different grid square than the robot
                    goal_grid_x, goal_grid_y = self.GoalFinder.map.to_grid_coords((goal_world_coords[0], goal_world_coords[1]))
                    robot_grid_x, robot_grid_y = self.GoalFinder.map.to_grid_coords((self.robot_x, self.robot_y))
                    self.current_goal = goal_world_coords
                    self.goal_reached = False
                    self.goal_start_time = time.time()  # Record when goal was set
                    print(f"New goal found in different grid square: {goal_world_coords} -> ({goal_grid_x}, {goal_grid_y})")
                else:
                    print("No new goal found - all visible areas may be explored")
        
            # Move robot toward the current goal (if we have one)
            if self.current_goal is not None and not self.goal_reached:
                # Get goal grid coordinates first
                goal_grid_x, goal_grid_y = self.GoalFinder.map.to_grid_coords((self.current_goal[0], self.current_goal[1]))
            
                # Check for goal timeout (if stuck for more than 15 seconds, abandon goal)
                if self.goal_start_time is not None and (time.time() - self.goal_start_time) > 20.0:
                    print(f"Goal timeout! Abandoning goal after 15 seconds: ({goal_grid_x}, {goal_grid_y})")
                    self.GoalFinder.set_goal()
                    self.goal_reached = True
                    self.current_goal = None
                    self.goal_start_time = None
                    # Stop robot movement
                    twist = Twist()
                    self.publisher.publish(twist)
                else:
                    if self.GoalFinder.near_goal((self.robot_x, self.robot_y)): 
                        self.GoalFinder.map.update_map((self.robot_x, self.robot_y, self.robot_theta), msg.ranges)
                        print(f"Reached goal! Robot in same grid square as goal: ({robot_grid_x}, {robot_grid_y})")
                        self.GoalFinder.set_goal()
                        self.goal_reached = True
                        self.current_goal = None
                        self.goal_start_time = None
                        # Stop robot movement
                        twist = Twist()
                        self.publisher.publish(twist)
                        print("Stopped robot movement - goal reached")
                    else:
                        goal_error = math.sqrt((self.current_goal[0] - self.robot_x)**2 + (self.current_goal[1] - self.robot_y)**2)
                        print(f"Moving toward goal in grid square: ({goal_grid_x}, {goal_grid_y}), Distance: {goal_error:.2f}")
                        self.explore_control(self.current_goal[0], self.current_goal[1], msg.ranges)

        # Create the image on first callback, then update the data
        if self.img is None:
            self.img = self.ax.imshow(grid, cmap='viridis', origin='lower', vmin=0, vmax=255, interpolation='nearest')
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

    def explore_control(self, goal_x, goal_y, ranges):
        """
        Enhanced obstacle avoidance control for robot exploration.
        Moves robot toward goal while avoiding obstacles with low threshold.
        """
        # Increased threshold for obstacle avoidance
        obstacle_threshold = 1.0  # Much higher threshold for early detection
        
        # Check for obstacles in different directions
        num_readings = len(ranges)
        front_start = num_readings // 2 - 15  # Front 30 degrees
        front_end = num_readings // 2 + 15
        left_start = num_readings // 4 - 10   # Left 20 degrees
        left_end = num_readings // 4 + 10
        right_start = 3 * num_readings // 4 - 10  # Right 20 degrees
        right_end = 3 * num_readings // 4 + 10
        
        # Get minimum distances in each direction
        front_ranges = ranges[front_start:front_end]
        left_ranges = ranges[left_start:left_end]
        right_ranges = ranges[right_start:right_end]
        
        min_front = min(front_ranges) if front_ranges else float('inf')
        min_left = min(left_ranges) if left_ranges else float('inf')
        min_right = min(right_ranges) if right_ranges else float('inf')
        
        # Calculate movement error
        error_x = goal_x - self.robot_x
        error_y = goal_y - self.robot_y
        error_distance = math.sqrt(error_x**2 + error_y**2)
        
        # Calculate desired heading
        desired_theta = math.atan2(error_y, error_x)
        error_theta = desired_theta - self.robot_theta
        
        # Normalize angle to [-π, π]
        while error_theta > math.pi:
            error_theta -= 2 * math.pi
        while error_theta < -math.pi:
            error_theta += 2 * math.pi
        
        # PID Controller calculations
        current_time = time.time()
        dt = current_time - self.prev_time
        if dt <= 0:
            dt = 0.01  # Prevent division by zero
        
        # PID gains - reduced for smoother control
        kp = 0.3  # Reduced proportional gain for smoother turning
        ki = 0.07  # Reduced integral gain
        kd = 0.3  # Reduced derivative gain
        
        # Calculate PID terms
        # Proportional term
        p_term = kp * error_theta
        
        # Integral term (with windup protection)
        self.integral_theta += error_theta * dt
        # Limit integral to prevent windup
        max_integral = 2.0
        if self.integral_theta > max_integral:
            self.integral_theta = max_integral
        elif self.integral_theta < -max_integral:
            self.integral_theta = -max_integral
        i_term = ki * self.integral_theta
        
        # Derivative term
        derivative_theta = (error_theta - self.prev_error_theta) / dt
        d_term = kd * derivative_theta
        
        # Calculate PID output
        pid_output = p_term + i_term + d_term
        
        # Update previous values
        self.prev_error_theta = error_theta
        self.prev_time = current_time
        
        # Check if robot is stuck (not moving much)
        current_pos = (self.robot_x, self.robot_y)
        distance_moved = math.sqrt((current_pos[0] - self.last_robot_pos[0])**2 + 
                                 (current_pos[1] - self.last_robot_pos[1])**2)
        
        # Update stuck counter
        if distance_moved < 0.1:  # Robot hasn't moved much
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.last_robot_pos = current_pos
        
        # Create twist message
        twist = Twist()
        
        # If stuck for too long, try backing up and turning
        if self.stuck_counter > 20:  # Stuck for 20 iterations
            print(f"Robot appears stuck! Backing up and turning (stuck for {self.stuck_counter} iterations)")
            twist.linear.x = -10.0  # Back up
            twist.angular.z = 1.5  # Turn right
            self.stuck_counter = 0  # Reset counter
            self.publisher.publish(twist)
            return
        
        # Integrated obstacle avoidance with PID goal seeking
        if min_front < obstacle_threshold:
            print(f"Front obstacle at {min_front:.2f}m - avoiding while seeking goal")
            # Obstacle in front - turn toward the side with MORE space
            if min_left > min_right:
                avoidance_turn = -0.3  # Turn LEFT toward more space
                print(f"Turning left to avoid front obstacle (left: {min_left:.2f}m, right: {min_right:.2f}m)")
            else:
                avoidance_turn = 0.3  # Turn RIGHT toward more space
                print(f"Turning right to avoid front obstacle (left: {min_left:.2f}m, right: {min_right:.2f}m)")
            # Combine avoidance with PID goal seeking (reduced PID influence)
            twist.angular.z = avoidance_turn + pid_output * 0.7
            # Reduce speed more when obstacle is closer
            if min_front < 0.4:
                twist.linear.x = 0.05  # Very slow when close
            else:
                twist.linear.x = 0.25  # Moderate forward movement
            
        elif min_left < obstacle_threshold * 0.8:  # Higher threshold for sides
            print(f"Left obstacle at {min_left:.2f}m - turning right toward goal")
            # Obstacle on left - turn right but still seek goal
            avoidance_turn = 0.2  # Further reduced turn right
            # Combine avoidance with PID goal seeking
            twist.angular.z = avoidance_turn + pid_output * 0.7
            # Reduce speed when side obstacle is close
            if min_left < 0.7:
                twist.linear.x = 0.10  # Slower when close to side obstacle
            else:
                twist.linear.x = 0.25  # Faster forward movement
            
        elif min_right < obstacle_threshold * 0.8:
            print(f"Right obstacle at {min_right:.2f}m - turning left toward goal")
            # Obstacle on right - turn left but still seek goal
            avoidance_turn = -0.2  # Further reduced turn left
            # Combine avoidance with PID goal seeking
            twist.angular.z = avoidance_turn + pid_output * 0.7
            # Reduce speed when side obstacle is close
            if min_right < 0.7:
                twist.linear.x = 0.15  # Slower when close to side obstacle
            else:
                twist.linear.x = 0.25  # Faster forward movement
            
        else:
            # Safe to move toward goal - use full PID control
            # Reduce speed if getting close to obstacles (more proactive)
            speed_factor = 1.7
            if min_front < 1.0:
                speed_factor = 0.2 
            elif min_front < 1.3:  # Start slowing down much earlier
                speed_factor = 0.8  # Significant slow down when approaching obstacles
            elif min_front < 2.5:
                speed_factor = 1.0  # Moderate slow down
            elif min_left < 1.2 or min_right < 1.2:  # Earlier side detection
                speed_factor = 1.3  # Moderate speed reduction for side obstacles
            
            # PID control with speed adjustment
            twist.linear.x = min(0.5 * speed_factor, error_distance * 0.3 * speed_factor)
            twist.angular.z = pid_output  # Use full PID output
            print(f"PID control - P: {p_term:.2f}, I: {i_term:.2f}, D: {d_term:.2f}, Speed: {speed_factor:.1f}")
        
        # Publish movement command
        self.publisher.publish(twist)

def main(args=None):

    rclpy.init(args=args)
    tracker_node = Tracker()
    rclpy.spin(tracker_node)
    tracker_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
