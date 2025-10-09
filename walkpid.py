# This version works pretty well

import rclpy
from rclpy.node import Node
import time
import math
import numpy as np 
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt 
import os
import signal
import sys
from datetime import datetime
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
    #Calculates the goal using lidar by sweeping from -3pi/4 to 3pi/4. 
    #Returns a pair of coordinates (x, y) that correspond to the furthest coordinates away from starting point. 
    def find_goal(self, robot_info, ranges, starting_x=None, starting_y=None): 
        # Lidar parameters
        num_readings = 270
        angle_min = -3 * math.pi / 4
        angle_max = 3 * math.pi / 4
        angle_increment = (angle_max - angle_min) / (num_readings - 1)
        robot_x, robot_y, robot_theta = robot_info 
        
        # Collect all frontier goals (points at the end of lidar rays)
        frontier_goals = []
        self._clear_occ_grid() 
        
        for i in range(num_readings):
            r = ranges[i]
            lidar_theta = angle_min + i * angle_increment + robot_theta
            Lx = robot_x + r * math.cos(lidar_theta)
            Ly = robot_y + r * math.sin(lidar_theta)
            mx, my = self.to_grid_coords((Lx, Ly))
            if 0 <= mx < self.map_dimensions and 0 <= my < self.map_dimensions:
                # Check if this point is not already reached
                if (mx, my) not in self.goals_reached:
                    frontier_goals.append((mx, my, Lx, Ly, r))
        
        if not frontier_goals:
            print("No frontier goals found - all visible areas may be explored")
            return (None, None)
        
        # Choose the goal that is furthest from the starting point
        if starting_x is not None and starting_y is not None:
            # Calculate distance from starting point for each goal
            best_goal = None
            max_distance_from_start = -1
            
            for goal in frontier_goals:
                mx, my, Lx, Ly, r = goal
                distance_from_start = math.sqrt((Lx - starting_x)**2 + (Ly - starting_y)**2)
                
                if distance_from_start > max_distance_from_start:
                    max_distance_from_start = distance_from_start
                    best_goal = goal
            
            if best_goal is not None:
                mx, my, Lx, Ly, r = best_goal
                print(f"Selected frontier goal furthest from start: ({Lx:.2f}, {Ly:.2f}) at distance {max_distance_from_start:.2f}m from start")
            else:
                # Fallback to original method if no starting point provided
                best_goal = max(frontier_goals, key=lambda x: x[4])  # x[4] is the lidar range r
                mx, my, Lx, Ly, r = best_goal
                print(f"Selected frontier goal by lidar range: ({Lx:.2f}, {Ly:.2f}) at range {r:.2f}m")
        else:
            # Fallback to original method if no starting point provided
            best_goal = max(frontier_goals, key=lambda x: x[4])  # x[4] is the lidar range r
            mx, my, Lx, Ly, r = best_goal
            print(f"Selected frontier goal by lidar range (no start point): ({Lx:.2f}, {Ly:.2f}) at range {r:.2f}m")
        
        self.cur_goal = (mx, my)
        self.occ_grid[my, mx] = 127
        return (Lx, Ly) 
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
    print("Started Tracker")
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
        # Publisher for robot movement commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.robot_x = 0
        self.robot_y = 0
        self.robot_theta = 0 
        # Track current goal state
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
        # Distance tracking from starting point
        self.starting_x = None  # Will be set on first odometry callback
        self.starting_y = None
        self.max_distance_from_start = 0.0  # Track maximum distance achieved
        # Area timeout tracking (4 grid square area)
        self.area_start_time = None  # When robot entered current 4x4 area
        self.current_area_center = None  # Center of current 4x4 area (grid coordinates)
        self.area_timeout_duration = 30.0  # 30 seconds timeout
        self.escape_rotation_active = False  # Flag for escape rotation mode
        self.escape_rotation_start_time = None  # When escape rotation started
        self.escape_route_found = False  # Flag indicating good escape route found
        self.escape_moving_forward = False  # Flag indicating robot is moving forward to escape
        self.stuck_area_grid_coords = None  # Grid coordinates of the stuck 4x4 area
        self.escape_avoiding_obstacle = False  # Flag indicating robot is avoiding obstacle during escape
    def listener_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_theta = math.atan2(2*(msg.pose.pose.orientation.w*msg.pose.pose.orientation.z + msg.pose.pose.orientation.x * msg.pose.pose.orientation.y), 
                                      1-2*(msg.pose.pose.orientation.y**2 + msg.pose.pose.orientation.z**2))
        
        # Set starting position on first callback
        if self.starting_x is None:
            self.starting_x = self.robot_x
            self.starting_y = self.robot_y
            print(f"Starting position set: ({self.starting_x:.2f}, {self.starting_y:.2f})")
        
        # Calculate current distance from starting point
        current_distance = math.sqrt((self.robot_x - self.starting_x)**2 + (self.robot_y - self.starting_y)**2)
        
        # Update maximum distance if current distance is greater
        if current_distance > self.max_distance_from_start:
            self.max_distance_from_start = current_distance
            print(f"NEW MAX DISTANCE! Current: {current_distance:.2f}m, Max: {self.max_distance_from_start:.2f}m")
        
        # Check area timeout (4x4 grid square area)
        self._check_area_timeout()

    def _check_area_timeout(self):
        """
        Check if robot has been in the same 4x4 grid area for too long (30 seconds).
        If so, activate escape rotation mode.
        """
        if self.starting_x is None or self.starting_y is None:
            return  # Can't check area timeout without starting position
        
        # Get current robot grid coordinates
        current_grid_x, current_grid_y = self.GoalFinder.to_grid_coords((self.robot_x, self.robot_y))
        
        # Calculate the center of the 4x4 area the robot is currently in
        # Each 4x4 area is centered on multiples of 4
        area_center_x = (current_grid_x // 4) * 4 + 2  # Center of 4x4 area
        area_center_y = (current_grid_y // 4) * 4 + 2  # Center of 4x4 area
        current_area_center = (area_center_x, area_center_y)
        
        # Check if robot has moved to a different 4x4 area
        if self.current_area_center != current_area_center:
            # Robot has moved to a new area
            self.current_area_center = current_area_center
            self.area_start_time = time.time()
            self.escape_rotation_active = False
            self.escape_rotation_start_time = None
            print(f"Robot entered new 4x4 area: center ({area_center_x}, {area_center_y})")
        else:
            # Robot is still in the same area, check if timeout has been reached
            if self.area_start_time is not None:
                time_in_area = time.time() - self.area_start_time
                if time_in_area >= self.area_timeout_duration and not self.escape_rotation_active:
                    print(f"AREA TIMEOUT! Robot stuck in 4x4 area for {time_in_area:.1f}s - activating escape rotation")
                    self.escape_rotation_active = True
                    self.escape_rotation_start_time = time.time()
                    self.escape_route_found = False
                    self.escape_moving_forward = False
                    self.escape_avoiding_obstacle = False
                    self.stuck_area_grid_coords = current_area_center  # Store the stuck area coordinates
                    # Force goal to be reached so robot will look for new goal after rotation
                    self.goal_reached = True
                    self.current_goal = None
                    self.goal_start_time = None

    def _handle_escape_rotation(self, ranges):
        """
        Handle intelligent escape rotation when robot is stuck in the same 4x4 area for too long.
        Rotates until it finds a good escape route, then moves forward to escape the area.
        """
        if self.escape_rotation_start_time is None:
            self.escape_rotation_start_time = time.time()
            print("Starting intelligent escape rotation to find escape route")
        
        # Check if we've found a good escape route and are moving forward
        if self.escape_route_found and self.escape_moving_forward:
            self._handle_escape_movement(ranges)
            return
        
        # Check if we've found a good escape route
        if not self.escape_route_found:
            good_route = self._check_for_escape_route(ranges)
            if good_route:
                print("Good escape route found! Moving forward to escape area")
                self.escape_route_found = True
                self.escape_moving_forward = True
                self._handle_escape_movement(ranges)
                return
        
        # Continue rotating to find escape route
        twist = Twist()
        twist.linear.x = 0.0  # Stop forward movement
        twist.angular.z = 0.6  # Rotate at moderate speed
        self.publisher.publish(twist)
        
        rotation_duration = time.time() - self.escape_rotation_start_time
        print(f"Escape rotation in progress: {rotation_duration:.1f}s - searching for escape route")
        
        # Safety timeout - if we can't find a route after 10 seconds, force escape
        if rotation_duration > 10.0:
            print("Escape rotation timeout - forcing escape attempt")
            self.escape_route_found = True
            self.escape_moving_forward = True

    def _check_for_escape_route(self, ranges):
        """
        Check if there's a good escape route in the front direction.
        Returns True if there's a clear path forward.
        """
        # Check front 60 degrees for obstacles
        num_readings = len(ranges)
        front_start = num_readings // 2 - 30  # Front 60 degrees
        front_end = num_readings // 2 + 30
        
        front_ranges = ranges[front_start:front_end]
        min_front = min(front_ranges) if front_ranges else float('inf')
        
        # Good escape route if front is clear for at least 2 meters
        escape_threshold = 2.0
        return min_front > escape_threshold

    def _handle_escape_movement(self, ranges):
        """
        Handle forward movement to escape the stuck area with obstacle detection.
        """
        # Get current robot grid coordinates
        current_grid_x, current_grid_y = self.GoalFinder.to_grid_coords((self.robot_x, self.robot_y))
        
        # Check if we've escaped the stuck 4x4 area
        if self.stuck_area_grid_coords is not None:
            stuck_area_x, stuck_area_y = self.stuck_area_grid_coords
            current_area_x = (current_grid_x // 4) * 4 + 2
            current_area_y = (current_grid_y // 4) * 4 + 2
            
            # Check if we're in a different 4x4 area
            if current_area_x != stuck_area_x or current_area_y != stuck_area_y:
                print("Successfully escaped the stuck area! Marking area as visited and resuming normal operation")
                self._mark_stuck_area_as_visited()
                self._reset_escape_state()
                return
        
        # Check for obstacles in front 30 degrees during escape movement
        if not self.escape_avoiding_obstacle:
            obstacle_detected = self._check_escape_obstacle(ranges)
            if obstacle_detected:
                print("Obstacle detected during escape! Stopping and rotating to avoid")
                self.escape_avoiding_obstacle = True
                self.escape_rotation_start_time = time.time()  # Reset rotation timer
                return
        
        # If we're avoiding an obstacle, continue rotating
        if self.escape_avoiding_obstacle:
            self._handle_escape_obstacle_avoidance(ranges)
            return
        
        # Continue moving forward to escape (no obstacles detected)
        twist = Twist()
        twist.linear.x = 0.4  # Move forward at moderate speed
        twist.angular.z = 0.0  # No rotation
        self.publisher.publish(twist)
        print("Moving forward to escape stuck area")

    def _check_escape_obstacle(self, ranges):
        """
        Check for obstacles in the front 30 degrees during escape movement.
        Returns True if obstacle is detected.
        """
        # Check front 30 degrees for obstacles
        num_readings = len(ranges)
        front_start = num_readings // 2 - 15  # Front 30 degrees
        front_end = num_readings // 2 + 15
        
        front_ranges = ranges[front_start:front_end]
        min_front = min(front_ranges) if front_ranges else float('inf')
        
        # Obstacle detected if front is closer than 1.5 meters
        obstacle_threshold = 1.5
        return min_front < obstacle_threshold

    def _handle_escape_obstacle_avoidance(self, ranges):
        """
        Handle obstacle avoidance during escape movement.
        Rotates until clear path is found, then resumes forward movement.
        """
        if self.escape_rotation_start_time is None:
            self.escape_rotation_start_time = time.time()
        
        # Check if obstacle is cleared
        obstacle_cleared = not self._check_escape_obstacle(ranges)
        
        if obstacle_cleared:
            print("Obstacle cleared during escape! Resuming forward movement")
            self.escape_avoiding_obstacle = False
            self.escape_rotation_start_time = None
            # Continue with forward movement
            twist = Twist()
            twist.linear.x = 0.4  # Move forward at moderate speed
            twist.angular.z = 0.0  # No rotation
            self.publisher.publish(twist)
            print("Resuming forward escape movement")
        else:
            # Continue rotating to avoid obstacle
            rotation_duration = time.time() - self.escape_rotation_start_time
            twist = Twist()
            twist.linear.x = 0.0  # Stop forward movement
            twist.angular.z = 0.6  # Rotate at moderate speed
            self.publisher.publish(twist)
            print(f"Escape obstacle avoidance: rotating for {rotation_duration:.1f}s")
            
            # Safety timeout - if we can't clear obstacle after 5 seconds, force forward
            if rotation_duration > 5.0:
                print("Escape obstacle avoidance timeout - forcing forward movement")
                self.escape_avoiding_obstacle = False
                self.escape_rotation_start_time = None

    def _mark_stuck_area_as_visited(self):
        """
        Mark the entire stuck 4x4 area as visited so robot never returns.
        """
        if self.stuck_area_grid_coords is None:
            return
        
        stuck_area_x, stuck_area_y = self.stuck_area_grid_coords
        
        # Mark all 16 grid squares in the 4x4 area as visited
        for dx in range(-2, 2):  # -2, -1, 0, 1
            for dy in range(-2, 2):  # -2, -1, 0, 1
                grid_x = stuck_area_x + dx
                grid_y = stuck_area_y + dy
                
                # Check if coordinates are within map bounds
                if (0 <= grid_x < self.GoalFinder.map_dimensions and 
                    0 <= grid_y < self.GoalFinder.map_dimensions):
                    # Mark as visited (255) in the occupancy grid
                    self.GoalFinder.occ_grid[grid_y, grid_x] = 255
                    # Add to goals_reached list to prevent future selection
                    if (grid_x, grid_y) not in self.GoalFinder.goals_reached:
                        self.GoalFinder.goals_reached.append((grid_x, grid_y))
        
        print(f"Marked entire 4x4 area centered at ({stuck_area_x}, {stuck_area_y}) as visited")

    def _reset_escape_state(self):
        """
        Reset all escape-related state variables.
        """
        self.escape_rotation_active = False
        self.escape_rotation_start_time = None
        self.escape_route_found = False
        self.escape_moving_forward = False
        self.escape_avoiding_obstacle = False
        self.stuck_area_grid_coords = None
        # Reset area tracking to start fresh
        self.area_start_time = time.time()
        self.current_area_center = None

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

        # Get occupancy grid
        grid = self.GoalFinder.get_occ_grid()
        
        # Mark robot's current grid square as explored
        robot_grid_x, robot_grid_y = self.GoalFinder.to_grid_coords((self.robot_x, self.robot_y))
        if 0 <= robot_grid_x < self.GoalFinder.map_dimensions and 0 <= robot_grid_y < self.GoalFinder.map_dimensions:
            grid[robot_grid_y, robot_grid_x] = 255  # Mark as explored (white)
            # Calculate current distance from starting point
            current_distance = math.sqrt((self.robot_x - self.starting_x)**2 + (self.robot_y - self.starting_y)**2)
            print(f"Robot in grid square: ({robot_grid_x}, {robot_grid_y}) - Distance from start: {current_distance:.2f}m, Max distance: {self.max_distance_from_start:.2f}m")
        
        # Handle escape rotation mode if active
        if self.escape_rotation_active:
            self._handle_escape_rotation(msg.ranges)
            return  # Skip normal goal finding and movement during escape rotation
        
        # Only find a new goal if we don't have a current goal or if we've reached the current goal
        if self.goal_reached:
            print(f"Looking for new goal. Reached goals: {self.GoalFinder.goals_reached}")
            goal_world_coords = self.GoalFinder.find_goal((self.robot_x, self.robot_y, self.robot_theta), msg.ranges, self.starting_x, self.starting_y)
            if goal_world_coords[0] is not None and goal_world_coords[1] is not None:
                # Check if the goal is in a different grid square than the robot
                goal_grid_x, goal_grid_y = self.GoalFinder.to_grid_coords((goal_world_coords[0], goal_world_coords[1]))
                robot_grid_x, robot_grid_y = self.GoalFinder.to_grid_coords((self.robot_x, self.robot_y))
                
                if goal_grid_x != robot_grid_x or goal_grid_y != robot_grid_y:
                    self.current_goal = goal_world_coords
                    self.goal_reached = False
                    self.goal_start_time = time.time()  # Record when goal was set
                    current_distance = math.sqrt((self.robot_x - self.starting_x)**2 + (self.robot_y - self.starting_y)**2)
                    print(f"New goal found in different grid square: {goal_world_coords} -> ({goal_grid_x}, {goal_grid_y}) - Distance from start: {current_distance:.2f}m, Max distance: {self.max_distance_from_start:.2f}m")
                else:
                    print(f"Goal found but in same grid square as robot: ({robot_grid_x}, {robot_grid_y}) - skipping")
            else:
                print("No new goal found - all visible areas may be explored")
        
        # Move robot toward the current goal (if we have one)
        if self.current_goal is not None and not self.goal_reached:
            # Get goal grid coordinates first
            goal_grid_x, goal_grid_y = self.GoalFinder.to_grid_coords((self.current_goal[0], self.current_goal[1]))
            
            # Check for goal timeout (if stuck for more than 15 seconds, abandon goal)
            if self.goal_start_time is not None and (time.time() - self.goal_start_time) > 15.0:
                print(f"Goal timeout! Abandoning goal after 15 seconds: ({goal_grid_x}, {goal_grid_y})")
                self.GoalFinder.set_goal()  # Mark as reached to avoid selecting again
                self.goal_reached = True
                self.current_goal = None
                self.goal_start_time = None
                # Stop robot movement
                twist = Twist()
                self.publisher.publish(twist)
            else:
                # Check if robot is in the same grid square as the goal
                robot_grid_x, robot_grid_y = self.GoalFinder.to_grid_coords((self.robot_x, self.robot_y))
                
                if robot_grid_x == goal_grid_x and robot_grid_y == goal_grid_y:
                    current_distance = math.sqrt((self.robot_x - self.starting_x)**2 + (self.robot_y - self.starting_y)**2)
                    print(f"Reached goal! Robot in same grid square as goal: ({robot_grid_x}, {robot_grid_y}) - Distance from start: {current_distance:.2f}m, Max distance: {self.max_distance_from_start:.2f}m")
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
                    current_distance = math.sqrt((self.robot_x - self.starting_x)**2 + (self.robot_y - self.starting_y)**2)
                    print(f"Moving toward goal in grid square: ({goal_grid_x}, {goal_grid_y}), Goal distance: {goal_error:.2f}m - Distance from start: {current_distance:.2f}m, Max distance: {self.max_distance_from_start:.2f}m")
                    self.explore_control(self.current_goal[0], self.current_goal[1], msg.ranges)

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
    
    def explore_control(self, goal_x, goal_y, ranges):
        """
        Enhanced obstacle avoidance control for robot exploration.
        Moves robot toward goal while avoiding obstacles with low threshold.
        """
        # Increased threshold for obstacle avoidance
        obstacle_threshold = 0.6  # Lowered threshold for closer detection
        
        # Check for obstacles in different directions
        num_readings = len(ranges)
        front_start = num_readings // 2 - 90  # Front 180 degrees (full semicircle)
        front_end = num_readings // 2 + 90
        
        # Narrow front collision detection (30 degrees)
        narrow_front_start = num_readings // 2 - 15  # Front 30 degrees for head-on collision
        narrow_front_end = num_readings // 2 + 15
        
        left_start = num_readings // 4 - 10   # Left 20 degrees
        left_end = num_readings // 4 + 10
        right_start = 3 * num_readings // 4 - 10  # Right 20 degrees
        right_end = 3 * num_readings // 4 + 10
        
        # Get minimum distances in each direction
        front_ranges = ranges[front_start:front_end]
        narrow_front_ranges = ranges[narrow_front_start:narrow_front_end]
        left_ranges = ranges[left_start:left_end]
        right_ranges = ranges[right_start:right_end]
        
        min_front = min(front_ranges) if front_ranges else float('inf')
        min_narrow_front = min(narrow_front_ranges) if narrow_front_ranges else float('inf')
        min_left = min(left_ranges) if left_ranges else float('inf')
        min_right = min(right_ranges) if right_ranges else float('inf')
        
        # Higher threshold for narrow front collision to prevent oscillation
        narrow_collision_threshold = obstacle_threshold * 1.5  # 0.9m threshold for narrow front
        
        # Analyze left and right halves of the wide front angle for obstacle bias
        front_center = len(front_ranges) // 2
        front_left_half = front_ranges[:front_center]  # Left half of front detection
        front_right_half = front_ranges[front_center:]  # Right half of front detection
        
        min_front_left = min(front_left_half) if front_left_half else float('inf')
        min_front_right = min(front_right_half) if front_right_half else float('inf')
        
        # Calculate obstacle bias - positive means right side has closer obstacles
        obstacle_bias = min_front_right - min_front_left
        
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
        ki = 0.05  # Reduced integral gain
        kd = 0.2  # Reduced derivative gain
        
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
        
        # Incorporate obstacle bias into PID control for smarter turning
        # obstacle_bias > 0: right side has closer obstacles, bias toward left turn (counterclockwise)
        # obstacle_bias < 0: left side has closer obstacles, bias toward right turn (clockwise)
        obstacle_bias_factor = 0.4  # Increased factor for more turning influence
        obstacle_influence = obstacle_bias * obstacle_bias_factor  # Positive bias = left turn, negative bias = right turn
        
        # Combine PID output with obstacle bias
        final_angular_output = pid_output + obstacle_influence
        
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
            twist.linear.x = -0.2  # Back up
            twist.angular.z = 0.5  # Turn right
            self.stuck_counter = 0  # Reset counter
            self.publisher.publish(twist)
            return
        
        # Check for head-on collision in narrow front angle (30 degrees)
        if min_narrow_front < narrow_collision_threshold:
            print(f"Head-on collision detected at {min_narrow_front:.2f}m in 30-degree front angle - stopping and rotating")
            twist.linear.x = 0.0  # Stop forward movement
            # Rotate based on obstacle bias to clear the narrow front
            if obstacle_bias > 0:  # Right side has closer obstacles
                twist.angular.z = 0.5  # Turn left to clear narrow front
                print(f"Turning left to clear narrow front (right side closer)")
            else:  # Left side has closer obstacles or equal
                twist.angular.z = -0.5  # Turn right to clear narrow front
                print(f"Turning right to clear narrow front (left side closer)")
            self.publisher.publish(twist)
            return
        
        # Integrated obstacle avoidance with PID goal seeking
        if min_front < obstacle_threshold:
            print(f"Front obstacle at {min_front:.2f}m - avoiding while seeking goal")
            print(f"Front obstacle bias: left={min_front_left:.2f}m, right={min_front_right:.2f}m, bias={obstacle_bias:.2f}")
            
            # Use obstacle bias to determine avoidance direction
            if obstacle_bias > 0.2:  # Right side has significantly closer obstacles
                avoidance_turn = 0.4  # Turn LEFT (counterclockwise) away from closer obstacles
                print(f"Turning left to avoid front obstacle (right side closer by {obstacle_bias:.2f}m)")
            elif obstacle_bias < -0.2:  # Left side has significantly closer obstacles
                avoidance_turn = -0.4  # Turn RIGHT (clockwise) away from closer obstacles
                print(f"Turning right to avoid front obstacle (left side closer by {-obstacle_bias:.2f}m)")
            else:  # Obstacles roughly equal on both sides, use goal-seeking bias
                avoidance_turn = final_angular_output * 0.6  # Use biased PID output with more influence
                print(f"Using goal-seeking bias for avoidance (bias={obstacle_bias:.2f})")
            
            # Combine avoidance with PID goal seeking (reduced PID influence)
            twist.angular.z = avoidance_turn + pid_output * 0.2
            # Reduce speed more when obstacle is closer
            if min_front < 0.8:
                twist.linear.x = 0.2  # Slow when close (increased from 0.1)
            else:
                twist.linear.x = 0.4  # Moderate forward movement (increased from 0.2)
            
        # elif min_left < obstacle_threshold * 0.8:  # Higher threshold for sides
        #     print(f"Left obstacle at {min_left:.2f}m - turning right toward goal")
        #     # Obstacle on left - turn right but still seek goal
        #     avoidance_turn = 0.2  # Further reduced turn right
        #     # Combine avoidance with PID goal seeking
        #     twist.angular.z = avoidance_turn + pid_output * 0.4
        #     # Reduce speed when side obstacle is close
        #     if min_left < 1.0:
        #         twist.linear.x = 0.15  # Slower when close to side obstacle
        #     else:
        #         twist.linear.x = 0.25  # Faster forward movement
            
        # elif min_right < obstacle_threshold * 0.8:
        #     print(f"Right obstacle at {min_right:.2f}m - turning left toward goal")
        #     # Obstacle on right - turn left but still seek goal
        #     avoidance_turn = -0.2  # Further reduced turn left
        #     # Combine avoidance with PID goal seeking
        #     twist.angular.z = avoidance_turn + pid_output * 0.4
        #     # Reduce speed when side obstacle is close
        #     if min_right < 1.0:
        #         twist.linear.x = 0.15  # Slower when close to side obstacle
        #     else:
        #         twist.linear.x = 0.25  # Faster forward movement
            
        else:
            # Safe to move toward goal - use full PID control
            # Reduce speed if getting close to obstacles (more proactive)
            speed_factor = 1.0
            if min_front < 2.0:  # Start slowing down much earlier
                speed_factor = 0.4  # Significant slow down when approaching obstacles
            elif min_front < 2.5:
                speed_factor = 0.6  # Moderate slow down
            elif min_left < 1.5 or min_right < 1.5:  # Earlier side detection
                speed_factor = 0.7  # Moderate speed reduction for side obstacles
            
            # PID control with speed adjustment and obstacle bias
            twist.linear.x = min(0.8 * speed_factor, error_distance * 0.4 * speed_factor)  # Increased max speed from 0.5 to 0.8
            twist.angular.z = final_angular_output  # Use obstacle-biased PID output
            current_distance = math.sqrt((self.robot_x - self.starting_x)**2 + (self.robot_y - self.starting_y)**2)
            print(f"PID control - P: {p_term:.2f}, I: {i_term:.2f}, D: {d_term:.2f}, Bias: {obstacle_influence:.2f}, Speed: {speed_factor:.1f} - Distance from start: {current_distance:.2f}m, Max distance: {self.max_distance_from_start:.2f}m")
        
        # Publish movement command
        self.publisher.publish(twist)
    
def main(args=None):
    print("Start")
    rclpy.init(args=args)
    tracker_node = Tracker()
    rclpy.spin(tracker_node)
    tracker_node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()