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
    
class Coords(): 
    #Coords are either grid coords or world coords(real coords). 
    GRID = 0 
    WORLD = 1 
    def __init__(self, xcoord, ycoord, type): 
        self.xcoord = xcoord 
        self.ycoord = ycoord 
        self.type = type 
    
    #Getters 
    def get_xcoord(self): 
        return self.xcoord
    
    def get_ycoord(self): 
        return self.ycoord 
    
    def get_coords(self): 
        return (self.xcoord, self.ycoord) 
    
    def get_type(self): 
        return self.type 
    
    #Setters 
    def set_xcoord(self, xcoord): 
        self.xcoord = xcoord 
    
    def set_ycoord(self, ycoord): 
        self.ycoord = ycoord 
    
    def set_coords(self, coords): 
        self.xcoord, self.ycoord = coords 
    
    def set_type(self, type): 
        self.type = type 

    #Overrides 
    def __eq__(self, other):
        if not isinstance(other, Coords): 
            return NotImplemented 
        return self.xcoord == other.xcoord and self.ycoord == other.ycoord and self.type == other.type 


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
    def __init__(self, map_dimension, resolution, wall_padding=0, starting_pos=None): 
        self.map_origin_x = map_dimension / 2
        self.map_origin_y = map_dimension/ 2 
        self.map_dimension = map_dimension 
        # What real world length each tile actually represents. 
        self.resolution = resolution 
        # wall_padding stored in meters; internal name avoids shadowing method
        self.wall_padding = wall_padding 
        self.starting_pos = starting_pos 
        #Keeps track of the furthest point from the starting position. 
        self.furthest_point = starting_pos 
        self.occ_grid = np.zeros((map_dimension, map_dimension), dtype = int)

    #Getters 
    def get_dimension(self): 
        return self.map_dimension
    
    def get_resolution(self): 
        return self.resolution
    
    def get_occ_grid(self): 
        return self.occ_grid 
    
    def get_furthest_point(self): 
        return self.furthest_point
    
    def get_starting_pos(self): 
        return self.starting_pos
    
    #Public Methods. 
    def find_furthest(self): 
        for i in range(self.map_dimension): 
            for j in range(self.map_dimension): 
                grid_coords = Coords(i, j, Coords.GRID)
                if self.check_space(grid_coords) == self.VIEWED: 
                    world_point = self.to_world_coords(grid_coords)
                    if (math.hypot(world_point.get_xcoord() - self.starting_pos.get_xcoord(), world_point.get_ycoord() - self.starting_pos.get_ycoord()) 
                    >math.hypot(self.furthest_point.get_xcoord() - self.starting_pos.get_xcoord(), self.furthest_point.get_ycoord() - self.starting_pos.get_ycoord())
                ): self.furthest_point = world_point

    #Converts world coordinates into grid coordinates. 
    def to_grid_coords(self, world_coords):
        if not isinstance(world_coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        if world_coords.get_type() != world_coords.WORLD: 
            raise ValueError("Coordinates Must Be WORLD")
        if not self.are_valid_world(world_coords): 
            raise ValueError("Coordinates Out Of World Bounds")
        # coords are world coordinates (rx, ry) in meters
        rx, ry = world_coords.get_coords()
        # convert meters -> cells, then offset by origin and floor to get integer cell index
        mx = int(math.floor(self.map_origin_x + (rx / self.resolution)))
        my = int(math.floor(self.map_origin_y + (ry / self.resolution)))
        # clamp to valid range
        mx = max(0, min(self.map_dimension - 1, mx))
        my = max(0, min(self.map_dimension - 1, my))
        return Coords(mx, my, Coords.GRID) 

    #Returns the world coordinates to the middle of the grid square. 
    def to_world_coords(self, grid_coords):
        if not isinstance(grid_coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        if grid_coords.get_type() != grid_coords.GRID: 
            raise ValueError("Coordinates Must Be GRID")
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        gx, gy = grid_coords.get_coords()
        wx = (gx - self.map_origin_x) * self.resolution + self.resolution / 2.0
        wy = (gy - self.map_origin_y) * self.resolution + self.resolution / 2.0
        return Coords(wx, wy, Coords.WORLD)
    
    #Checks if a pair of coordinates will fit within the grid. 
    def are_valid_grid(self, grid_coords): 
        if not isinstance(grid_coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        if grid_coords.get_type() != grid_coords.GRID: 
            raise ValueError("Coordinates Must Be GRID")
        if 0 <= grid_coords.get_xcoord()< self.map_dimension and 0 <= grid_coords.get_ycoord() < self.map_dimension: 
            return True 
        return False  
    
    #Checks if a pair of world coordinates will fit in the grid. 
    def are_valid_world(self, world_coords): 
        if not isinstance(world_coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        if world_coords.get_type() != world_coords.WORLD: 
            raise ValueError("Coordinates Must Be World")
        return True  
    
    #Returns the value of a space in the occ_grid. 
    def check_space(self, grid_coords): 
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        return self.occ_grid[grid_coords.get_ycoord(), grid_coords.get_xcoord()] 
    #Setters 
    #Methods for setting the value of a grid square. 
    def set_occ_grid_wall(self, grid_coords): 
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        xcoord, ycoord = grid_coords.get_coords() 
        self.occ_grid[ycoord, xcoord] = self.WALL 

    #Sets a square in the grid to the value passed in.  
    def set_occ_grid_viewed(self, grid_coords): 
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        xcoord, ycoord = grid_coords.get_coords() 
        self.occ_grid[ycoord, xcoord] = self.VIEWED

    def set_occ_grid_goal(self, grid_coords): 
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        xcoord, ycoord = grid_coords.get_coords() 
        self.occ_grid[ycoord, xcoord] = self.GOAL

    def set_occ_grid_visited(self, grid_coords): 
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        xcoord, ycoord = grid_coords.get_coords() 
        self.occ_grid[ycoord, xcoord] = self.VISITED 

    #Uses breenham algorithm to calculate and return the number of squares that are 
    #touched by the line between two endpoints. 
    def _bresenham(self, start_coords, end_coords):
        if not isinstance(start_coords, Coords) or not isinstance(end_coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        if start_coords.get_type() != Coords.GRID or end_coords.get_type() != Coords.GRID: 
            raise ValueError("Coordinates Must Be Grid")
        x0, y0 = start_coords.get_coords() 
        x1, y1 = end_coords.get_coords()  
        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy 
        x, y = x0, y0
        #Using yield here to avoid repeated operations. 
        while True:
            yield Coords(x, y, Coords.GRID)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        
    #Adds extra walls around a wall to help robot navigate. 
    def wall_padding(self, grid_coords): 
        if not isinstance(grid_coords, Coords): 
            raise ValueError("Data Must Be Coords")

        pad_cells = int(math.ceil(self.wall_padding / float(self.resolution)))

        cx, cy = grid_coords.get_coords()
        # Iterate over square of neighbors within pad_cells and set them to WALL
        for x in range(-pad_cells, pad_cells + 1):
            for y in range(-pad_cells, pad_cells + 1):
                nx = cx + x
                ny = cy + y
                # check bounds
                if 0 <= nx < self.map_dimension and 0 <= ny < self.map_dimension:
                    cell = Coords(nx, ny, Coords.GRID)
                    # Do not overwrite visited or goal markers
                    cur = self.check_space(cell)
                    if cur not in (self.VISITED, self.GOAL, self.WALL):
                        if (x*x + y*y) <= (pad_cells * pad_cells):
                            self.occ_grid[ny, nx] = self.WALL

    #Uses lidar ranges to update the internal map. 
    def update_map(self, robot_coords, robot_theta, ranges): 
        if not self.are_valid_world(robot_coords): 
            raise ValueError("Robot Coordinates are outside boundaries")
        #Lidar parameters. 
        max_range = 5 
        num_readings = len(ranges) 
        angle_min = -3 * math.pi / 4
        angle_max = 3 * math.pi / 4
        angle_increment = (angle_max - angle_min) / (num_readings - 1)
        #Robot data.  
        #Loop through all of the lidar readings and select the furthest point away. 
        for i in range(num_readings):
            lidar_theta = angle_min + i * angle_increment + robot_theta
            temp_coords = Coords(robot_coords.get_xcoord() + ranges[i] * math.cos(lidar_theta), robot_coords.get_ycoord() + ranges[i] * math.sin(lidar_theta), Coords.WORLD)
            temp_grid_coords = self.to_grid_coords(temp_coords) 
            #print(self.furthest_point.get_xcoord())
            #print(self.furthest_point.get_ycoord())
            #Lidar points that go outside of the map are ignored. 
            if self.are_valid_world(temp_coords): 
                #Check if lidar has gone its max range. If not then it has hit a wall. 
                if (ranges[i] != max_range ): 
                    self.set_occ_grid_wall(temp_grid_coords)
                #Find all of the squares that lidar has passed through and set to to "viewed" if they contained no previous state. 
                points = self._bresenham(self.to_grid_coords(robot_coords), temp_grid_coords) 
                if points: 
                    for j in points: 
                        if self.check_space(j) == self.UNKNOWN and self.are_valid_grid(j): 
                            self.set_occ_grid_viewed(j) 

    #Returns the number of unknown neighbors near a square 
    def unknown_neighbors(self, grid_coords): 
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        possible_neighbors = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
        x, y = grid_coords.get_coords() 
        total = 0
        for i, j in possible_neighbors: 
            if self.are_valid_grid(Coords(x+i, y+j, Coords.GRID)) and self.check_space(Coords(x+i, y+j, Coords.GRID)) == self.UNKNOWN: 
                total = total +1 
        return total 
    
    def wall_count(self, grid_coords): 
        if not self.are_valid_grid(grid_coords): 
            raise ValueError("Coordinates Out Of Grid Bounds")
        possible_neighbors = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
        x, y = grid_coords.get_coords() 
        total = 0 
        for i, j in possible_neighbors: 
            if self.are_valid_grid(Coords(x+i, y+j, Coords.GRID)) and self.check_space(Coords(x+i, y+j, Coords.GRID)) == self.WALL: 
                total = total +1 
        return total
    
class GoalFinder(): 
    #Initializer 
    #Max_path_dist is the maximum distance away from the robot that the goal finder can place a point. 
    #Goal_dist is how close the robot can be in to a goal before it is registered. 
    def __init__(self, max_path_dist, goal_dist): 
        self.cur_goal = None 
        self.max_path_dist = max_path_dist
        self.goal_dist = goal_dist

    #Getters 
    def get_cur_goal(self): 
        return self.cur_goal 
    
    #Public Methods
    #Calculates the nearest goal using lidar by sweeping from -3pi/4 to 3pi/4. 
    #Returns a pair of coordinates (x, y) that correspond to the furthest coordinates away. 
    def find_goal(self, robot_coords, map, find_furthest): 
        if not isinstance(map, Map): 
            raise TypeError("Data Must Be Map")
        if not isinstance(robot_coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        #Set goal to the end of the path unless the path is really long then choose a closer point. 
        #This should prevent the robot from having to travel long distances. 
        # Choose a goal along the BFS-generated path. If the endpoint is
        # farther than max_path_dist, pick the first node along the path
        # (searching from goal back toward start) that lies within
        # max_path_dist (meters) of the robot.
        if self.cur_goal is not None:
            map.set_occ_grid_viewed(self.cur_goal)
            self.cur_goal = None 
        if find_furthest: 
            map.find_furthest()
            print(map.to_grid_coords(map.get_furthest_point()).get_coords())
            path = self._BFS(robot_coords, map, self._Furthest)
        else: 
            path = self._BFS(robot_coords, map, self._Frontier)
        # _BFS returns none if nothing was found. 
        if not path:
            return None
        goal_coords = path[-1]
        # If the final goal is far, try to select a closer node along the path
        for node in reversed(path):
            node_world = map.to_world_coords(node)
            dist = math.hypot(node_world.get_xcoord() - robot_coords.get_xcoord(), node_world.get_ycoord() - robot_coords.get_ycoord())
            if dist <= self.max_path_dist: #and map.check_space(node)!= map.VISITED:
                goal_coords = node
                break
        self.cur_goal = goal_coords
        map.set_occ_grid_goal(goal_coords)
        return map.to_world_coords(goal_coords)
    
    #Returns true if the robot is within a set distance of the goal point. 
    def near_goal(self, robot_coords, map): 
        """Return True if robot_coords (WORLD) is within goal_dist (meters) of the current goal.

        self.cur_goal is stored as GRID coordinates by find_goal, so convert it to
        world coordinates using the provided map before computing Euclidean distance.
        """
        if not isinstance(robot_coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        if not isinstance(map, Map):
            raise TypeError("Data Must Be Map")
        if robot_coords.get_type() != Coords.WORLD:
            raise ValueError("robot_coords must be WORLD coordinates")
        if self.cur_goal is None:
            return False
        goal_world = map.to_world_coords(self.cur_goal)
        dx = robot_coords.get_xcoord() - goal_world.get_xcoord()
        dy = robot_coords.get_ycoord() - goal_world.get_ycoord()
        return (dx*dx + dy*dy) <= (self.goal_dist**2)
    
    #This function should be called when the PID controller is able to navigate to 
    #the goal selected by find_goal. This ensures that the goal will be stored 
    #and not picked again. 
    def set_goal(self, map): 
        if self.cur_goal != None: 
           #Sets the goal reached on the occ grid for debugging purposes. 
           map.set_occ_grid_visited(self.cur_goal)
           self.cur_goal = None 

    #Private Methods 
    def _valid_node(self, coords, map, visited): 
        if not isinstance(map, Map) or not isinstance(visited, Map): 
            raise TypeError("Data Must Be Map")
        if not isinstance(coords, Coords): 
            raise TypeError("Data Must Be Coords") 
        #A node may only be added to the queue if
        #it has not already been added 
        #it is not a wall 
        #it was not a previous goal 
        if (map.are_valid_grid(coords) 
            and map.check_space(coords) != map.WALL 
            and map.check_space(coords) != map.UNKNOWN
            and visited.check_space(coords) != visited.VISITED 
        ): return True 
        return False 
    
    #For use as a BFS goal. Returns true if the coords are the furthest point detected on the map. 
    def _Furthest(self, grid_coords, map): 
        if grid_coords == map.to_grid_coords(map.get_furthest_point()): 
            print("Furhtest Goal Found")
            return True 
        return False 
    
    #For use as a BFS goal finding algorithm. Returns true if a point has unknown neighbors and is not near a wall. 
    def _Frontier(self, grid_coords, map): 
        if (map.check_space(grid_coords) == map.VIEWED
            and map.wall_count(grid_coords) == 0 
            and map.unknown_neighbors(grid_coords) > 0
        ): return True 
        return False 
    
    def _BFS(self, rcoords, map, goal_func):
        if not isinstance(map, Map): 
            raise TypeError("Data Must Be Map") 
        if not map.are_valid_world(rcoords): 
            raise TypeError("Coords Not Within World")  
        #Needs grid coordinates 
        grid_queue = Queue() 
        #Using another map here to keep track of visited nodes. 
        visited = Map(map.get_dimension(), map.get_resolution())
        #Place the robot's starting position in queue and in the visited map.  
        grid_queue.push((map.to_grid_coords(rcoords), [map.to_grid_coords(rcoords)])) 
        while not grid_queue.is_empty(): 
            temp_coords, temp_path = grid_queue.pop() 
            visited.set_occ_grid_visited(temp_coords)
            #Goal found return the shortest path. 
            if goal_func(temp_coords, map): 
                return temp_path  
            #Check if any neighbors need to be searched. 
            possible_neighbors = ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1))
            for i, j in possible_neighbors: 
                # Use x + i and y + j for neighbor coords (previously y used x by mistake)
                nx = temp_coords.get_xcoord() + i
                ny = temp_coords.get_ycoord() + j
                neighbor = Coords(nx, ny, Coords.GRID)
                if self._valid_node(neighbor, map, visited): 
                    grid_queue.push((neighbor, temp_path + [neighbor]))
                    visited.set_occ_grid_visited(neighbor)
        return None 

map_dimension = 128
map_resolution = 0.5
max_path_dist = 1.5
goal_dist = 0.6
furthest_time = 200
class Tracker(Node):
    def __init__(self):
        super().__init__('Track')
        self.startTime = time.time()
        self.time = 0.1
        # recovery behavior: when stuck, back up for a number of control steps
        self.recovery_counter = 0
        self.recovery_steps = 8  # number of steps to back up when stuck
        self.recovery_back_speed = 0.4  # m/s reverse speed during recovery
        self.recovery_turn = 0.8  # rad/s turning during recovery
        #self.timer = self.create_timer(self.time, self.timer_callback)
        self.GoalFinder = GoalFinder(max_path_dist, goal_dist)
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
        self.startPos = [0, 0]
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
        if self.robot_set == False: 
            self.startPos = [self.robot_x, self.robot_y] 
            self.map = Map(map_dimension, map_resolution, 2.8, Coords(self.robot_x, self.robot_y, Coords.WORLD))
            self.robot_set = True 
        
        dx = self.robot_x - self.startPos[0] 
        dy = self.robot_y - self.startPos[1] 

        distanceTraveled = math.sqrt(dx*dx + dy*dy) 
        curTime = time.time() 
        elapsed = curTime - self.startTime 
        print("Elapsed Time: " + str(elapsed)) 
        print("Distance: " + str(distanceTraveled))

    def sensor_callback(self, msg): 
        if self.robot_set: 
            grid = self.map.get_occ_grid()
            self.map.update_map(Coords(self.robot_x, self.robot_y, Coords.WORLD), self.robot_theta, msg.ranges)
            #robot_grid_coords = self.map.to_grid_coords(Coords(self.robot_x, self.robot_y, Coords.WORLD))
            # Only find a new goal if we don't have a current goal or if we've reached the current goal

            if self.GoalFinder.get_cur_goal() == None: 
                if time.time() - self.startTime > furthest_time:  
                    print("Going To Furthest Away Goal")
                    goal = self.GoalFinder.find_goal(Coords(self.robot_x, self.robot_y, Coords.WORLD), self.map, True)
                else: 
                    goal = self.GoalFinder.find_goal(Coords(self.robot_x, self.robot_y, Coords.WORLD), self.map, False) 
                if goal is None: 
                    print("No New Goal Found - All Areas May Be Explored") 
                else: 
                    self.goal_start_time = time.time()  
            elif self.GoalFinder.near_goal(Coords(self.robot_x, self.robot_y, Coords.WORLD), self.map): 
                self.GoalFinder.set_goal(self.map) 
                self.goal_start_time = None
                #print("Stopped Robot Movement - Goal Reached")
            elif self.goal_start_time is not None and (time.time() - self.goal_start_time) > 7: 
                self.GoalFinder.set_goal(self.map)
                self.goal_start_time = None 
                twist = Twist()
                self.publisher.publish(twist) 
                #print("Goal timeout! Abandoning goal after 15 seconds") 
            else: 
                x,y = self.map.to_world_coords(self.GoalFinder.get_cur_goal()).get_coords() 
                goal_error = math.sqrt((x - self.robot_x)**2 + (y - self.robot_y)**2)
                #print(f"Moving Toward Goal, Distance: {goal_error:.2f}")
                self.explore_control(x, y, msg.ranges)

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
        obstacle_threshold = 1.5  # Much higher threshold for early detection
        
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
        
        # If currently executing recovery (multi-step back up), do it here
        if getattr(self, 'recovery_counter', 0) > 0:
            twist.linear.x = -self.recovery_back_speed
            # alternate turning direction each step to wiggle out
            turn_dir = 1 if (self.recovery_counter % 2 == 0) else -1
            twist.angular.z = turn_dir * self.recovery_turn
            self.publisher.publish(twist)
            self.recovery_counter -= 1
            return

        # If stuck for too long, schedule backing up and turning (multi-step)
        if self.stuck_counter > 20:  # Stuck for 20 iterations
            self.recovery_counter = self.recovery_steps
            self.stuck_counter = 0
            # perform immediate first recovery step
            twist.linear.x = -self.recovery_back_speed
            twist.angular.z = self.recovery_turn
            self.publisher.publish(twist)
            return
        
        # Integrated obstacle avoidance with PID goal seeking
        if min_front < obstacle_threshold:
            #print(f"Front obstacle at {min_front:.2f}m - avoiding while seeking goal")
            # Obstacle in front - turn toward the side with MORE space
            if min_left > min_right:
                avoidance_turn = -0.3  # Turn LEFT toward more space
                #print(f"Turning left to avoid front obstacle (left: {min_left:.2f}m, right: {min_right:.2f}m)")
            else:
                avoidance_turn = 0.3  # Turn RIGHT toward more space
                #print(f"Turning right to avoid front obstacle (left: {min_left:.2f}m, right: {min_right:.2f}m)")
            # Combine avoidance with PID goal seeking (reduced PID influence)
            twist.angular.z = avoidance_turn + pid_output * 0.2
            # Reduce speed more when obstacle is closer
            if min_front < 0.8:
                twist.linear.x = 0.1  # Very slow when close
            else:
                twist.linear.x = 0.2  # Moderate forward movement
            
        elif min_left < obstacle_threshold: #* 0.8:  # Higher threshold for sides
            #print(f"Left obstacle at {min_left:.2f}m - turning right toward goal")
            # Obstacle on left - turn right but still seek goal
            avoidance_turn = 0.2  # Further reduced turn right
            # Combine avoidance with PID goal seeking
            twist.angular.z = avoidance_turn + pid_output * 0.4
            # Reduce speed when side obstacle is close
            if min_left < 1.0:
                twist.linear.x = 0.15  # Slower when close to side obstacle
            else:
                twist.linear.x = 0.25  # Faster forward movement
            
        elif min_right < obstacle_threshold: #* 0.8:
            #print(f"Right obstacle at {min_right:.2f}m - turning left toward goal")
            # Obstacle on right - turn left but still seek goal
            avoidance_turn = -0.2  # Further reduced turn left
            # Combine avoidance with PID goal seeking
            twist.angular.z = avoidance_turn + pid_output * 0.4
            # Reduce speed when side obstacle is close
            if min_right < 1.0:
                twist.linear.x = 0.13  # Slower when close to side obstacle
            else:
                twist.linear.x = 0.25  # Faster forward movement
            
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
            
            # PID control with speed adjustment
            twist.linear.x = min(0.5 * speed_factor, error_distance * 0.3 * speed_factor)
            twist.angular.z = pid_output  # Use full PID output
            #print(f"PID control - P: {p_term:.2f}, I: {i_term:.2f}, D: {d_term:.2f}, Speed: {speed_factor:.1f}")
        
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
