from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
from utilities import Utilities

"""
Custom Environment describing a maze-like Labyrinth Forest Environment through which 
agents can be passed through to detect deforested squares as they move from a starting point
to an end point
"""
class LabyrinthForest(Env):

    """
    Initialize all necessary primary parameters 
    """
    def __init__(self, env_config={}, duration=10000, shape=[16,12]):
        # Input Parameters (Fixed)
        self.shape = shape
        self.duration = duration
        # Constant Parameters (Fixed)
        self.action_space = Discrete(4) # [Actions] 0: UP | 1: RIGHT | 2: DOWN | 3: LEFT
        self.observation_space = Box(low=np.array([0,0]), high=np.array([15,11]), dtype=np.int32)   # Defines observation space
        self.actions = ['UP','RIGHT','DOWN','LEFT']
        self.obs_states = np.arange(5)  # Define observation state indices
        self.obs_states_symbols = np.array(['S','E','D','N','F'])   # Define observation state symbols
        self.repeat_visits_reward = 0.5 # Reward for already visited reward states
        self.initialize_env()
        # Generated Parameters (Dynamic): Must be reset
        self.state = np.copy(self.start_state)    # initial state is start state
        self.steps = int(self.duration)  # initial number of steps equals given duration
        self.visited_deforested_locations = np.array([])    # Define visited deforested locations initially
    
    """
    Initialize the fixed state matrices corresponding to the forest environment
    """
    def initialize_env(self):
        self.start_state = np.array([0,0])
        self.end_state = np.array([11,11])
        self.deforested_locations = Utilities.order_array(
            self.shape, 
            np.array([[0,11], [1,3], [5,7], [7,3], [9,9], [11,4], [15,0], [15,8]])
        )
        self.non_forest_area_locations = Utilities.order_array(
            self.shape, 
            Utilities.merge_arrays([
                Utilities.fill_array([2,1],[2,5]),
                Utilities.fill_array([4,4],[8,4]),
                Utilities.fill_array([6,2],[12,2]),
                Utilities.fill_array([2,8],[5,8]),
                Utilities.fill_array([8,8],[10,8]),
                Utilities.fill_array([10,10],[12,10]),
                Utilities.fill_array([14,6],[14,10]),
                Utilities.fill_array([14,0],[14,2]),
                Utilities.fill_array([10,5],[12,5]),
                Utilities.fill_array([5,6],[7,6]),
                np.array([
                    [0,1],[1,1],[3,3],[4,3],[1,4],[8,3],[12,3],[12,4],[10,6],[15,6],[2,7],
                    [13,8],[5,9],[6,9],[8,9],[10,9],[0,10],[1,10],[5,10],[8,10],[12,11]
                ])
            ])
        )
        self.forest_area_locations = Utilities.order_array(
                self.shape, 
                Utilities.merge_arrays([
                    np.array([self.start_state]),
                    np.array([self.end_state]),
                    self.deforested_locations,
                    self.non_forest_area_locations
                ]),
                inverse=True
            )
        
        self.initialize_observation_data()
    
    """
    Initialize observation state and reward matrices
    """
    def initialize_observation_data(self):
        # Initialize all entries with 0 for the observation state matrix
        obs_state_matrix = [[0 for y in np.arange(self.shape[1])] for x in np.arange(self.shape[0])]
        for x in np.arange(self.shape[0]):
            for y in np.arange(self.shape[1]):
                # Start State: 0
                if [x,y] in [self.start_state.tolist()]:
                    obs_state_matrix[x][y] = 0
                # End State: 1
                elif [x,y] in [self.end_state.tolist()]:
                    obs_state_matrix[x][y] = 1
                # Deforested Area State: 2
                elif [x,y] in self.deforested_locations.tolist():
                    obs_state_matrix[x][y] = 2
                # Non-Forest Area State: 3
                elif [x,y] in self.non_forest_area_locations.tolist():
                    obs_state_matrix[x][y] = 3
                # Forest Area State: 4
                else:
                    obs_state_matrix[x][y] = 4
        # Initialize observation data
        # Define observation state matrix
        self.obs_state_matrix = np.array(obs_state_matrix)
        # Define reward matrix
        self.rewards = Utilities.get_reward_matrix(np.copy(self.obs_state_matrix))

    """
    Function defines the changes at each step taken on the environment
    """
    def step(self, action):
        reward = 0.0
        done = False
        info = {}
        if self.is_action_valid(action):
            # Calculate next state
            self.state = self.calculate_next_state(action)
            
            # Add next state to list of visited deforested locations, if next state is of deforested type
            is_reward_state_visited = Utilities.item_in_array(self.visited_deforested_locations.tolist(), self.state.tolist())
            if ((self.state_code() == 2) and not is_reward_state_visited):
                curr_visited_locations = self.visited_deforested_locations.tolist()
                curr_state_as_list = self.state.tolist()
                curr_visited_locations.append(curr_state_as_list)
                self.visited_deforested_locations = np.array(curr_visited_locations)
            
            # Calculate reward
            reward = self.rewards[self.state[0]][self.state[1]]
            if is_reward_state_visited: # Lesser or no rewards upon repeated visits on reward states
                reward = self.repeat_visits_reward
            # Decrement number of steps, i.e., count one step
            self.steps = int(self.steps)-1
            # Set done condition to be true if end state [1] reached (or if number of steps equals 0)
            if (self.state_code() == 1) or (int(self.steps) == 0):
                done = True
        # Return data after taking a step
        return self.state, reward, done, info

    """
    Function for rendering environment on display (optional)
    """
    def render(self):
        pass

    """
    Function defines the changes each time the environment is reset
    """
    def reset(self):
        self.state = np.copy(self.start_state)    # reset initial state to start state
        self.steps = int(self.duration)  # reset number of steps
        self.visited_deforested_locations = np.array([]) # reset visited deforested locations
        return self.state   # return initial state
    
    """
    Returns code for current state.
    Possible values: {0: Start State, 1: End State, 2: Deforested Area, 3: Non-Forest Area, 4: Forest Area State}
    """
    def state_code(self):
        return self.obs_state_matrix[self.state[0]][self.state[1]]

    """
    Checks and returns if next action is valid
    """
    def is_action_valid(self, next_action):
        x_size = self.shape[0]
        y_size = self.shape[1]

        next_state = self.calculate_next_state(next_action)
        
        # Reject next state if its dimensions exceeds bounds
        if ((next_state[0] < 0) or (next_state[0] >= x_size) or (next_state[1] < 0) 
            or (next_state[1] >= y_size)):
            return False
        # Reject next state if it is in a Non-Forest Area
        if (self.obs_state_matrix[next_state[0]][next_state[1]] == 3):
            return False
        return True
    
    """
    Returns the coordinates of the following state corresponding to the given action
    """
    def calculate_next_state(self, next_action):
        curr_state = np.copy(self.state)
        next_state = curr_state
        if (next_action == 0):      # 0: UP
            next_state[1] += 1
        elif (next_action == 1):    # 1: RIGHT
            next_state[0] += 1
        elif (next_action == 2):    # 2: DOWN
            next_state[1] -= 1
        else:                       # 3: LEFT
            next_state[0] -= 1
        return next_state

    """
    Gets action corresponding to its mapped index
    """
    @staticmethod
    def get_action_from_id(action_id):
        actions = ['UP','RIGHT','DOWN','LEFT']
        if (action_id is None) or ((int(action_id)<0) or (int(action_id)>=len(actions))):
            return None
        else:
            return actions[int(action_id)]
    
    """
    Displays following Graphs with respect to all Episodes:-
    - Rewards
    - No. of deforested areas Found
    - No. of steps (until completion)
    """
    @staticmethod
    def display_graphs(all_episodes, all_rewards, all_num_deforested_areas_found, all_steps, algorithm_name='Standard', show_average=False):
        Utilities.display_graph(all_episodes, all_rewards, 'Episodes', 'Rewards', f'Reward Curve ({algorithm_name} Algorithm)', show_average)
        Utilities.display_graph(all_episodes, all_num_deforested_areas_found, 'Episodes', 'No. of deforested areas found', f'Deforestation Detection Curve ({algorithm_name} Algorithm)', show_average)
        Utilities.display_graph(all_episodes, all_steps, 'Episodes', 'No. of Steps', f'Steps Curve ({algorithm_name} Algorithm)', show_average)
        pass

    """
    Print all matrices corresponding to the fixed forest environment map
    """
    def print_env_log(self):
        print(f'=======================')
        print(f'Start State:-\n{self.start_state}')
        print(f'=======================')
        print(f'End State:-\n{self.end_state}')
        print(f'=======================')
        print(f'Deforested Locations:-\n{self.deforested_locations}')
        print(f'=======================')
        print(f'Non-Forest Area Locations:-\n{self.non_forest_area_locations}')
        print(f'=======================')
        print(f'Forest Area Locations:-\n{self.forest_area_locations}')
        print(f'=======================')
        print(f'Reward Matrix:-\n\n{Utilities.get_symbol_matrix_string(self.rewards)}')
        print(f'=======================')
        pass
    
    """
    Prints a 2D map-like matrice, which describes the states (here, symbols) 
    in each square (location) of the labyrinth forest environment
    """
    def print_env(self):
        print(f'=======================')
        symbol_matrix_str = Utilities.get_symbol_matrix_string(self.obs_state_matrix, self.obs_states_symbols)
        print(f'Labyrinth Forest Environment:-\n\n{symbol_matrix_str}')
        print(f'=======================')
        pass

    """
    Prints fixed log data corresponding to the environment
    """
    def print_log_info(self):
        self.print_env()
        self.print_env_log()
