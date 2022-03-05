import ray
import ray.rllib.agents.ppo as ppo
from labyrinthForest import LabyrinthForest
import os

"""
Defining Constants at the start
"""
OBJECT_STORE_MEMORY = 2*(10**8) # Limiting object store memory to 200MB
AGENT_TRAINING_STEPS = 2    # Number of steps to train the agent

"""
Run the Standard Random Search Algorithm on the custom "Labyrinth Forest" Environment
"""
def run_labyrinth_forest_standard(episodes=10, steps=10000):
    print(f'==============')
    print(f'Algorithm: Standard (Random Search)')
    print(f'==============')
    print(f'Total no. of episodes: {episodes} | Total no. of steps: {steps}')
    print(f'==============')

    # Initiate Labyrinth Forest environment
    env = LabyrinthForest(duration=steps)

    # Initialize overall values
    total_num_deforested_areas = len(env.deforested_locations.tolist())
    all_episodes = [(i+1) for i in range(episodes)]
    all_rewards = list()
    all_num_deforested_areas_found = list()
    all_steps = list()

    # Iterate through all episodes and perform rollout
    for episode in range(episodes):
        """ print(f'Episode {episode+1}...\n')  # [DEBUG] """
        next_state = env.reset()
        total_reward = 0.0
        for step in range(steps):
            #env.render()

            # Sample and Update action until valid action sample found
            action = env.action_space.sample()
            while not env.is_action_valid(action):
                action = env.action_space.sample()
            
            # Perform a step in the environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            """ [DEBUG]
            print(step, next_state, reward, done, info, action)
            print(f'Step: {step} | Next State: {next_state} | Reward: {reward} | Done: {done} | Info: {info} | Action: {action} ({LabyrinthForest.get_action_from_id(action)})')
            """
            # Break if done condition met
            if done:
                """ print(f'Total reward: {total_reward}') # [DEBUG] """
                break
        
        # Update overall values
        num_deforested_areas_found = len(env.visited_deforested_locations.tolist())
        all_rewards.append(total_reward)
        all_num_deforested_areas_found.append(num_deforested_areas_found)
        all_steps.append(step+1)

        """ [DEBUG]
        print(f'Total reward: {total_reward} | Total no. of steps: {step+1}/{steps}')
        print(f'No. of deforested areas found: {num_deforested_areas_found} | Deforested areas: {env.visited_deforested_locations}')
        print(f'==============')
        """
    
    # Printing overall values
    print(f'==============')
    print(f'Average reward: {round((sum(all_rewards)/episodes),2)}')
    print(f'Average no. of deforested areas found: {round((sum(all_num_deforested_areas_found)/episodes),2)} (out of {total_num_deforested_areas})')
    print(f'Average total no. of steps: {round((sum(all_steps)/episodes),2)} (out of {steps})')
    print(f'==============')
    """ print(f'==============')   # [DEBUG] """

    # Displaying graphs
    LabyrinthForest.display_graphs(all_episodes, all_rewards, all_num_deforested_areas_found, all_steps, 'Standard', True)


"""
Run PPO (Proximal Policy Optimization) Algorithm on the custom "Labyrinth Forest" Environment
"""
def run_labyrinth_forest_ppo(episodes=10, steps=10000):
    print(f'==============')
    print(f'Algorithm: PPO (Proximal Policy Optimization)')
    print(f'==============')
    print(f'Total no. of episodes: {episodes} | Total no. of steps: {steps}')
    print(f'==============')
    
    # Initiate Labyrinth Forest environment
    env = LabyrinthForest(duration=steps)

    # Start up Ray. This must be done before we instantiate any RL agents.
    os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"  # Necessary to solve memory insufficiency problem
    ray.init(object_store_memory=OBJECT_STORE_MEMORY)

    # Define PPO agent - set environment to algorithm
    agent = ppo.PPOTrainer(env=LabyrinthForest, config={
        "env_config": {}
        })
    
    # Train agent on the env for given number of steps
    for i in range(AGENT_TRAINING_STEPS):
        result = agent.train()

    # Initialize overall values
    total_num_deforested_areas = len(env.deforested_locations.tolist())
    all_episodes = [(i+1) for i in range(episodes)]
    all_rewards = list()
    all_num_deforested_areas_found = list()
    all_steps = list()

    # Iterate through all episodes and perform rollout
    for episode in range(episodes):
        """ print(f'Episode {episode+1}...\n')  # [DEBUG] """
        next_state = env.reset()
        total_reward = 0.0
        for step in range(steps):
            #env.render()

            # Compute and Update action until valid action sample found
            action = agent.compute_single_action(next_state)
            while not env.is_action_valid(action):
                action = agent.compute_single_action(next_state, explore=True)
            
            # Perform a step in the environment
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            """ [DEBUG]
            print(step, next_state, reward, done, info, action)
            print(f'Step: {step} | Next State: {next_state} | Reward: {reward} | Done: {done} | Info: {info} | Action: {action} ({LabyrinthForest.get_action_from_id(action)})')
            """
            # Break if done condition met
            if done:
                """ print(f'Total reward: {total_reward}')  # [DEBUG] """
                break
        
        # Update overall values
        num_deforested_areas_found = len(env.visited_deforested_locations.tolist())
        all_rewards.append(total_reward)
        all_num_deforested_areas_found.append(num_deforested_areas_found)
        all_steps.append(step+1)

        """ [DEBUG]
        print(f'Total reward: {total_reward} | Total no. of steps: {step+1}/{steps}')
        print(f'No. of deforested areas found: {num_deforested_areas_found} | Deforested areas: {env.visited_deforested_locations}')
        print(f'==============')
        """

    # Display overall values
    print(f'==============')
    print(f'Average reward: {round((sum(all_rewards)/episodes),2)}')
    print(f'Average no. of deforested areas found: {round((sum(all_num_deforested_areas_found)/episodes),2)} (out of {total_num_deforested_areas})')
    print(f'Average total no. of steps: {round((sum(all_steps)/episodes),2)} (out of {steps})')
    print(f'==============')
    """ print(f'==============')   # [DEBUG] """

    # Displaying graphs
    LabyrinthForest.display_graphs(all_episodes, all_rewards, all_num_deforested_areas_found, all_steps, 'PPO', True)

"""
Displays fixed log info (rough environment description)
"""
def display_log():
    # Initiate Labyrinth Forest environment
    env = LabyrinthForest()
    # Print Log Info (Fixed)
    env.print_log_info()

"""
Test Function to sample action and observation data
"""
def sample_data():
    # Initiate Labyrinth Forest environment
    env = LabyrinthForest()
    action = env.action_space.sample()
    obs = env.observation_space.sample()
    print(f'Action Sample: {action}')
    print(f'Observation Sample: {obs}')


"""
Main Function where RL-Algorithm based agents can be run on the environment
"""
if __name__ == "__main__":
    # Standard Algorithm: Random Search (Working output)
    run_labyrinth_forest_standard(10, 10000)    # Params: 10 episodes, 10000 steps

    # PPO Algorithm: Proximal Policy Optimization (Working output)
    run_labyrinth_forest_ppo(10, 10000)    # Params: 10 episodes, 10000 steps

    """ [DEBUG] Test Functions
    # Display Environment Log Info
    #display_log()
    # Collect action and observation samples
    #sample_data()
     """
    pass
