import gym
import random
from starlette.requests import Request
import requests
import ray
import ray.rllib.agents.ppo as ppo
from ray import serve
from ray import tune
from ray.tune.logger import pretty_print
from labyrinthForest import LabyrinthForest
import numpy as np

from utilities import Utilities

OBJECT_STORE_MEMORY = 2*(10**8) # Limiting object store memory to 200MB
AGENT_TRAINING_STEPS = 2    # Number of steps to train the agent


"""
Run the Standard Random Search Algorithm on the custom "Labyrinth Forest" Environment
"""
def run_labyrinth_forest_standard(episodes=1000, steps=10000):
    print(f'==============')
    print(f'Algorithm: Standard (Random Search)')
    print(f'==============')
    print(f'Total no. of episodes: {episodes} | Total no. of steps: {steps}')
    print(f'==============')

    # Initiate Labyrinth Forest environment
    env = LabyrinthForest(duration=steps)

    total_num_deforested_areas = len(env.deforested_locations.tolist())

    all_episodes = [(i+1) for i in range(episodes)]
    all_rewards = list()
    all_num_deforested_areas_found = list()
    all_steps = list()

    for episode in range(episodes):
        #print(f'Episode {episode+1}...\n')  # DEBUG_LOG
        next_state = env.reset()
        total_reward = 0.0
        for step in range(steps):
            #env.render()
            action = env.action_space.sample() # Debug_old
            # Update action until valid action sample found
            while not env.is_action_valid(action):
                action = env.action_space.sample()
            #action = test_agent.compute_action(next_state)   # Debug_new
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            #print(step, next_state, reward, done, info, action) # DEBUG_LOG
            #print(f'Step: {step} | Next State: {next_state} | Reward: {reward} | Done: {done} | Info: {info} | Action: {action} ({LabyrinthForest.get_action_from_id(action)})') # DEBUG_LOG
            if done:
                #print(f'Total reward: {total_reward}') # DEBUG_LOG
                break
        
        num_deforested_areas_found = len(env.visited_deforested_locations.tolist())
        all_rewards.append(total_reward)
        all_num_deforested_areas_found.append(num_deforested_areas_found)
        all_steps.append(step+1)

        #print(f'Total reward: {total_reward} | Total no. of steps: {step+1}/{steps}')  # Debug_new
        #print(f'No. of deforested areas found: {num_deforested_areas_found} | Deforested areas: {env.visited_deforested_locations}')  # Debug_new
        #print(f'==============')   # DEBUG_LOG
    
    # Printing overall values
    print(f'==============')
    print(f'Average reward: {round((sum(all_rewards)/episodes),2)}')
    print(f'Average no. of deforested areas found: {round((sum(all_num_deforested_areas_found)/episodes),2)} (out of {total_num_deforested_areas})')
    print(f'Average total no. of steps: {round((sum(all_steps)/episodes),2)} (out of {steps})')
    print(f'==============')
    #print(f'==============')   # DEBUG_LOG

    # Displaying graphs
    LabyrinthForest.display_graphs(all_episodes, all_rewards, all_num_deforested_areas_found, all_steps, 'Standard')


"""
Run PPO (Proximal Policy Optimization) Algorithm on the custom "Labyrinth Forest" Environment
"""
def run_labyrinth_forest_ppo(episodes=1000, steps=10000):
    print(f'==============')
    print(f'Algorithm: PPO (Proximal Policy Optimization)')
    print(f'==============')
    print(f'Total no. of episodes: {episodes} | Total no. of steps: {steps}')
    print(f'==============')
    
    # Initiate Labyrinth Forest environment
    env = LabyrinthForest(duration=steps)
    #env = gym.make(env_name)

    # Start up Ray. This must be done before we instantiate any RL agents.
    ray.init(object_store_memory=OBJECT_STORE_MEMORY)   # Debug_testing

    # PPO Configuration
    config = ppo.DEFAULT_CONFIG.copy()
    config['num_workers'] = 1
    config['num_sgd_iter'] = 30
    config['sgd_minibatch_size'] = 128
    config['model']['fcnet_hiddens'] = [100, 100]
    config['num_cpus_per_worker'] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed

    # Define PPO agent - set environment to algorithm
    agent = ppo.PPOTrainer(config, env=LabyrinthForest) # DEBUG_TEST
    # Train agent on the env for given number of steps
    for i in range(AGENT_TRAINING_STEPS):
        result = agent.train()
    # Save trained agent
    checkpoint_path = agent.save()
    print(f'Checkpoint Path:- {checkpoint_path}')

    # Create Test agent and restore saved agent info
    trained_config = config.copy()
    test_agent = ppo.PPOTrainer(trained_config, env=LabyrinthForest)    # DEBUG_TEST
    test_agent.restore(checkpoint_path)

    
    total_num_deforested_areas = len(env.deforested_locations.tolist())

    all_episodes = [(i+1) for i in range(episodes)]
    all_rewards = list()
    all_num_deforested_areas_found = list()
    all_steps = list()

    ppo_run_successful = True

    for episode in range(episodes):
        #print(f'Episode {episode+1}...\n')  # DEBUG_LOG
        next_state = env.reset()
        total_reward = 0.0
        ppo_episode_run_successful = True
        for step in range(steps):
            #env.render()
            action = test_agent.compute_action(next_state)   # Debug_new
            # Update action until valid action sample found
            ppo_compute_action_count = 0
            while not env.is_action_valid(action):
                if ppo_compute_action_count == 2:
                    ppo_episode_run_successful = False  # DEBUG_LOG
                    break
                action = test_agent.compute_action(next_state)   # Debug_new
                ppo_compute_action_count += 1
            if ppo_compute_action_count == 2:
                #print(f'Error (PPO): Same invalid action computed upon multiple runs')  # DEBUG_LOG
                break
            #action = test_agent.compute_action(next_state)   # Debug_new
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            #print(step, next_state, reward, done, info, action) # DEBUG_LOG
            #print(f'Step: {step} | Next State: {next_state} | Reward: {reward} | Done: {done} | Info: {info} | Action: {action} ({LabyrinthForest.get_action_from_id(action)})') # DEBUG_LOG
            if done:
                #print(f'Total reward: {total_reward}') # DEBUG_LOG
                break
        
        if not ppo_episode_run_successful:
            #print(f'Error (PPO): Episode {episode} run not successful. Same invalid action computed upon multiple runs')  # DEBUG_LOG
            ppo_run_successful = False
            break
        
        num_deforested_areas_found = len(env.visited_deforested_locations.tolist())
        all_rewards.append(total_reward)
        all_num_deforested_areas_found.append(num_deforested_areas_found)
        all_steps.append(step+1)

        #print(f'Total reward: {total_reward} | Total no. of steps: {step+1}/{steps}')  # Debug_new
        #print(f'No. of deforested areas found: {num_deforested_areas_found} | Deforested areas: {env.visited_deforested_locations}')  # Debug_new
        #print(f'==============')   # DEBUG_LOG
    
    if not ppo_run_successful:
        print(f'Error (PPO): PPO run not successful. Same invalid action computed upon multiple runs for an episode')  # DEBUG_LOG
        return -1

    # Printing overall values
    print(f'==============')
    print(f'Average reward: {round((sum(all_rewards)/episodes),2)}')
    print(f'Average no. of deforested areas found: {round((sum(all_num_deforested_areas_found)/episodes),2)} (out of {total_num_deforested_areas})')
    print(f'Average total no. of steps: {round((sum(all_steps)/episodes),2)} (out of {steps})')
    print(f'==============')
    #print(f'==============')   # DEBUG_LOG

    # Displaying graphs
    LabyrinthForest.display_graphs(all_episodes, all_rewards, all_num_deforested_areas_found, all_steps, 'PPO')


"""
Displays fixed log info (rough environment description)
"""
def display_log():
    # Initiate Labyrinth Forest environment
    env = LabyrinthForest()
    # Print Log Info (Fixed)
    env.print_log_info()


if __name__ == "__main__":
    # Standard Algorithm: Random Search (Working output)
    run_labyrinth_forest_standard(1000, 10000)    # Params: 1000 episodes, 10000 steps

    # PPO Algorithm: Proximal Policy Optimization (leading to errors)
    #run_labyrinth_forest_ppo(1000, 10000)    # Params: 1000 episodes, 10000 steps
    #run_labyrinth_forest_ppo(10, 50)    # Params: 10 episodes, 50 steps

    # Display Environment Log Info
    #display_log()
    pass
