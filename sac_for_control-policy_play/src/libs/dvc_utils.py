import argparse
import random
from src.libs.rl import AbstractReinforcementLearningModel
from src.libs.replay_buffer import ReplayBuffer
from src.libs.connector import Connector, RealConnector, AbstractConnector
import pandas as pd
import time
import numpy as np
import math
import logging
from pathlib import Path

MOVING_AVERAGE_WINDOW = 100


def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='SAC')
    parser.add_argument('--model_path', '-mp', type=str, default='data/models/simulated/sweep_experiment/',
                        required=False,
                        help='path to save model')
    parser.add_argument('--output_history_dir', '-ohd', type=str, default='data/experiment_data/sweep_exp/',
                        required=False,
                        help='path to save logs of learning')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()


def reward_gauss_normed(y_true: float, y_target: float, scale: int = 1):
    e = y_target - y_true
    if e < 0:
        e_norm = e * (1.74 / (1.74 - y_target))
    else:
        e_norm = e * (1.74 / y_target)
    reward = 1.26 * math.exp(-5 * (e_norm ** 2)) - 0.63
    reward *= scale
    return reward


def reward_gauss(y_true: float, y_target: float, scale: int = 1):
    reward = 1.26 * math.exp(-5 * ((y_target - y_true) ** 2)) - 0.63
    reward *= scale
    return reward


def random_float(low, high):
    return random.random() * (high - low) + low


def save_step_response(filename: str, time_ms: list, action: list, current: list, position: list,
                       angular_velocity: list, object_velocity: list):
    df = pd.DataFrame([])
    actions_array = np.array(action).squeeze()
    df = df.assign(time=time_ms)
    df = df.assign(action=actions_array)
    df = df.assign(current=current)
    df = df.assign(position=position)
    df = df.assign(angular_velocity=angular_velocity)
    df = df.assign(object_velocity=object_velocity)
    df.to_csv(f'data/{filename}', index=False)


def save_and_add_history(out_history: Path, row: dict):
    if not out_history.exists():
        row = {key: [item] for key, item in row.items()}
        df = pd.DataFrame(row,
                          columns=['episode', 'metric', 'episode_reward', 'moving_average', 'y_target'])
    else:
        df = pd.read_csv(out_history)
        df = df[['episode', 'metric', 'episode_reward', 'moving_average', 'y_target']]
        df = df.append(row, ignore_index=True)
    df.to_csv(out_history)


def passing_between_episodes_on_stand(done: bool, global_step: int, connector: RealConnector, flag: int,
                                      y_target: float):
    if global_step != 1:
        while done:
            connector.step(0, flag, y_target)
            next_state, metric, done = connector.receive(y_target)
        flag = 0
    return flag


def run_single_episode_on_model(done: bool, global_step: int, flag: int, y_target: float,
                                rl_model: AbstractReinforcementLearningModel,
                                buffer: ReplayBuffer, connector: Connector, additional_params: dict,
                                general_params: dict, response_dict: dict, verbose: bool = False):
    current_state = [0 for _ in range(int(general_params['model_observation']))]
    metric = 0
    step = 1
    episode_reward = 0
    t0 = time.perf_counter()
    while not done:
        if global_step < additional_params['start_steps']:
            if np.random.uniform() > 0.8:
                action = random_float(0, 10)
            else:
                action = rl_model.sample_action(current_state)
        else:
            action = rl_model.sample_action(current_state)

        # Execute action, observe next state and reward
        t = time.perf_counter() - t0
        connector.step(action=action)
        next_state, metric, _, done = connector.receive()

        y_true = next_state[1]
        reward = reward_gauss(y_true=y_true, y_target=y_target)

        episode_reward += reward
        _ = action.numpy()

        response_dict['current'].append(next_state[0])
        response_dict['position'].append(y_true)
        response_dict['angular_velocity'].append(next_state[2])
        response_dict['object_velocity'].append(next_state[3])
        response_dict['time_c'].append(t)
        response_dict['action_list'].append(_[0])

        # Set end to 0 if the episode ends otherwise make it 1
        # although the meaning is opposite but it is just easier to multiply
        # with reward for the last step.
        if done:
            end = 0
            flag = 1
        else:
            end = 1

        if verbose:
            logging.info(f'Global step: {global_step}')
            logging.info(f'current_state: {current_state}')
            logging.info(f'action: {action}')
            logging.info(f'reward: {reward}')
            logging.info(f'next_state: {next_state}')
            logging.info(f'end: {end}')

        # Store transition in replay buffer
        buffer.store(current_state, action, reward, next_state, end)

        # Update current state
        current_state = next_state

        step += 1
        global_step += 1
    return response_dict, episode_reward, metric, global_step, flag


def run_single_episode_on_stand(done: bool, global_step: int, flag: int, y_target: float,
                                rl_model: AbstractReinforcementLearningModel, buffer: ReplayBuffer,
                                connector: RealConnector, additional_params: dict, general_params: dict,
                                response_dict: dict, verbose: bool = False):
    current_state = [0 for _ in general_params['model_observation']]
    metric = 0
    step = 1
    episode_reward = 0
    flag = passing_between_episodes_on_stand(done=done, global_step=global_step, connector=connector,
                                             flag=flag, y_target=y_target)
    t0 = time.perf_counter()
    while not done:
        if global_step < additional_params['start_steps']:
            if np.random.uniform() > 0.8:
                action = random_float(0, 10)
            else:
                action = rl_model.sample_action(current_state)
        else:
            action = rl_model.sample_action(current_state)

        # Execute action, observe next state and reward
        t = time.perf_counter() - t0
        connector.step(action=action, flag=flag, y_target=y_target)
        next_state, metric, done = connector.receive(y_target=y_target)

        y_true = next_state[1]
        reward = reward_gauss(y_true=y_true, y_target=y_target)

        episode_reward += reward
        _ = action.numpy()

        response_dict['current'].append(next_state[0])
        response_dict['position'].append(y_true)
        response_dict['angular_velocity'].append(next_state[2])
        response_dict['object_velocity'].append(next_state[3])
        response_dict['time_c'].append(t)
        response_dict['action_list'].append(_[0])

        # Set end to 0 if the episode ends otherwise make it 1
        # although the meaning is opposite but it is just easier to mutiply
        # with reward for the last step.
        if done:
            end = 0
            flag = 1
        else:
            end = 1

        if verbose:
            logging.info(f'Global step: {global_step}')
            logging.info(f'current_state: {current_state}')
            logging.info(f'action: {action}')
            logging.info(f'reward: {reward}')
            logging.info(f'next_state: {next_state}')
            logging.info(f'end: {end}')

        # Store transition in replay buffer
        buffer.store(current_state, action, reward, next_state, end)

        # Update current state
        current_state = next_state

        step += 1
        global_step += 1
    return response_dict, episode_reward, metric, global_step, flag


def compute_statistics(rl_model: AbstractReinforcementLearningModel, history_dict: dict, episode: int,
                       full_path: Path, learning: bool = True, episode_limit: int = np.inf):
    episode_rewards = history_dict['episode_reward']

    avg_episode_reward = sum(episode_rewards[-1000:]) / len(episode_rewards[-1000:])

    print(f"Episode {episode} reward: {episode_rewards[-1] / 1}")
    print(f"{episode} Average episode reward: {avg_episode_reward / 1}")
    if episode > episode_limit:
        learning = False
        rl_model.save_model(full_path.parent, full_path.name)
        print('***** Episode limit is reached! ***** \n'
              '#####################################\n'
              'Learning is ended. Best model is saved. \n'
              f'Model name is {full_path.name}')

    if len(episode_rewards) > MOVING_AVERAGE_WINDOW:
        moving_average = sum(episode_rewards[-MOVING_AVERAGE_WINDOW:]) / MOVING_AVERAGE_WINDOW
        print(f"Episode {episode} moving average reward: {moving_average}")

        if moving_average > 60:
            learning = False
            rl_model.save_model(full_path.parent, full_path.name)
            print('Learning is ended. Best model is saved. \n'
                  f'Model name is {full_path.name}')
        elif moving_average > 30 * 2:
            rl_model.update_learning_rate(0.0002)
        elif moving_average > 0 * 2:
            rl_model.update_learning_rate(0.0005)
        elif moving_average > -30 * 2:
            rl_model.update_learning_rate(0.001)
        elif moving_average > -60 * 2:
            rl_model.update_learning_rate(0.002)
        elif moving_average > -90 * 2:
            rl_model.update_learning_rate(0.003)
        elif moving_average > -100 * 2:
            rl_model.update_learning_rate(0.004)
    print(f'Final value of metric {history_dict["metric"][-1]}')
    print(f'Target {history_dict["y_target"][-1]}')
    return learning


def set_connector(general_params: dict, learning_mode: str) -> (AbstractConnector, callable):
    math_model_full_address = (general_params.get('math_model_address'), general_params.get('math_model_port'))
    if learning_mode == 'stand':
        connector = RealConnector(math_model_full_address)
        episode_executing_function = run_single_episode_on_stand
    elif learning_mode == 'model':
        connector = Connector(math_model_full_address)
        episode_executing_function = run_single_episode_on_model
    else:
        raise KeyError('Please check learning mode')
    return connector, episode_executing_function


def run_learning(output_path: Path, history_path: Path, rl_model: AbstractReinforcementLearningModel,
                 buffer: ReplayBuffer, additional_params: dict, general_params: dict, neural_network_params: dict,
                 connector: AbstractConnector, episode_executing_function: callable, training_params=None,
                 episode_limit: int = np.inf, experiment_params=None):
    # Repeat until convergence
    if training_params is None:
        training_params = neural_network_params
    response_dict = {
        'time_c': [],
        'action_list': [],
        'position': [],
        'current': [],
        'angular_velocity': [],
        'object_velocity': []
    }
    history_dict = {
        'episode_reward': [],
        'metric': [],
        'y_target': []
    }
    global_step = 1
    episode = 0
    done = False
    flag = 1

    learning = True
    moving_average = 0

    while learning:
        if general_params['y_target_mode'] == 'fixed':
            y_target = general_params['y_target']
        else:
            y_target = random_float(low=0.2, high=1.4)
        response_dict, episode_reward, metric, global_step, flag = episode_executing_function(done=done,
                                                                                              global_step=global_step,
                                                                                              flag=flag,
                                                                                              y_target=y_target,
                                                                                              rl_model=rl_model,
                                                                                              additional_params=additional_params,
                                                                                              general_params=general_params,
                                                                                              response_dict=response_dict,
                                                                                              buffer=buffer,
                                                                                              connector=connector)
        history_dict['episode_reward'].append(episode_reward)
        history_dict['metric'].append(metric)
        history_dict['y_target'].append(y_target)
        learning = compute_statistics(rl_model=rl_model, history_dict=history_dict, episode=episode, learning=learning,
                                      full_path=output_path, episode_limit=episode_limit)
        rl_model.complex_training(buffer=buffer, training_params=training_params)
        if episode % 15 == 0:
            rl_model.save_model(output_path.parent, output_path.name)
        row = {'episode': episode, 'metric': metric, 'episode_reward': episode_reward,
               'moving_average': moving_average, 'y_target': y_target}
        save_and_add_history(history_path / f'{history_path.name}_dynamic_his.csv', row)
        episode += 1
        if experiment_params is not None:
            connector.reset_simulation(simulation_transfer=int(
                2 * (experiment_params.get('simulation_time') / general_params.get('discretization_step'))))
