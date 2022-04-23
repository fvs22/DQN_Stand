import logging

import pandas as pd
import yaml
import tensorflow as tf

from pathlib import Path

from src.libs.dvc_utils import parser_args_for_sac, run_learning, set_connector
from src.sac.sac import SoftActorCritic
from src.libs.replay_buffer import ReplayBuffer
from sklearn.model_selection import ParameterGrid

tf.keras.backend.set_floatx('float64')
logging.basicConfig(level='INFO')


def save_parametric_sweep_mapping_table(grid_to_df: pd.DataFrame, output_dir: Path):
    grid_to_df.to_csv(output_dir / 'mapping_tabel.csv', index=False)


if __name__ == '__main__':
    args = parser_args_for_sac()

    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)

    additional_params = params_all['sac_params'].get('additional_params')
    general_params = params_all['sac_params'].get('general_params')
    neural_network_params = params_all['sac_params'].get('neural_network_params')
    experiment_params = params_all['sac_params'].get('experiment_params')
    param_sweep = params_all['param_sweep']

    episode_limit = param_sweep.get('episode_limit', 10000)

    tf.random.set_seed(additional_params.get('seed'))

    state_space = general_params['model_observation']
    action_space = general_params['model_action_space']

    params_grid = ParameterGrid(param_sweep.get('sweep'))

    full_path = Path(args.model_path)
    history_path = Path(args.output_history_dir)

    full_path.mkdir(exist_ok=True, parents=True)
    history_path.mkdir(exist_ok=True, parents=True)
    num = 0

    save_parametric_sweep_mapping_table(grid_to_df=pd.DataFrame(list(params_grid)), output_dir=full_path)

    connector, do_episode = set_connector(general_params=general_params,
                                          learning_mode=experiment_params.get('learning_mode'))

    for num, param_sample in enumerate(params_grid):
        # set sweep path
        sweep_path = full_path / (full_path.name + f'_{num}')
        sweep_history_path = history_path / (history_path.name + f'_{num}')

        # Initialize Replay buffer.
        replay = ReplayBuffer(state_space, action_space)

        # Initialize policy and Q-function parameters.

        sac = SoftActorCritic(action_space, layer_size=param_sample['layer_size'],
                              learning_rate=param_sample['learning_rate'],
                              gamma=neural_network_params['gamma'], polyak=neural_network_params['polyak'])

        sweep_path.mkdir(exist_ok=True, parents=True)
        sweep_history_path.mkdir(exist_ok=True, parents=True)

        run_learning(output_path=sweep_path, history_path=sweep_history_path, rl_model=sac, buffer=replay,
                     additional_params=additional_params, general_params=general_params,
                     neural_network_params=neural_network_params, connector=connector,
                     episode_executing_function=do_episode, episode_limit=episode_limit,
                     training_params={'epochs': param_sample.get('epochs'),
                                      'batch_size': param_sample.get('batch_size')},
                     experiment_params=experiment_params)

        del sac
        connector.reset_simulation(simulation_transfer=int(
            2 * (experiment_params.get('simulation_time') / general_params.get('discretization_step'))))
    print('EXPERIMENTS ENDED')
