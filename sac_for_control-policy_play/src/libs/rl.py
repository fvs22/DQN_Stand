from pathlib import Path
from src.libs.replay_buffer import ReplayBuffer

class AbstractReinforcementLearningModel:

    def __init__(self, general_params: dict, neural_network_params: dict):
        pass

    def sample_action(self, current_state):
        pass

    def train(self, current_states, actions, rewards, next_states, ends):
        pass

    def update_learning_rate(self, lr: float):
        pass

    def update_alpha(self, current_states):
        pass

    def complex_training(self, buffer: ReplayBuffer, training_params: dict, verbose: bool = False):
        pass

    def save_model(self, model_folder: Path, model_name: str):
        pass

    def load_model(self, model_folder: Path, model_name):
        pass
