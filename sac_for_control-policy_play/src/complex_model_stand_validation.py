import time

from libs.connector import Connector
import pandas as pd
from pathlib import Path

MODEL_ADDRESS = ('10.24.1.206', 5000)

signal_count = 0

Y_TARGET = 0.71039850878084

FOLDER = Path('../validation_data/stand_signal/')
OUTPUT_FOLDER = Path('../validation_data/model_stand_signal6/')
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

connector_to_model = Connector(MODEL_ADDRESS)

for path in FOLDER.glob('*.csv'):
    df = pd.read_csv(path)
    actions = df['signal_value'].tolist()
    count = 0
    done = 0
    responses = []
    angular_velocities = []
    action_out = []
    times = []
    signal = []
    next_state = []
    if len(actions) < 500:
        while True:
            try:
                action = actions[count]
                if action > 10.0:
                    action = 9.99
                elif action < 0:
                    action = 0
                if count % 100 == 0:
                    print(action)
            except IndexError:
                break
            # var = input()
            t0 = time.perf_counter()
            connector_to_model.step(action)
            next_state, metric, y_target, done = connector_to_model.receive()
            t = time.perf_counter()

            responses.append(next_state[1])
            signal.append(action)
            angular_velocities.append(next_state[2])
            times.append(t - t0)
            count += 1
            action_out.append(action)
            # print(action)
            if count % 100 == 0:
                print(count)

        df = df.assign(signal_value=action_out)
        df = df.assign(model_response=responses)
        df = df.assign(model_angular_velocity=angular_velocities)

        df.to_csv(f'../validation_data/model_stand_signal6/model_signal_{signal_count}_actor.csv', index=False)
        signal_count += 1

        for item in range(202):
            connector_to_model.step(0.1)
            next_state, metric, y_target, done = connector_to_model.receive()

    responses.clear()
    times.clear()
    angular_velocities.clear()
    actions.clear()
    action_out.clear()
    if next_state:
        next_state.clear()

connector_to_model.close()
