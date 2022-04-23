import time

from libs.connector import RealConnector
import pandas as pd
from pathlib import Path

MODEL_ADDRESS = ('10.24.1.206', 5000)

signal_count = 0

Y_TARGET = 0.71039850878084

FOLDER = Path('../validation_data/signal')

connector_to_model = RealConnector(MODEL_ADDRESS)

for path in FOLDER.glob('*.csv'):
    df = pd.read_csv(path)
    actions = df['signal_value'].tolist()
    count = 0
    done = 0
    responses = []
    angular_velocities = []
    action_out = []
    times = []

    while True:
        try:
            action = actions[count]
            if count % 100 == 0:
                print(action)
        except IndexError:
            break
        # var = input()
        t0 = time.perf_counter()
        connector_to_model.step(action, 0, Y_TARGET)
        next_state, metric, done = connector_to_model.receive(Y_TARGET)
        t = time.perf_counter()

        responses.append(next_state[1])
        angular_velocities.append(next_state[2])
        times.append(t - t0)
        count += 1
        action_out.append(action)
        if count % 100 == 0:
            print(count)

    df = df.assign(time=times)
    df = df.assign(stand_response=responses)
    df = df.assign(stand_angular_velocity=angular_velocities)

    responses.clear()
    times.clear()
    angular_velocities.clear()
    actions.clear()
    action_out.clear()

    df.to_csv(f'../validation_data/stand_signal/stand_signal_{signal_count}.csv', index=False)
    signal_count += 1
    connector_to_model.step(0, 0, Y_TARGET)
    next_state, metric, done = connector_to_model.receive(Y_TARGET)
    time.sleep(60)
connector_to_model.close()
