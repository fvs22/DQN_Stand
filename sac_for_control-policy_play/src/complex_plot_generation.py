import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path

FOLDER = Path('../validation_data/model_stand_signal6/')
OUTPUT_FOLDER = Path('../validation_data/plots6/')
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)



plt.style.use('seaborn-whitegrid')


for file in FOLDER.glob('*_actor.csv'):
    df = pd.read_csv(file)
    actions = df['signal_value'].tolist()
    x_ax = [x / 10 for x in range(len(actions))]
    filename = file.name.replace('.csv', '')
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex='col')

    ax0.plot(x_ax, df['signal_value'].tolist())
    ax0.set(ylabel='Signal value, V', xlabel='Seconds')

    ax1.plot(x_ax, df['stand_response'].tolist(), label='Stand response')
    ax1.plot(x_ax, df['model_response'].tolist(), label='Model response')
    ax1.legend()

    ax1.set(ylabel='Response, m', xlabel='Seconds')

    ax2.plot(x_ax, df['stand_angular_velocity'].tolist(), label='Stand angular velocity')
    ax2.plot(x_ax, df['model_angular_velocity'].tolist(), label='Model angular velocity')

    ax2.set(ylabel='Angular velocity, V', xlabel='Seconds')
    ax2.legend()

    output_filename = OUTPUT_FOLDER / f'{filename}_actor_plot.jpg'
    fig.savefig(output_filename)
    plt.close(fig)

