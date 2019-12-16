import argparse
from collections import namedtuple
import json
import os

import matplotlib.pyplot as plt
from scipy import signal


Experiment = namedtuple('Experiment', ('path', 'metrics', 'args'))


def load_experiment(path):
    args_path = os.path.join(path, 'args.json')
    metrics_path = os.path.join(path, 'metrics.json')
    with open(args_path) as f:
        args = json.load(f)
    with open(metrics_path) as f:
        metrics = [json.loads(l) for l in f.readlines()]
    experiment = Experiment(path=path, metrics=metrics, args=args)
    return experiment


def plot_experiments(experiments):
    assert len(experiments) > 0
    plt.figure(figsize=(9, 3))

    for experiment in experiments:
        plot_experiment(experiment)

    max_steps = max([ max(metric['step'] for metric in exp.metrics) for exp in experiments ])
    plt.subplot(131)
    plt.plot(range(1, max_steps+1), [25 for _ in range(max_steps)], 'g--')
    plt.legend()
    plt.subplot(132)
    plt.plot(range(1, max_steps+1), [25 for _ in range(max_steps)], 'r--')
    plt.legend()
    plt.subplot(133)
    plt.plot(range(1, max_steps+1), [0.025 for _ in range(max_steps)], 'r--')
    plt.legend()

    plt.show()


def plot_experiment(experiment):
    name = experiment.path.strip().strip('/').split('/')[-1]

    # Average return
    plt.subplot(131)
    x = []
    y = []
    for metric in experiment.metrics:
        step = metric['step']
        avg_return = metric.get('avg_return')
        if avg_return is not None:
            x.append(step)
            y.append(avg_return)
    y = signal.savgol_filter(y, 31, 3)
    plt.plot(x, y, label=name)
    print(f'avg_ret: x:{len(x)}, y:{len(y)}')

    # Average Cost
    plt.subplot(132)
    x = []
    y = []
    for metric in experiment.metrics:
        step = metric['step']
        avg_cost = metric.get('avg_cost')
        if avg_cost is not None:
            x.append(step)
            y.append(avg_cost)
    y = signal.savgol_filter(y, 31, 3)
    plt.plot(x, y, label=name)
    print(f'avg_cost, x:{len(x)}, y:{len(y)}')

    # Cost Rate
    plt.subplot(133)
    total_cost = 0
    x = []
    y = []
    for metric in experiment.metrics:
        step = metric['step']
        avg_cost = metric.get('avg_cost')
        episodes = metric.get('episodes')
        if episodes is not None:
            total_cost += episodes * avg_cost
            cost_rate = total_cost / step
            x.append(step)
            y.append(cost_rate)
    plt.plot(x, y, label=name)
    print(f'cost_rate: x:{len(x)}, y:{len(y)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_paths', nargs='+')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    experiments = [load_experiment(path) for path in args.run_paths]
    plot_experiments(experiments)



if __name__ == '__main__':
    main()
