import argparse
from collections import namedtuple
import json
import os


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


def calculate_metrics(experiment):
    print(experiment.path)
    final_metrics = {}

    metrics_with_ret = []
    for metric in experiment.metrics:
        if metric.get('avg_return') is not None:
            metrics_with_ret.append(metric)
    assert len(metrics_with_ret) >= 5
    last5 = metrics_with_ret[-5:]
    total_return = 0
    total_episodes = 0
    for metric in last5:
        print(metric['avg_return'])
        total_return += metric['avg_return'] * metric['episodes']
        total_episodes += metric['episodes']
    # print("total_return:", total_return)
    # print("total_episodes", total_episodes)
    avg_ret = total_return / total_episodes
    final_metrics['avg_return'] = avg_ret
    # print('avg_ret:', avg_ret)

    total_episodes = 0
    total_violated = 0
    for metric in experiment.metrics:
        episodes = metric.get('episodes')
        if episodes is not None:
            violated_pctg = metric['violation']
            total_episodes += episodes
            total_violated += violated_pctg * episodes
    violated_pctg = total_violated / total_episodes
    final_metrics['violated_pctg'] = violated_pctg
    final_metrics['total_violated'] = total_violated

    return final_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_paths', nargs='+')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    experiments = [load_experiment(path) for path in args.experiment_paths]
    results = [calculate_metrics(experiment) for experiment in experiments]

    for experiment, result in zip(experiments, results):
        path = experiment.path
        name = path.strip().strip('/').split('/')[-1]
        result['name'] = name
        result['temp'] = experiment.args['risk_temperature']
        result['end_coef'] = experiment.args['risk_end_coef']

    results = sorted(results, key=lambda x: x['avg_return'])
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
