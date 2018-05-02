# title          : analyze_models.py
# description    : Analyze how models perform over time
# author         : Isaiah Rawlinson
# date           : Wednesday,  2 May 2018.
# python_version : 3.6.4
# ==================================================

import os
import matplotlib.pyplot as plt


def read_logs(directory):
    logs = {}
    for f in os.listdir(directory):
        filename = os.fsdecode(f)
        model = filename.replace('.txt', '')
        filename = os.path.join(directory, filename)

        log = open(filename, 'r').readlines()
        logs[model] = []
        epoch = {}
        # print(model)
        for i in range(0, len(log), 5):
            first = log[i][log[i].find('loss: ') + len('loss: '):
                           log[i].rfind(' | time:')]
            second = log[i+1][log[i+1].find('loss: ') + len('loss: '):
                              log[i+1].rfind(' -- iter:')]
            epoch = {
                'total_loss': float(first),
                'loss': float(second[:7]),
                'acc': float(second[15:21]),
                'val_loss': float(second[34:42]),
                'val_acc': float(second[53:59])
            }
            # pprint(epoch)
            logs[model].append(epoch)

    return logs, logs[model][0].keys()


def plot_metric(logs, metric):
    ids = [model for model in logs.keys()]
    ys = []
    for model, values in logs.items():
        points = []
        for value in values:
            points.append(value[metric])
        ys.append(points)

    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(metric + ' over ' + str(len(ys[0])) + ' epochs')

    for i in range(len(ys)):
        plt.plot(ys[i], label=ids[i])

    plt.legend()
    plt.show()


logs, metrics = read_logs('Logs')

for metric in metrics:
    plot_metric(logs, metric)
# plot_metric(logs, 'val_loss')
# plot_metric(logs, 'total_loss')
# plot_metric(logs, 'acc')
