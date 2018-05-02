# title           : clean_logs.py
# description     : Scrape logs down to final result of each epoch
# author          : Becker, Brett, Tak, and Rawlinson
# date            : Monday, 30 April 2018.
# python_version  : 3.6.4
# ==================================================

import sys


def parse(filename):
    log_file = open(filename).readlines()
    lines = []

    for i in range(len(log_file)):
        if 'val_loss' in log_file[i]:
            line = log_file[i-2] + log_file[i]
            line = line.replace("[1m[32m", '').replace('[0m[0m', '')
            lines.append(line.strip())

    sys.stdout = open(filename, 'w')
    for line in lines:
        print(line, '\n\n', '-'*80, '\n')
    sys.stdout = sys.__stdout__
