import os
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    
    # path to the raw data file
    parser.add_argument(
        '--config_file',
        type=str,
        default='configs/default.yaml'
    )
    parser.add_argument(
        '--load_model',
        type=str,
        default=None
    )
    FLAGS, _ = parser.parse_known_args()

    return FLAGS, _


def logging(msg, log_file):
    with open(log_file, 'a') as fw:
        fw.write("{}\n".format(msg))
    
