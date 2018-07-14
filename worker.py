from argparse import ArgumentParser
from src.worker.main import main

# CLI for our worker

parser = ArgumentParser()

parser.add_argument("-p", "--procs", dest="procs",default=1, type=int)
parser.add_argument("-e", "--env", dest="env", default="CartPole-v0")

args = parser.parse_args()


main(
    env_name = args.env,
    procs    = args.procs
)
