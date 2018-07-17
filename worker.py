from argparse import ArgumentParser
from src.worker.main import main

# CLI for our worker

parser = ArgumentParser()
parser.add_argument("-p", "--procs", dest="procs",default=1, type=int)
args = parser.parse_args()


main(args.procs)
