from argparse import ArgumentParser
from src.main import main

# CLI for our program

parser = ArgumentParser()

parser.add_argument("-p", "--procs", dest="procs",default=1, type=int)
parser.add_argument("-v", "--visual", nargs='?', dest="visual", default=False)
parser.add_argument("-e", "--env", dest="env", default="CartPole-v0")
parser.add_argument("-m", "--model", dest="model", default="64x64")

args = parser.parse_args()

if args.visual == None:
    args.visual = True

main(
    visual   = args.visual,
    env_name = args.env,
    _model   = args.model,
    procs    = args.procs
)
