from argparse import ArgumentParser
from src.main import main

# CLI for our program

parser = ArgumentParser()

parser.add_argument("-t", "--test", nargs='?', dest="test", default=False)
parser.add_argument("-v", "--visual", nargs='?', dest="visual", default=False)
parser.add_argument("-e", "--env", dest="env", default="CartPole-v0")
parser.add_argument("-b", "--benchmark", nargs='?', dest="bench", default=False)

args = parser.parse_args()

if args.test == None:
    args.test = True

if args.visual == None:
    args.visual = True

if args.bench == None:
    args.bench = True

print(args)

main(
    test=args.test,
    visual=args.visual,
    env_name=args.env,
    bench=True
)
