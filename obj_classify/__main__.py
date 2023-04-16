from argparse import ArgumentParser, FileType
from .inference import Inference
from .training import Trainer


if __name__ == "__main__":
    parser = ArgumentParser(prog="Object Classifier", description="Classifies models")
    subparsers = parser.add_subparsers(required=True, dest="command")
    infer_parse = subparsers.add_parser(name="infer")
    infer_parse.add_argument("filename", type=FileType("r", -1, "UTF-8"))
    train_parse = subparsers.add_parser(name="train")

    args = parser.parse_args()
    if args.command == 'infer':
        infer = Inference()
        print("File was classified as:", infer.classify(args.filename))

    else:
        train = Trainer()
        train.train()
