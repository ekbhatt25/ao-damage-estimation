"""
CLI entry point for the Mask R-CNN car damage pipeline.

Commands:
  train  --mode parts|damage          Fine-tune a model
  eval   --mode parts|damage          Evaluate a trained model (mAP + latency)
  infer  --image <path> [--out <path>] Run inference on a single image
  run    --mode parts|damage          Train then immediately evaluate

Examples:
  python run_maskrcnn.py train --mode parts
  python run_maskrcnn.py train --mode damage
  python run_maskrcnn.py eval  --mode parts
  python run_maskrcnn.py eval  --mode damage
  python run_maskrcnn.py infer --image data/Car\\ damages\\ dataset/File1/img/Car\\ damages\\ 100.png
  python run_maskrcnn.py run   --mode parts
"""
import argparse
import sys
from pathlib import Path

# Make sure the project root is on the path
sys.path.insert(0, str(Path(__file__).parent))


def cmd_train(args):
    from backend.mask_rcnn.train import train
    train(args.mode)


def cmd_eval(args):
    from backend.mask_rcnn.evaluate import evaluate
    evaluate(args.mode)


def cmd_infer(args):
    from backend.mask_rcnn.inference import infer_and_save
    result = infer_and_save(args.image, args.out)
    import json
    print(json.dumps(result, indent=2, default=str))


def cmd_run(args):
    from backend.mask_rcnn.train import train
    from backend.mask_rcnn.evaluate import evaluate
    train(args.mode)
    evaluate(args.mode)


def main():
    parser = argparse.ArgumentParser(
        description="Mask R-CNN car damage pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Fine-tune a model")
    p_train.add_argument("--mode", choices=["parts", "damage"], required=True)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a trained model")
    p_eval.add_argument("--mode", choices=["parts", "damage"], required=True)

    # infer
    p_infer = sub.add_parser("infer", help="Run inference on a single image")
    p_infer.add_argument("--image", required=True)
    p_infer.add_argument("--out",   default=None)

    # run (train + eval)
    p_run = sub.add_parser("run", help="Train then immediately evaluate")
    p_run.add_argument("--mode", choices=["parts", "damage"], required=True)

    args = parser.parse_args()
    dispatch = {
        "train": cmd_train,
        "eval":  cmd_eval,
        "infer": cmd_infer,
        "run":   cmd_run,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
