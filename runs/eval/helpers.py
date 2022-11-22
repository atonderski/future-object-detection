import os


def add_hardcoded_eval_args(args, default_checkpoint_name):
    args.epochs = 1
    args.load_only_net = True
    args.restart = False
    args.no_checkpoints = True
    args.short_train = True
    args.debug = False
    args.wandb_resume_id = None
    if args.checkpoint is None:
        args.checkpoint = os.path.join("checkpoints", default_checkpoint_name + ".pth.tar")
    assert os.path.exists(args.checkpoint), "Need to provide a valid checkpoint"
