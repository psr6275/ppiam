import argparse
import logging
import os

parser = argparse.ArgumentParser(description="Train the NN models")
parser.add_argument(
    "-d",
    "--dataset",
    default="cifar10",
    type=str,
    help = "select dataset among mnist and cifar10"
)
parser.add_argument(
    "--ood",
    default=False,
    action="store_true",
    help="out-of-distribution regularization"
)
parser.add_argument(
    "--seed", default=1228, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "-f",
    "--skip-fakenet",
    default=False,
    action="store_true",
    help="Skip fake network training",
)
parser.add_argument(
    "--net1-location",
    default=None,
    type=str,
    metavar="PATH",
    help="ckpt path for the trained original network (default: None)",
)
parser.add_argument(
    "--ckpt-location",
    default="../results",
    type=str,
    metavar="PATH",
    help="path to model checkpoint",
)
parser.add_argument(
    "--data-location",
    default="../data",
    type=str,
    metavar="PATH",
    help="path to dataset",
)
parser.add_argument(
    "--HE",
    default=False,
    action="store_true",
    help="train the network for HE"
)
parser.add_argument(
    "--swd",
    default=False,
    action="store_true",
    help="train the fake network with SWD regularization"
)
parser.add_argument(
    "--same-net",
    default=False,
    action="store_true",
    help="use the same size of network",
)

parser.add_argument(
    "--device",
    default='cuda:0',
    type=str,
    help="device for training",
)

def _run_training(args):
    from main_utils import run_cifar_train, run_mnist_train
    if args.dataset == "mnist":
        from config import mnist_config as train_config
    else:
        from config import cifar_config as train_config

    assert args.dataset in ['cifar10', 'mnist']
    # level=logging.INFO
    # if "RANK" in os.environ and os.environ["RANK"] != "0":
    #     level = logging.CRITICAL
    # logging.getLogger().setLevel(level)

    if not os.path.exists(args.ckpt_location):
        os.makedirs(args.ckpt_location)

    train_config.save_dir = args.ckpt_location
    train_config.data_dir = args.data_location
    train_config.net1_location = args.net1_location

    log_file = os.path.join(args.ckpt_location, "train_%s_HE_%s_OE_%s_SWD_%s_samenet_%s.log"\
                                %(args.dataset,args.HE,args.ood,args.swd, args.same_net))
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    if args.dataset =='cifar10':
        run_cifar_train(
            train_config,
            args.device,
            args.seed,
            args.ood, 
            args.HE,
            args.skip_fakenet,
            args.swd,
            args.same_net,
        )
    else:
        run_mnist_train(
            train_config,
            args.device,
            args.seed,
            args.ood, 
            args.HE,
            args.skip_fakenet,
            args.swd,
            args.same_net,
            )

def main(run_experiment):
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main(_run_training)