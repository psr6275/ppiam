import argparse
import logging
import os

parser = argparse.ArgumentParser(description="Evaluate the NN models")
parser.add_argument(
    "-d",
    "--dataset",
    default="cifar10",
    type=str,
    help = "select dataset among mnist and cifar10"
)
parser.add_argument(
    "--seed", default=1228, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--net1-location",
    default=None,
    type=str,
    metavar="PATH",
    help="ckpt path for the trained original network (default: None)",
)
parser.add_argument(
    "--net2-location",
    default=None,
    type=str,
    metavar="PATH",
    help="ckpt path for the trained fake network (default: None)",
)
parser.add_argument(
    "--HE",
    default=False,
    action="store_true",
    help="the network for HE"
)
parser.add_argument(
    "--soft",
    default=False,
    action="store_true",
    help="soft attack (default:False)"
)
parser.add_argument(
    "--swd",
    default=False,
    action="store_true",
    help="the fake network with SWD regularization"
)
parser.add_argument(
    "--vanilla",
    default=False,
    action="store_true",
    help="vanilla model"
)
parser.add_argument(
    "--smooth",
    default=False,
    action="store_true",
    help="smoothing combined network (default:False)"
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
def _run_evaluate(args):
    from main_utils import run_cifar_eval, run_mnist_eval
    from config import eval_config

    # level=logging.INFO
    # if "RANK" in os.environ and os.environ["RANK"] != "0":
    #     level = logging.CRITICAL
    # logging.getLogger().setLevel(level)

    log_file = os.path.join(eval_config.save_dir, "evaluate_%s_HE_%s_SWD_%s_samenet_%s_soft_%s.log"\
                                %(args.dataset,args.HE,args.swd, args.same_net, args.soft))
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    eval_config.net1_location = args.net1_location
    eval_config.net2_location = args.net2_location
    if args.dataset =='cifar10':
        run_cifar_eval(
            eval_config,
            args.device,
            args.seed,
            args.HE,
            args.vanilla,
            args.swd,
            args.same_net,
            args.soft,
            args.smooth,
        )
    else:
        run_mnist_eval(
            eval_config,
            args.device,
            args.seed,
            args.HE,
            args.vanilla,
            args.swd,
            args.same_net,
            args.soft,
            args.smooth,
        )

def main(run_experiment):
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main(_run_evaluate)