import argparse
import logging
import os

parser = argparse.ArgumentParser(description="Attack the NN models")
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
    help="the network for HE"
)
parser.add_argument(
    "--soft",
    default=False,
    action="store_true",
    help="soft attack (default:False)"
)
parser.add_argument(
    "--smooth",
    default=False,
    action="store_true",
    help="smoothing combined network (default:False)"
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
    "--same-net",
    default=False,
    action="store_true",
    help="use the same size of network",
)
parser.add_argument("--tau", default=0.8, type=float, help="threshold parameter for the combined network")

parser.add_argument(
    "--device",
    default='cuda:0',
    type=str,
    help="device for training",
)

def _run_attack(args):
    from main_utils import run_cifar_attack, run_mnist_attack
    from config import attack_config

    # level=logging.INFO
    # if "RANK" in os.environ and os.environ["RANK"] != "0":
    #     level = logging.CRITICAL
    # logging.getLogger().setLevel(level)

    # if not os.path.exists(args.ckpt_location):
    #     os.makedirs(args.ckpt_location)

    log_file = os.path.join(args.ckpt_location, "attack_%s_HE_%s_SWD_%s_samenet_%s_tau_%.4f_soft_%s.log"\
                                %(args.dataset,args.HE,args.swd, args.same_net, args.tau, args.soft))
    logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
    
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    attack_config.save_dir = args.ckpt_location
    attack_config.data_dir = args.data_location
    attack_config.net1_location = args.net1_location
    attack_config.net2_location = args.net2_location
    if args.dataset =='cifar10':
        run_cifar_attack(
            attack_config,
            args.device,
            args.seed,
            args.HE,
            args.vanilla,
            args.swd,
            args.tau,
            args.same_net,
            args.soft,
            args.smooth,
        )
    else:
        run_mnist_attack(
            attack_config,
            args.device,
            args.seed,
            args.HE,
            args.vanilla,
            args.swd,
            args.tau,
            args.same_net,
            args.soft,
            args.smooth,
        )

def main(run_experiment):
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main(_run_attack)