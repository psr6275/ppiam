import argparse
import logging
import os

from crypten_utils import MultiProcessLauncher
import crypten.communicator as comm

parser = argparse.ArgumentParser(description="CrypTen Evaluation")
parser.add_argument(
    "-d",
    "--dataset",
    default="cifar10",
    type=str,
    help = "select dataset among mnist and cifar10"
)

parser.add_argument(
    "--world_size",
    type=int,
    default=2,
    help="The number of parties to launch. Each party acts as its own process",
)

parser.add_argument(
    "-b",
    "--batch-size",
    default=128,
    type=int,
    metavar="N",
    help="mini-batch size (default: 128)",
)

parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)

parser.add_argument(
    "--net1-location",
    default=None,
    type=str,
    metavar="PATH",
    help="path to real model checkpoint (default: none)",
)

parser.add_argument(
    "--net2-location",
    default=None,
    type=str,
    metavar="PATH",
    help="path to fake model checkpoint (default: none)",
)

parser.add_argument(
    "--data-location",
    default="../data/mnist",
    type=str,
    metavar="PATH",
    help="path to fake model checkpoint (default: none)",
)

parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)

parser.add_argument("--tau", default=0.99, type=float, help="threshold parameter for the combined network")

parser.add_argument(
    "--skip-plaintext",
    default=False,
    action="store_true",
    help="Skip validation for plaintext network",
)

parser.add_argument(
    "--multiprocess",
    default=True,
    action="store_true",
    help="Run example in multiprocess mode",
)

parser.add_argument(
    "--cond-bool",
    default=True,
    action="store_false",
    help="use the soft combination",
)

parser.add_argument(
    "--same-net",
    default=False,
    action="store_true",
    help="use the soft combination",
)
parser.add_argument(
    "--vanilla",
    default=False,
    action="store_true",
    help="vanilla model"
)

def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    from crypten_utils import run_mpc_mnist, run_mpc_cifar, run_mpc_cifar_vanilla, run_mpc_mnist_vanilla

    # Only Rank 0 will display logs.
    level = logging.INFO
    if "RANK" in os.environ and os.environ["RANK"] != "0":
        level = logging.CRITICAL
    logging.getLogger().setLevel(level)
    if args.dataset =="mnist":
        if not args.vanilla:
            run_mpc_mnist(
                args.batch_size,
                args.net1_location,
                args.net2_location,
                args.data_location,
                args.seed,
                args.tau,
                args.skip_plaintext,
                args.print_freq,
                args.cond_bool,
                args.same_net,
            )
        else:
            run_mpc_mnist_vanilla(
            args.batch_size,
            args.net1_location,
            args.data_location,
            args.seed,
            args.skip_plaintext,
            args.print_freq,
            )

    else:
        # args.net1_location = "../results/cifar_orig.pth"
        # if args.same_net:
        #     args.net2_location = "../results/cifar_fake_swd.pth"
        # else:
        #     args.net2_location = "../results/cifar_fake_small_swd.pth"
        if not args.vanilla:
            run_mpc_cifar(
            args.batch_size,
            args.net1_location,
            args.net2_location,
            args.data_location,
            args.seed,
            args.tau,
            args.skip_plaintext,
            args.print_freq,
            args.cond_bool,
            args.same_net,
            )
        else:
            run_mpc_cifar_vanilla(
                args.batch_size,
                args.net1_location,
                args.data_location,
                args.seed,
                args.skip_plaintext,
                args.print_freq,
            )
    

    print("="*10)
    print("total communication stats")
    comm.get().print_communication_stats()


    
    
def main(run_experiment):
    args = parser.parse_args()
    if args.multiprocess:
        launcher = MultiProcessLauncher(args.world_size, run_experiment, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main(_run_experiment)