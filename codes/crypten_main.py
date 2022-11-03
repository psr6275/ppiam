import argparse
import logging
import os



parser = argparse.ArgumentParser(description="CrypTen Experiment")
parser.add_argument(
    "-d",
    "--dataset",
    default="cifar10",
    type=str,
    help = "select dataset among mnist and cifar10"
)
parser.add_argument(
    "--exp", default=1, type=int, help="experiment number 1-3"
)
parser.add_argument("--tau", default=0.9, type=float, help="threshold parameter for the combined network")
parser.add_argument(
    "--swd",
    default=False,
    action="store_true",
    help="the fake network with SWD regularization"
)

def _run_experiment(args):
    # only import here to initialize crypten within the subprocesses
    if args.dataset=="cifar10":
        data_location = "../data/cifar10"
    else:
        data_location = "../data/mnist"

    if args.exp==1:
        ## vanilla network
        net1_location, _ = model_location(args.dataset, False)
        default_setup = ["--dataset "+args.dataset, "--vanilla", "--net1-location "+net1_location, \
                            "--data-location "+data_location]
    elif args.exp==2:
        ## AM or IAM (--swd) model
        net1_location, net2_location = model_location(args.dataset, args.swd)
        default_setup = ["--dataset "+args.dataset, "--net1-location "+net1_location, "--net2-location "+net2_location,\
                         "--data-location "+data_location, "--tau "+str(args.tau)]
    elif args.exp==3:
        ## attack model for AM or IAM (--swd) (tau=0.8)
        if args.dataset=="cifar10":
            if args.swd:
                net1_location = "../results/cifar_att_small_swd_%s.pth"%args.tau
            else:
                net1_location = "../results/cifar_att_small_%s.pth"%args.tau
        else:
            if args.swd:
                net1_location = "../results/mnist_att_small_swd_%s.pth"%args.tau
            else:
                net1_location = "../results/mnist_att_small_%s.pth"%args.tau
        
        default_setup = ["--dataset "+args.dataset, "--vanilla", "--net1-location "+net1_location, \
                            "--data-location "+data_location]                     
    
    cmd = "python crypten_evaluate.py "
    for elem in default_setup:
        cmd += str(elem) + " "
    os.system(cmd)


def model_location(dataset, swd):    
    if dataset=="cifar10":
        net1_location = "../results/cifar_orig.pth"
        if swd:
            net2_location = "../results/cifar_fake_small_swd.pth"
        else:
            net2_location = "../results/cifar_fake_small.pth"
    else:
        net1_location = "../results/mnist_orig.pth"
        if swd:
            net2_location = "../results/mnist_fake_small_swd.pth"
        else:
            net2_location = "../results/mnist_fake_small.pth"
    return net1_location, net2_location

def main(run_experiment):
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main(_run_experiment)