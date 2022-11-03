import os
import argparse
from config import eval_config

parser = argparse.ArgumentParser(description="Experiments")
parser.add_argument(
    "-d",
    "--dataset",
    default="cifar10",
    type=str,
    help = "select dataset among mnist and cifar10"
)
parser.add_argument(
    "--exp", default=1, type=int, help="experiment number 1-4"
)
parser.add_argument(
    "--evaluate", default=False, action="store_true", help="only evaluation"
)
parser.add_argument(
    "--smooth",
    default=False,
    action="store_true",
    help="smoothing combined network (default:False)"
)
parser.add_argument(
    "--device",
    default='cuda:0',
    type=str,
    help="device for training",
)


def _run_experiment(args):     
    from main_utils import construct_fake_model   
    if args.exp==1:
        # soft attack for MPC AM
        net1_location = model_location(args.dataset, False)
        _,net2_location = construct_fake_model(args.dataset,False, False, False)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir,"--soft --dataset "+args.dataset, "--net1-location "+net1_location,"--net2-location "+net2_location, "--device "+args.device]        
    elif args.exp==2:
        # soft attack for MPC IAM
        net1_location = model_location(args.dataset, False)
        _,net2_location = construct_fake_model(args.dataset,False, False, True)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir,"--soft --dataset "+args.dataset, "--net1-location "+net1_location,"--net2-location "+net2_location, "--swd", "--device "+args.device]        
    elif args.exp==3:
        # hard attack for MPC AM
        net1_location = model_location(args.dataset, False)
        _,net2_location = construct_fake_model(args.dataset,False, False, False)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir,"--dataset "+args.dataset, "--net1-location "+net1_location,"--net2-location "+net2_location, "--device "+args.device]        
    elif args.exp==4:
        # hard attack for MPC IAM
        net1_location = model_location(args.dataset, False)
        _,net2_location = construct_fake_model(args.dataset,False, False, True)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir, "--dataset "+args.dataset, "--net1-location "+net1_location,"--net2-location "+net2_location, "--swd", "--device "+args.device]        
    elif args.exp==5:
        # soft attack for HE AM
        net1_location = model_location(args.dataset, True)
        _,net2_location = construct_fake_model(args.dataset,True, True, False)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir,"--soft --dataset "+args.dataset, "--HE", "--net1-location "+net1_location,"--net2-location "+net2_location, "--same-net", "--device "+args.device]
    elif args.exp==6:
        # soft attack for HE IAM
        net1_location = model_location(args.dataset, True)
        _,net2_location = construct_fake_model(args.dataset,True, True, True)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir,"--soft --dataset "+args.dataset, "--HE","--net1-location "+net1_location,"--net2-location "+net2_location, "--swd", "--same-net", "--device "+args.device]
    elif args.exp==7:
        # hard attack for HE AM
        net1_location = model_location(args.dataset, True)
        _,net2_location = construct_fake_model(args.dataset,True, True, False)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir,"--dataset "+args.dataset, "--HE", "--net1-location "+net1_location,"--net2-location "+net2_location, "--same-net", "--device "+args.device]
    elif args.exp==8:
        # hard attack for HE IAM
        net1_location = model_location(args.dataset, True)
        _,net2_location = construct_fake_model(args.dataset,True, True, True)
        net2_location = os.path.join(eval_config.save_dir, net2_location)
        default_setup = ["--ckpt-location "+eval_config.save_dir,"--dataset "+args.dataset, "--HE","--net1-location "+net1_location,"--net2-location "+net2_location, "--swd", "--same-net", "--device "+args.device]
    else:
        raise NotImplementedError("experiment should be chosen within 1-8")


    
    if args.smooth:
        assert args.exp<=4
        default_setup += ["--smooth"]

    if args.exp<=4 and args.dataset=="mnist":
        tau_list = eval_config.tau_list_h
    else:
        tau_list = eval_config.tau_list

    # exp_list = []
    if not args.evaluate:
        for tau in tau_list:
            exp_setup = default_setup + ["--tau %s"%tau]
            cmd = "python attack.py "
            for elem in exp_setup:
                cmd += str(elem) + " "
            os.system(cmd)
       
    
    cmd = "python evaluate.py "
    for elem in default_setup[1:]:
        cmd += str(elem) + " "
    os.system(cmd)

def model_location(dataset, HE):
    if HE:
        if dataset =="mnist":
            net1_location = os.path.join(eval_config.save_dir,"mnist_HE_orig_oe.pth")
        else:
            net1_location = os.path.join(eval_config.save_dir,"cifar_HE_orig_oe.pth")
    else:
        if dataset =="mnist":
            net1_location = os.path.join(eval_config.save_dir,"mnist_orig_oe.pth")
        else:
            net1_location = os.path.join(eval_config.save_dir,"cifar_orig_oe.pth")
    return net1_location


def main(run_experiment):
    args = parser.parse_args()
    run_experiment(args)

if __name__ == "__main__":
    main(_run_experiment)