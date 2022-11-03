import logging
import os
import random
import shutil
import tempfile
import time

import logging
import multiprocessing
import os
import uuid

import crypten
import crypten.communicator as comm
from crypten.config import cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from examples.util import NoopContextManager
from torchvision import datasets, transforms

from utils import load_cifar10, load_mnist

from models import CifarNet, CifarSmNet, MnistNet, MnistSmNet

cfg.communicator.verbose = True # add this line

def get_input_size(val_loader, batch_size):
    input, target = next(iter(val_loader))
    return input.size()

def construct_private_model(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = CifarNet()
        
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model
def construct_private_model_small(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = CifarSmNet()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model

def construct_private_model_mnist(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = MnistNet()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model
def construct_private_model_small_mnist(input_size, model):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = MnistSmNet()
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model

class AverageMeter:
    """Measures average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def add(self, value, n=1):
        self.sum += value * n
        self.count += n

    def value(self):
        return self.sum / self.count

def run_mpc_mnist(batch_size =128, net1_location = None, net2_location=None, data_location = "../data/mnist", 
                  seed = None, tau = 0.6, skip_plaintext=False, print_freq=10, cond_bool = True, same_net = True):    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
#     crypten.init()
    
    _, test_loader = load_mnist(data_dir=data_location, batch_size=batch_size, 
                                       test_batch = batch_size,train_shuffle=True)
    criterion = nn.NLLLoss()
    
    net1 = MnistNet()
    if same_net:
        net2 = MnistNet()        
    else:
        net2 = MnistSmNet()
    
    net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
    net2.load_state_dict(torch.load(net2_location,map_location='cpu'))
    
    if not skip_plaintext:
        logging.info("===== Evaluating plaintext combined LeNet network =====")
        validate_comb(test_loader, net1, net2, criterion, tau, print_freq, cond_bool)
    logging.info("===== Evaluating Private combined LeNet network =====")
#     print("*"*30)
#     crypten.print_communication_stats()
#     print("*"*30)
    input_size = get_input_size(test_loader, batch_size)
    net1_enc = construct_private_model_mnist(input_size, net1)
#     net2_enc = construct_private_model(input_size, net2)
    if same_net:
        net2_enc = construct_private_model_mnist(input_size, net2)        
    else:        
        net2_enc = construct_private_model_small_mnist(input_size, net2)
        
    validate_comb(test_loader, net1_enc, net2_enc, criterion, tau, print_freq, cond_bool)
    print("*"*30)
    crypten.print_communication_stats()
    print("*"*30)    
    
def run_mpc_cifar(batch_size =128, net1_location = None, net2_location=None, data_location = "../data/cifar", 
                  seed = None, tau = 0.6, skip_plaintext=False, print_freq=10, cond_bool = True, same_net = True):    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
#     crypten.init()
    
    _, test_loader = load_cifar10(data_dir="../data/cifar10", batch_size=batch_size, 
                                       test_batch = batch_size,train_shuffle=True)
    criterion = nn.NLLLoss()
    
    net1 = CifarNet()
    if same_net:
        net2 = CifarNet()        
    else:
        net2 = CifarSmNet()
    
    net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
    net2.load_state_dict(torch.load(net2_location,map_location='cpu'))
    
    if not skip_plaintext:
        logging.info("===== Evaluating plaintext combined LeNet network =====")
        validate_comb(test_loader, net1, net2, criterion, tau, print_freq, cond_bool)
    logging.info("===== Evaluating Private combined LeNet network =====")
#     print("*"*30)
#     crypten.print_communication_stats()
#     print("*"*30)
    input_size = get_input_size(test_loader, batch_size)
    net1_enc = construct_private_model(input_size, net1)
#     net2_enc = construct_private_model(input_size, net2)
    if same_net:
        net2_enc = construct_private_model(input_size, net2)        
    else:        
        net2_enc = construct_private_model_small(input_size, net2)
        
    validate_comb(test_loader, net1_enc, net2_enc, criterion, tau, print_freq, cond_bool)
    print("*"*30)
    crypten.print_communication_stats()
    print("*"*30)

def run_mpc_mnist_vanilla(batch_size =128, net_location = None, data_location = "../data/mnist", 
                  seed = None, skip_plaintext=False, print_freq=10):    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
#     crypten.init()
    
    _, test_loader = load_mnist(data_dir=data_location, batch_size=batch_size, 
                                       test_batch = batch_size,train_shuffle=True)
    criterion = nn.NLLLoss()
    
    net = MnistNet()
    
    net.load_state_dict(torch.load(net_location,map_location='cpu'))
#     net2.load_state_dict(torch.load(net2_location,map_location='cpu'))
    
    if not skip_plaintext:
        logging.info("===== Evaluating plaintext combined LeNet network =====")
        validate(test_loader, net, criterion, print_freq)
    logging.info("===== Evaluating Private combined LeNet network =====")
#     print("*"*30)
#     crypten.print_communication_stats()
#     print("*"*30)
    input_size = get_input_size(test_loader, batch_size)
    net_enc = construct_private_model_mnist(input_size, net)
#     net2_enc = construct_private_model(input_size, net2)
    validate(test_loader, net_enc, criterion, print_freq)
    print("*"*30)
    crypten.print_communication_stats()
    print("*"*30)    
    
def run_mpc_cifar_vanilla(batch_size =128, net_location = None, data_location = "../data/cifar", 
                  seed = None, skip_plaintext=False, print_freq=10):    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
#     crypten.init()
    
    _, test_loader = load_cifar10(data_dir="../data/cifar10", batch_size=batch_size, 
                                       test_batch = batch_size,train_shuffle=True)
    criterion = nn.NLLLoss()
    
    net = CifarNet()
    
    net.load_state_dict(torch.load(net_location,map_location='cpu'))
#     net2.load_state_dict(torch.load(net2_location,map_location='cpu'))
    
    if not skip_plaintext:
        logging.info("===== Evaluating plaintext combined LeNet network =====")
        validate(test_loader, net, criterion, print_freq)
    logging.info("===== Evaluating Private combined LeNet network =====")
#     print("*"*30)
#     crypten.print_communication_stats()
#     print("*"*30)
    input_size = get_input_size(test_loader, batch_size)
    net_enc = construct_private_model(input_size, net)
#     net2_enc = construct_private_model(input_size, net2)
    validate(test_loader, net_enc, criterion, print_freq)
    print("*"*30)
    crypten.print_communication_stats()
    print("*"*30)
    
def validate(val_loader, model, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if isinstance(model, crypten.nn.Module) and not crypten.is_encrypted_tensor(
                input
            ):
                input = encrypt_data_tensor_with_src(input)
#                 print("*"*10,"start inference","*"*10)
#                 crypten.print_communication_stats()
#                 print("*"*30)

            # compute output
            output = model(input)
            output = output.softmax(dim=1)
#             if isinstance(model, crypten.nn.Module):
#                 print("*"*10,"end batch",i,"*"*10)
#                 crypten.print_communication_stats()
#                 print("*"*30)
                
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text()
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output, target, topk=(1, 3))
            losses.add(loss.item(), input.size(0))
            top1.add(prec1[0], input.size(0))
            top3.add(prec3[0], input.size(0))

            # measure elapsed time
            current_batch_time = time.time() - end
            batch_time.add(current_batch_time)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec3[0],
                        top3.value(),
                    )
                )
            

        logging.info(
            " * Prec@1 {:.3f} Prec@3 {:.3f}".format(top1.value(), top3.value())
        )
    
    return top1.value()

def validate_comb(val_loader, model1, model2, criterion, tau = 0.5, print_freq=10, cond_bool = False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    
    print("world size",comm.get().get_world_size())

    # switch to evaluate mode
    model1.eval()
    model2.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            private = isinstance(model1, crypten.nn.Module) and isinstance(model2, crypten.nn.Module) and not crypten.is_encrypted_tensor(input)
#             print("private",private)
            if private:
                input = encrypt_data_tensor_with_src(input)
#                 print("*"*10,"start inference","*"*10)
#                 crypten.print_communication_stats()
#                 print("*"*30)
                
            # compute output
            output = inference_combmodel(model1, model2, input, tau,cond_bool,private=private)
            
            if private:
                print("*"*10,"end batch",i,"*"*10)
                crypten.print_communication_stats()
                print("*"*30)
                
            if crypten.is_encrypted_tensor(output):
                output = output.get_plain_text()
            loss = criterion(output.log(), target)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output, target, topk=(1, 3))
            losses.add(loss.item(), input.size(0))
            top1.add(prec1[0], input.size(0))
            top3.add(prec3[0], input.size(0))

            # measure elapsed time
            current_batch_time = time.time() - end
            batch_time.add(current_batch_time)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logging.info(
                    "\nTest: [{}/{}]\t"
                    "Time {:.3f} ({:.3f})\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 {:.3f} ({:.3f})   \t"
                    "Prec@5 {:.3f} ({:.3f})".format(
                        i + 1,
                        len(val_loader),
                        current_batch_time,
                        batch_time.value(),
                        loss.item(),
                        losses.value(),
                        prec1[0],
                        top1.value(),
                        prec3[0],
                        top3.value(),
                    )
                )

        logging.info(
            " * Prec@1 {:.3f} Prec@3 {:.3f}".format(top1.value(), top3.value())
        )
    return top1.value()

def inference_combmodel(net1_enc, net2_enc, x, tau=0.5, cond_bool = False, nu=1.0,private=True):
    x1 = net1_enc(x).softmax(dim=1)
    x2 = net2_enc(x).softmax(dim=1)
    max_val = x1.max(dim=1)[0]
    
    if cond_bool:
        cond_in = max_val>tau
    else:
        cond_in = (nu*(tau-max_val)).sigmoid()

    if private==False:
        cond_in = cond_in.float()
    out = x1*cond_in.view(-1,1) + x2*(1-cond_in.view(-1,1))
#     out = x1*cond_in.view(-1,1)
    return out

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size())
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input

class MultiProcessLauncher:

    # run_process_fn will be run in subprocesses.
    def __init__(self, world_size, run_process_fn, fn_args=None):
        env = os.environ.copy()
        env["WORLD_SIZE"] = str(world_size)
        multiprocessing.set_start_method("spawn")

        # Use random file so multiple jobs can be run simultaneously
        INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
        env["RENDEZVOUS"] = INIT_METHOD

        self.processes = []
        for rank in range(world_size):
            process_name = "process " + str(rank)
            process = multiprocessing.Process(
                target=self.__class__._run_process,
                name=process_name,
                args=(rank, world_size, env, run_process_fn, fn_args),
            )
            self.processes.append(process)

        if crypten.mpc.ttp_required():
            ttp_process = multiprocessing.Process(
                target=self.__class__._run_process,
                name="TTP",
                args=(
                    world_size,
                    world_size,
                    env,
                    crypten.mpc.provider.TTPServer,
                    None,
                ),
            )
            self.processes.append(ttp_process)

    @classmethod
    def _run_process(cls, rank, world_size, env, run_process_fn, fn_args):
        for env_key, env_value in env.items():
            os.environ[env_key] = env_value
        os.environ["RANK"] = str(rank)
        orig_logging_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.INFO)
        crypten.init()
        logging.getLogger().setLevel(orig_logging_level)
        if fn_args is None:
            run_process_fn()
        else:
            run_process_fn(fn_args)

    def start(self):
        for process in self.processes:
            process.start()

    def join(self):
        for process in self.processes:
            process.join()
            assert (
                process.exitcode == 0
            ), f"{process.name} has non-zero exit code {process.exitcode}"

    def terminate(self):
        for process in self.processes:
            process.terminate()


class LeNet(nn.Sequential):
    """
    Adaptation of LeNet that uses ReLU activations
    """

    # network architecture:
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class CombNet_CrypTen(crypten.nn.Module):
    def __init__(self, model1, model2, tau=0.5):
        super(CombNet_CrypTen, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.tau = tau
        self.sigmoid = crypten.nn.Sigmoid()
        
    def forward(self,x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        max_val = x1.max(dim=1)[0]
        cond_in = max_val>self.tau
        out = x1*cond_in.view(-1,1)+x2*(1-cond_in.view(-1,1))
        return out