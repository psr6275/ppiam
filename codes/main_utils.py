import logging
import os, random
import shutil, tempfile
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from utils import load_cifar10, load_mnist, load_kmnist, load_cifar100, load_emnist_letters, load_tinyimagenet
from utils import Net_logsoftmax, Net_softmax
from models import CifarNet, CifarSmNet, CifarHESmNet, CifarHENet
from models import MnistHENet, MnistNet, MnistSmNet, MnistHESmNet

from config import cifar_names as cn
from config import mnist_names as mn

def run_cifar_eval(config, device, seed, HE, vanilla, swd, same_net, soft, smooth):
    data = "cifar10"
    batch_size = config.batch_size
    if config.net1_location is None:
        net1,ckpt1=construct_orig_model(data,HE)
        net1_location = os.path.join(config.save_dir,ckpt1)
    else:
        if os.path.exists(config.net1_location):    
            net1_location = config.net1_location
            net1,_=construct_orig_model(data,HE)        
        else:
            net1,ckpt1=construct_orig_model(data,HE)
            net1_location = os.path.join(config.save_dir,ckpt1)
        
    net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
    
    if not vanilla:
        ## load fake net
        if config.net2_location is None:
            net2, ckpt2 = construct_fake_model(data, HE, same_net, swd)
            net2_location = os.path.join(config.save_dir,ckpt2)
        else:
            if os.path.exists(config.net2_location):    
                net2_location = config.net2_location
                net2, _ = construct_fake_model(data, HE, same_net, swd)
            else:                
                net2, ckpt2 = construct_fake_model(data, HE, same_net, swd)
                # print(ckpt2)
                net2_location = os.path.join(config.save_dir,ckpt2)
        net2.load_state_dict(torch.load(net2_location,map_location='cpu'))

    train_loader, test_loader = load_cifar10(data_dir = "../data/cifar10", batch_size=batch_size, 
                                                test_batch=batch_size, train_shuffle=False)
    
    if vanilla:
        combnet = net1
        
        if HE:
            att_name = cn.att_HE
        else:
            att_name = cn.att                
        
        if soft:
            
            att_net = CifarNet()            
            att_name = append_model_name(att_name,"_soft_kl")
        else:
            att_net = CifarNet()            
        
        att_net.load_state_dict(torch.load(os.path.join(config.save_dir,att_name)))
        _,_= eval_tradeoff(combnet, att_net, test_loader,device)
    else:
        tradeoffs = np.ones(shape=(len(config.tau_list),2))
        for i,tau in enumerate(config.tau_list):        
            combnet = construct_comb_model(net1, net2,tau, HE, smooth)            
            
            att_name = generate_att_model_name(data, tau, HE, swd, same_net, smooth)
            if soft:                
                att_net = CifarNet()     
                att_name = append_model_name(att_name,"_soft_kl")
            else:
                att_net = CifarNet()     
            
            att_net.load_state_dict(torch.load(os.path.join(config.save_dir,att_name)))
            logging.info("===== The loaded attack model:  %s/%s ====="%(config.save_dir,att_name))
            
            tradeoffs[i,0],tradeoffs[i,1] = eval_tradeoff(combnet, att_net, test_loader,device)
    
        res = ".".join(generate_att_model_name(data, "", HE, swd, same_net).split(".")[:-1])+".npy"
        np.save(os.path.join(config.save_dir,res), tradeoffs)
        

def run_mnist_eval(config, device, seed, HE, vanilla, swd, same_net, soft, smooth):
    data = "mnist"
    batch_size = config.batch_size
    if config.net1_location is None:
        net1,ckpt1=construct_orig_model(data,HE)
        net1_location = os.path.join(config.save_dir,ckpt1)
    else:
        if os.path.exists(config.net1_location):    
            net1_location = config.net1_location
            net1,_=construct_orig_model(data,HE)        
        else:
            net1,ckpt1=construct_orig_model(data,HE)
            net1_location = os.path.join(config.save_dir,ckpt1)
        
    net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
    
    
    if not vanilla:
        ## load fake net
        if config.net2_location is None:
            net2, ckpt2 = construct_fake_model(data, HE, same_net, swd)
            net2_location = os.path.join(config.save_dir,ckpt2)
        else:
            if os.path.exists(config.net2_location):    
                net2_location = config.net2_location
                net2, _ = construct_fake_model(data, HE, same_net, swd)
            else:
                net2, ckpt2 = construct_fake_model(data, HE, same_net, swd)
                net2_location = os.path.join(config.save_dir,ckpt2)
        net2.load_state_dict(torch.load(net2_location,map_location='cpu'))

    train_loader, test_loader = load_mnist(data_dir = "../data/mnist", batch_size=batch_size, 
                                                test_batch=batch_size, train_shuffle=False)
    if not HE:
        tau_list = config.tau_list_h
    else:
        tau_list = config.tau_list
    tradeoffs = np.ones(shape=(len(tau_list),2))
    if vanilla:
        combnet = net1
        
        if HE:
            att_name = mn.att_HE
        else:
            att_name = mn.att
        
                   
        
        if soft:            
            att_net = MnistNet()
            att_name = append_model_name(att_name,"_soft_kl")
        else:
            att_net = MnistNet()
        
        att_net.load_state_dict(torch.load(os.path.join(config.save_dir,att_name)))
        _,_= eval_tradeoff(combnet, att_net, test_loader,device)
    else:
        for i,tau in enumerate(tau_list):        
            combnet = construct_comb_model(net1, net2,tau, HE, smooth)
               
            att_name = generate_att_model_name(data, tau, HE, swd, same_net, smooth)
            if soft:                
                att_net = MnistNet()  
                att_name = append_model_name(att_name,"_soft_kl")
            else:
                att_net = MnistNet()  
            
            att_net.load_state_dict(torch.load(os.path.join(config.save_dir,att_name)))
            logging.info("===== The loaded attack model:  %s/%s ====="%(config.save_dir,att_name))
            
            tradeoffs[i,0],tradeoffs[i,1] = eval_tradeoff(combnet, att_net, test_loader,device)
    
        res = ".".join(generate_att_model_name(data, "", HE, swd, same_net).split(".")[:-1])+".npy"
        np.save(os.path.join(config.save_dir,res), tradeoffs)

def eval_tradeoff(target_net, att_net, test_loader, device):
    from utils import test_model
    loss = nn.NLLLoss()
    target_net.to(device)
    _, acc_avg, acc_std = test_model(target_net, test_loader, loss, device, 100.0, pred_prob = True)
    target_net.cpu()
    att_net.to(device)
    _, acc_att_avg, acc_att_std = test_model(att_net, test_loader, loss, device, 100.0, pred_prob = True)
    att_net.cpu()
    logging.info("*****Target Acc/Attack Acc: {:.3f} ({:.3f})/{:.3f} ({:.3f})*****".format(acc_avg, acc_std, acc_att_avg, acc_att_std))
    return acc_avg, acc_att_avg


def run_cifar_attack(config, device, seed, HE, vanilla, swd,tau, same_net, soft, smooth=False):
    batch_size = config.batch_size
    ## load orig net
    if config.net1_location is None:
        net1,ckpt1=construct_orig_model("cifar10",HE)
        net1_location = os.path.join(config.save_dir,ckpt1)
    else:
        if os.path.exists(config.net1_location):    
            net1_location = config.net1_location
            net1,_=construct_orig_model("cifar10",HE)        
        else:
            net1,ckpt1=construct_orig_model("cifar10",HE)
            net1_location = os.path.join(config.save_dir,ckpt1)
        
    net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
    
    if vanilla:
        combnet = net1 
        logging.info("===== Target model: %s====="%net1_location)       
    else:
        ## load fake net
        if config.net2_location is None:
            net2, ckpt2 = construct_fake_model("cifar10", HE, same_net, swd)
            net2_location = os.path.join(config.save_dir,ckpt2)
        else:
            if os.path.exists(config.net2_location):    
                net2_location = config.net2_location
                net2, _ = construct_fake_model("cifar10", HE, same_net, swd)
            else:
                net2, ckpt2 = construct_fake_model("cifar10", HE, same_net, swd)
                net2_location = os.path.join(config.save_dir,ckpt2)
        # print(net2_location)        
        net2.load_state_dict(torch.load(net2_location,map_location='cpu'))        
        combnet = construct_comb_model(net1, net2,tau, HE, smooth)
        logging.info("===== Target model (combnet) net1|net2 with tau %s: %s|%s====="%(tau, net1_location,net2_location))
    
    ## load steal loader
    stealloader,_ = load_cifar100(data_dir="../data/cifar100", batch_size=config.batch_size, train_shuffle=True)
    
    ## attack model    
    att_net = CifarNet()
    optim_att = optim.Adam(att_net.parameters(), config.learning_rate)
    if vanilla: 
        if HE:
            save_model = cn.att_HE
        else:
            save_model = cn.att
    else:
        save_model = generate_att_model_name("cifar10", tau, HE, swd, same_net, smooth)
    
    if soft:
        logging.info("===== Soft Attack for CIFAR10 =====")
        from utils import train_attmodel_soft
        save_model = append_model_name(save_model,"_soft_kl")
        
        loss = nn.KLDivLoss()        
        att_net = train_attmodel_soft(att_net,combnet, stealloader, optim_att, device, loss, config.att_epoch, 
                                      kl_loss = True, save_dir=config.save_dir, save_model=save_model)
    else:
        from utils import train_attmodel_hard
        logging.info("===== Attack for CIFAR10 =====")
        loss = nn.CrossEntropyLoss()
        att_net = train_attmodel_hard(att_net,combnet , stealloader, optim_att, device, loss, config.att_epoch, 
                                    save_dir=config.save_dir, save_model=save_model)
        
    torch.save(att_net.state_dict(), os.path.join(config.save_dir,save_model))
    logging.info("===== The attack model saved in  %s/%s ====="%(config.save_dir,save_model))

def run_mnist_attack(config, device, seed, HE, vanilla, swd, tau,same_net, soft, smooth=False):
    batch_size = config.batch_size
    ## load orig net
    if config.net1_location is None:
        net1,ckpt1=construct_orig_model("mnist",HE)
        net1_location = os.path.join(config.save_dir,ckpt1)
    else:
        if os.path.exists(config.net1_location):    
            net1_location = config.net1_location
            net1,_=construct_orig_model("mnist",HE)        
        else:
            net1,ckpt1=construct_orig_model("mnist",HE)
            net1_location = os.path.join(config.save_dir,ckpt1)
    net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
    
    if vanilla:
        combnet = net1 
        logging.info("===== Target model: %s====="%net1_location)          
    else:
        ## load fake net
        if config.net2_location is None:
            net2, ckpt2 = construct_fake_model("mnist", HE, same_net, swd)
            net2_location = os.path.join(config.save_dir,ckpt2)
        else:
            if os.path.exists(config.net2_location):    
                net2_location = config.net2_location
                net2, _ = construct_fake_model("mnist", HE, same_net, swd)
            else:
                net2, ckpt2 = construct_fake_model("mnist", HE, same_net, swd)
                net2_location = os.path.join(config.save_dir,ckpt2)
        net2.load_state_dict(torch.load(net2_location,map_location='cpu'))
        combnet = construct_comb_model(net1, net2,tau, HE, smooth)
        logging.info("===== Target model (combnet) net1|net2 with tau %s: %s|%s====="%(tau, net1_location,net2_location))
    
    ## load steal loader
    stealloader,_ = load_emnist_letters(data_dir="../data/mnist", batch_size=config.batch_size, train_shuffle=True)
    
    
    ## attack model    
    att_net = MnistNet()
    optim_att = optim.Adam(att_net.parameters(), config.learning_rate)
    if vanilla: 
        if HE:
            save_model = mn.att_HE
        else:
            save_model = mn.att
    else:
        save_model = generate_att_model_name("mnist", tau, HE, swd, same_net, smooth)

    if soft:
        logging.info("===== Soft Attack for MNIST =====")
        from utils import train_attmodel_soft
        save_model = append_model_name(save_model,"_soft_kl")        
        loss = nn.KLDivLoss()   
        att_net = train_attmodel_soft(att_net,combnet, stealloader, optim_att, device, loss, config.att_epoch, 
                                    kl_loss = True, save_dir=config.save_dir, save_model=save_model)
    else:
        from utils import train_attmodel_hard
        logging.info("===== Attack for MNIST =====")
        loss = nn.CrossEntropyLoss()
        att_net = train_attmodel_hard(att_net,combnet , stealloader, optim_att, device, loss, config.att_epoch, 
                                    save_dir=config.save_dir, save_model=save_model)

    torch.save(att_net.state_dict(), os.path.join(config.save_dir,save_model))
    logging.info("===== The attack model saved in  %s/%s ====="%(config.save_dir,save_model))   

def run_cifar_train(config, device, seed, ood, 
                    HE, skip_fakenet, swd, same_net):
    
    batch_size = config.batch_size
    net1_location = config.net1_location

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    train_loader, _ = load_cifar10(data_dir = "../data/cifar10", batch_size=batch_size, 
                                                test_batch=batch_size, train_shuffle=True)
    from utils import train_valid_split
    tr_loader, val_loader = train_valid_split(train_loader, total_data=50000, ratio = 0.9, batch_size = 128,seed=seed,drop_last=True)

    torch.cuda.set_device(device)    

    logging.info("===== Training Original Network for CIFAR10 =====")

    if HE:
        net1 = CifarHENet()        
    else:
        net1 = CifarNet(10)
    
    if config.net1_location is not None:
        # if os.path.exists(config.net1_location):
        net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
        print("loaded model in %s"%net1_location)
    else:
        net1 = cifar_train_model(net1, HE, tr_loader, config, device, ood, val_loader)
        

    if skip_fakenet ==False:
        from utils import make_st_loader, make_logit_st_loader
        if swd:            
            st_loader = make_st_loader(net1, val_loader, device)
        else:
            st_loader = None
        cifar_train_fake_model(tr_loader, st_loader, HE, config, device, swd, same_net, val_loader)        
    return

def run_mnist_train(config, device, seed, ood, 
                    HE, skip_fakenet, swd, same_net):
    
    
    batch_size = config.batch_size
    net1_location = config.net1_location
    

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    train_loader, _ = load_mnist(data_dir = "../data/mnist", batch_size=batch_size, 
                                                test_batch=batch_size, train_shuffle=True)
    from utils import train_valid_split
    tr_loader, val_loader = train_valid_split(train_loader, total_data=50000, ratio = 0.9, batch_size = 128,
                                                seed=seed, datatype="mnist")

    torch.cuda.set_device(device)    

    logging.info("===== Training Original Network for MNIST =====")

    if HE:
        net1 = MnistHENet()
    else:
        net1 = MnistNet()
    
    if config.net1_location is not None:
        # if os.path.exists(config.net1_location):
        net1.load_state_dict(torch.load(net1_location,map_location='cpu'))
        print("loaded model in %s"%net1_location)
        # else:

    else:
        net1 = mnist_train_model(net1, HE, tr_loader, config, device, ood, val_loader)

    if skip_fakenet ==False:
        from utils import make_st_loader, make_logit_st_loader
        if swd:
            st_loader = make_st_loader(net1, tr_loader, device)
        else:
            st_loader = None

        mnist_train_fake_model(tr_loader, st_loader, HE, config, device, swd, same_net, val_loader)
    return    

def mnist_train_model(net, HE, train_loader, config, device, ood, test_loader=None):
       
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    if HE:
        loss = nn.NLLLoss()
        pred_prob = True
    else:
        loss = nn.CrossEntropyLoss()
        pred_prob = False
    
    if ood:
        from utils import train_model_with_oe_KL

        loss_out = nn.KLDivLoss()

        if HE:
            save_model = mn.orig_HE_oe
        else:
            save_model = mn.orig_oe

        batch_size = train_loader.batch_size
        
        outlier_loader,_ = load_kmnist(data_dir="../data/kmnist", 
                                   batch_size=batch_size, train_shuffle=True)
        
        logging.info("===== Training model for MNIST with OE (KMNIST)=====")
        net = train_model_with_oe_KL(net, train_loader, outlier_loader, optimizer, device,loss, loss_out, 
                                    config.oe_weight, config.orig_epoch, pred_prob,test_loader, config.save_dir, 
                                    save_model=save_model)
    else:
        from utils import train_model

        if HE:
            save_model = mn.orig_HE
        else:
            save_model = mn.orig

        logging.info("===== Training model for MNIST =====")
        net = train_model(net, train_loader, optimizer, device, loss, config.orig_epoch, pred_prob, test_loader, config.save_dir,
                        save_model=save_model)
    logging.info("===== The original model saved in  %s/%s ====="%(config.save_dir,save_model))
    
    return net

def mnist_train_fake_model(train_loader, st_loader, HE, config, device, swd, same_net, val_loader):
    net, save_model = construct_fake_model("mnist", HE, same_net,swd)
    optimizer = optim.Adam(net.parameters(), lr = config.learning_rate)
    loss = nn.NLLLoss()
    if HE:        
        pred_prob = True
        
    else:
        pred_prob = False
        

    epochs = config.fake_epoch
    
    if swd:
        
        from utils import train_fake_model_NLL_swd
        logging.info("===== Training fake model with SWD for MNIST =====")
        net = train_fake_model_NLL_swd(net, train_loader, st_loader, optimizer, device, loss, epochs, 
                                pred_prob, config.swd_weight, val_loader, config.save_dir, save_model)
    else:
        from utils import train_fakenet_NLL
        logging.info("===== Training fake model for MNIST =====")
        loss = nn.NLLLoss()
        net = train_fakenet_NLL(net, train_loader, optimizer, device, loss, epochs, pred_prob, 
                                val_loader, config.save_dir, save_model)
    logging.info("===== The fake model saved in  %s/%s ====="%(config.save_dir,save_model))

def cifar_train_fake_model(train_loader, st_loader, HE, config, device, swd, same_net, val_loader):
    net, save_model = construct_fake_model("cifar10", HE, same_net,swd)

    optimizer = optim.Adam(net.parameters(), lr = config.learning_rate)
    loss = nn.NLLLoss()
    if HE:        
        pred_prob = True
        
    else:        
        pred_prob = False
        

    epochs = config.fake_epoch
    
    if swd:
        
        from utils import train_fakenet_NLL,train_fake_model_NLL_swd
        logging.info("===== Training fake model with SWD for CIFAR10 =====")
        net = train_fakenet_NLL(net, train_loader, optimizer, device, loss, epochs, pred_prob, 
                                val_loader, config.save_dir, save_model)
        
        net = train_fake_model_NLL_swd(net, train_loader, st_loader, optimizer, device, loss, config.swd_epoch, 
                                pred_prob, config.swd_weight, val_loader, config.save_dir, save_model)
        
    else:
        from utils import train_fakenet_NLL
        logging.info("===== Training fake model for CIFAR10 =====")
        loss = nn.NLLLoss()
        net = train_fakenet_NLL(net, train_loader, optimizer, device, loss, epochs, pred_prob, 
                                val_loader, config.save_dir, save_model)
    logging.info("===== The fake model saved in %s/%s ====="%(config.save_dir,save_model))
    
def cifar_train_fake_model_OE(net1, train_loader, st_loader, HE, config, device, swd, same_net, val_loader):
    net, save_model = construct_fake_model("cifar10", HE, same_net,swd)
    
    from utils import make_st_loader, make_logit_st_loader, load_tinyimagenet
    outlier_loader = load_tinyimagenet(data_dir="../data/tiny-imagenet-200", 
                                   batch_size=train_loader.batch_size, train_shuffle=True)    
    st_outloader = make_st_loader(net1, outlier_loader, device)

    optimizer = optim.Adam(net.parameters(), lr = config.learning_rate)
    loss = nn.NLLLoss()
    if HE:        
        pred_prob = True
        
    else:        
        pred_prob = False
        

    epochs = config.fake_epoch
    
    if swd:
        
        from utils import train_fake_model_NLL_swd_OE
        logging.info("===== Training fake model with SWD for CIFAR10 =====")
        net = train_fake_model_NLL_swd_OE(net, train_loader, outlier_loader, st_loader, st_outloader, 
                                optimizer, device, loss, epochs, 
                                pred_prob, config.swd_weight, val_loader, config.save_dir, save_model)
        
    else:
        from utils import train_fakenet_NLL
        logging.info("===== Training fake model for CIFAR10 =====")
        loss = nn.NLLLoss()
        net = train_fakenet_NLL(net, train_loader, optimizer, device, loss, epochs, pred_prob, 
                                val_loader, config.save_dir, save_model)
    logging.info("===== The fake model saved in %s/%s ====="%(config.save_dir,save_model))


def cifar_train_model(net, HE, train_loader,config, device, ood, test_loader=None):
       
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
    if HE:
        loss = nn.NLLLoss()
        pred_prob = True
    else:
        loss = nn.CrossEntropyLoss()
        pred_prob = False
    
    if ood:
        from utils import train_model_with_oe_KL, train_model

        loss_out = nn.KLDivLoss()

        if HE:
            save_model = cn.orig_HE_oe
        else:
            save_model = cn.orig_oe

        batch_size = train_loader.batch_size
        outlier_loader = load_tinyimagenet(data_dir="../data/tiny-imagenet-200", 
                                   batch_size=batch_size, train_shuffle=True)
        
        logging.info("===== Training model for CIFAR10 with OE (tiny-imagenet)=====")
        net = train_model(net, train_loader, optimizer, device, loss, config.orig_epoch, pred_prob, test_loader, config.save_dir,
                        save_model=cn.orig)
        net = train_model_with_oe_KL(net, train_loader, outlier_loader, optimizer, device,loss, loss_out, 
                                    config.oe_weight, config.oe_epoch, pred_prob,test_loader, config.save_dir, 
                                    save_model=save_model)
    else:
        from utils import train_model

        if HE:
            save_model = cn.orig_HE
        else:
            save_model = cn.orig

        logging.info("===== Training model for CIFAR10 =====")
        net = train_model(net, train_loader, optimizer, device, loss, config.orig_epoch, pred_prob, test_loader, config.save_dir,
                        save_model=save_model)
    logging.info("===== The original model saved in  %s/%s ====="%(config.save_dir,save_model))
    
    return net

def make_model_names(save_dir, save_model):
    if os.path.exists(os.path.join(save_dir, save_model)):
        save_model = make_model_names(save_dir, ".".join(save_model.split('.')[:-1]) +"0.pth")
    return save_model

def append_model_name(save_model, append_str):    
    save_model = ".".join(save_model.split('.')[:-1]) +append_str+".pth"
    return save_model

def generate_att_model_name(data, tau, HE, swd, same_net, smooth=False):
    if data =="cifar10":
        mname = cn
    else:
        mname = mn
    

    if HE:
        if same_net:
            if swd:
                save_model = append_model_name(mname.att_HE,"_swd_%s"%tau)
            else:
                save_model = append_model_name(mname.att_HE,"_%s"%tau)
        else:
            if swd:
                save_model = append_model_name(mname.att_HE,"_small_swd_%s"%tau)
            else:
                save_model = append_model_name(mname.att_HE,"_small_%s"%tau)
    else:
        if same_net:
            if swd:
                save_model = append_model_name(mname.att,"_swd_%s"%tau)
            else:
                save_model = append_model_name(mname.att,"_%s"%tau)
        else:
            if swd:
                save_model = append_model_name(mname.att,"_small_swd_%s"%tau)
            else:
                save_model = append_model_name(mname.att,"_small_%s"%tau)
    if smooth:
        assert HE ==False
        save_model = append_model_name(save_model, "_smooth")
    return save_model


def construct_comb_model(net1, net2,tau, HE, smooth=False):
    if HE:
        from utils import CombNetHE
        combnet = CombNetHE(net1,net2,tau)
    else:
        from utils import CombNet, Net_softmax, CombNet_smooth
        if smooth:
            combnet = CombNet_smooth(Net_softmax(net1),Net_softmax(net2), tau, nu=1.0)        
        else:
            combnet = CombNet(Net_softmax(net1),Net_softmax(net2), tau)
    return combnet

def construct_orig_model(data, HE, ood=False):
    if data=="cifar10":
        if HE:
            net = CifarHENet()            
            if ood:
                save_model = cn.orig_HE_oe
            else:
                save_model = cn.orig_HE
        else:
            net = CifarNet()            
            if ood:
                save_model = cn.orig_oe
            else:
                save_model = cn.orig
    elif data =="mnist":
        if HE:
            net = MnistHENet()
            if ood:
                save_model = mn.orig_HE_oe
            else:
                save_model = mn.orig_HE
        else:
            net = MnistNet()
            if ood:
                save_model = mn.orig_oe
            else:
                save_model = mn.orig
    return net, save_model

def construct_fake_model(data, HE, same_net, swd):
    if data =="cifar10":
        if HE:
            if same_net:
                net = CifarHENet()                
                if swd:
                    save_model = append_model_name(cn.fake_HE,"_swd")
                else:
                    save_model = cn.fake_HE
            else:
                net=CifarHESmNet()
                if swd:
                    save_model = append_model_name(cn.fake_HE_sm,"_swd")
                else:
                    save_model = cn.fake_HE_sm
        else:
            if same_net:
                net = CifarNet()
                if swd:
                    save_model = append_model_name(cn.fake,"_swd")
                else:
                    save_model = cn.fake
            else:
                net = CifarSmNet()
                if swd:
                    save_model = append_model_name(cn.fake_sm,"_swd")
                else:
                    save_model = cn.fake_sm
    elif data =="mnist":
        if HE:
            if same_net:
                net = MnistHENet()
                if swd:
                    save_model = append_model_name(mn.fake_HE,"_swd")
                else:
                    save_model = mn.fake_HE
            else:
                net=MnistHESmNet()
                if swd:
                    save_model = append_model_name(mn.fake_HE_sm,"_swd")
                else:
                    save_model = mn.fake_HE_sm
        else:
            if same_net:
                net = MnistNet()
                if swd:
                    save_model = append_model_name(mn.fake,"_swd")
                else:
                    save_model = mn.fake
            else:
                net = MnistSmNet()
                if swd:
                    save_model = append_model_name(mn.fake_sm,"_swd")
                else:
                    save_model = mn.fake_sm
    else:
        raise Exception("Choose data in mnist, cifar10")
    return net, save_model