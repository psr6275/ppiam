import torch
import torch.nn as nn
from .train_utils import test_model
from .eval_utils import AverageVarMeter, accuracy, correspondence_score

from torch.utils.data import SubsetRandomSampler
import torch.nn.functional as F

import numpy as np
from livelossplot import PlotLosses
import os

from .net_utils import CombNet, CombNet_smooth
from .train_utils import test_binary_model
from .eval_utils import get_prediction

from torch.utils.data import TensorDataset, DataLoader

from torch.utils.data import Subset

def knockoff_adpt_att(adpt_att_net, st_loader, comb_net, device, save_model, dataset='cifar', soft = True, 
                      save_dir = '../results/adapt_att_models'):
    adpt_loader = make_adaptive_loader(comb_net, st_loader, comb_net.tau, device, batch_size=256, shuffle=False)
    cm_preds = get_prediction(adpt_att_net,adpt_loader,device)
    cm_cond = cm_preds>0.5 # prediction of data from target network
    cm_idx = cm_cond.nonzero()
    att_dataset = Subset(st_loader.dataset, cm_idx.flatten())
    att_loader = DataLoader(att_dataset, shuffle=True, batch_size=128)
    
    ## train attack model
    print("train model for", dataset)
    if dataset == 'cifar':
        att_net = CifarNet()
        epochs = 50
    else:
        att_net = MnistNet()
        epochs = 30
    print("model is", os.path.join(save_dir, save_model))
    optim_att = optim.Adam(att_net.parameters(), 0.0001)    
    if soft:
        from .train_utils import train_attmodel_soft
        loss_clf = nn.KLDivLoss()
        print("soft attack")
        att_net = train_attmodel_soft(att_net, comb_net, att_loader, optim_att, device, loss_clf, epochs, 
                                      kl_loss = True, save_dir = save_dir,save_model=save_model)
    else:
        from .train_utils import train_attmodel_hard
        loss = nn.CrossEntropyLoss()
        print("hard attack")
        att_net = train_attmodel_hard(att_net,comb_net , att_loader, optim_att, device, loss, epochs, 
                                    save_dir=save_dir, save_model=save_model)
    return att_net

def make_adaptive_loader(victim_net, dataloader, tau, device, batch_size=128, shuffle=True):

    # model_prob.to(device).eval()
    victim_net.to(device).eval()
    model_prob = victim_net.net_orig.to(device).eval() 
    labels = []
    ori_out = []
    with torch.no_grad():
        for x,y in dataloader:
            with torch.no_grad():
                out = victim_net(x.to(device)).detach().cpu()
                ori_out.append(out)
            pred = torch.max(model_prob(x.to(device)).detach().cpu(),axis=1)
#             print(pred)
            labels.append(torch.tensor(pred[0]>tau).float())
            del x,y,pred, out
    data = torch.cat(ori_out,dim=0)
    labels = torch.cat(labels,dim=0)
    dataloader_ = DataLoader(TensorDataset(data, labels),batch_size=batch_size, shuffle=shuffle)
    model_prob.cpu()
    victim_net.cpu()
    return dataloader_

def make_adaptive_loader2(victim_net, dataloader, tau, device, batch_size=128, shuffle=True):

    # model_prob.to(device).eval()
    victim_net.to(device).eval()
    labels = []
    ori_out = []
    with torch.no_grad():
        for x,y in dataloader:
            with torch.no_grad():
                out = victim_net(x.to(device)).detach().cpu()
                ori_out.append(out)
            pred = torch.max(out,axis=1)
            # labels.append((pred[1]==y).type(torch.FloatTensor))
#             pred = torch.max(model_prob(x.to(device)).detach().cpu(),axis=1)
#             print(pred)
            labels.append(torch.tensor(pred[0]<tau).float())
            del x,y,pred, out
    data = torch.cat(ori_out,dim=0)
    labels = torch.cat(labels,dim=0)
    dataloader_ = DataLoader(TensorDataset(data, labels),batch_size=batch_size, shuffle=shuffle)
    victim_net.cpu()
    return dataloader_

def train_adapt_model(adpt_att_model, adpt_loader, criterion, optimizer, epochs, device, adpt_testloader = None, save_dir = "../results", save_model = "cifar_adapt_model.pth"):
    adpt_att_model.to(device)
    
    logs_clf = {}
    best_acc = 0.0
    liveloss_tr = PlotLosses()
    
    for epoch in range(epochs):
        adpt_att_model.train()
        
        for x,y in adpt_loader:

            adpt_att_model.zero_grad()

            out = adpt_att_model(x.to(device))
            loss = criterion(out.flatten().float(), y.to(device).float())
            
            loss.backward()
            optimizer.step()
            
            del x,out,  y, loss
            torch.cuda.empty_cache()
        logs_clf['loss'], logs_clf['acc']= test_binary_model(adpt_att_model, adpt_loader, criterion, device, 100.0, save_dir, save_model)
        if adpt_testloader is not None:
            logs_clf['val_loss'], logs_clf['val_acc']= test_binary_model(adpt_att_model, adpt_testloader, criterion, device, 0.0, save_dir, save_model)
        liveloss_tr.update(logs_clf)
        liveloss_tr.send()
    adpt_att_model.cpu()
    return adpt_att_model, logs_clf            

def select_data(trainset, nb_stolen,batch_size=128, select_shuffle = False):  #attack용 데이터 중 원하는 개수 추출
    x = trainset.data
    nb_stolen = np.minimum(nb_stolen, x.shape[0])
    rnd_index = np.random.choice(x.shape[0], nb_stolen, replace=False)
    sampler = SubsetRandomSampler(rnd_index)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=select_shuffle, sampler=sampler)
    return trainloader

def query_label(x, victim_clf, victim_clf_fake, tau, use_probability=False):   #tau 조건에 따라 net 또는 fakenet 불러옴
    victim_clf.to(device).eval()
    victim_clf_fake.to(device).eval()
    labels_in = victim_clf(x)
    labels_out = victim_clf_fake(x)
    cond_in = torch.max(labels_in, dim=1).values>tau
    labels = (labels_in*cond_in.view(-1,1)+labels_out*(~cond_in.view(-1,1)))
    
    if not use_probability:
        labels = torch.argmax(labels, axis=1)
        #labels = to_categorical(labels, nb_classes)
    
    victim_clf.cpu()
    victim_clf_fake.cpu()
    
    return labels
    
def test_model_from_taus(model, model_fake, tau_list, test_loader, criterion, device, soft = False):
    lss = []
    acs = []
    for tau in tau_list:
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        if soft:
            model_comb = CombNet_soft(model, model_fake,tau).to(device).eval()
        else:
            model_comb = CombNet(model, model_fake,tau).to(device).eval()
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            p_y = model_comb(x)
            loss = criterion(p_y,y)

            acc = accuracy(p_y.detach().cpu(), y.detach().cpu())

            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
    #         print(acc)
            del x, y, p_y, acc, loss
            torch.cuda.empty_cache()
        lss.append(losses.avg.detach().cpu())
        acs.append(accs.avg.detach().cpu())
        print("Accuracy/Loss for tau {:.1f} : {:.2f}/{:.4f}".format(tau,acs[-1],lss[-1]))
        del model_comb
        torch.cuda.empty_cache()
    return lss, acs

def test_corr_model(model1, model2, test_loader, criterion, device):
    
    model1.to(device).eval()
    model2.to(device).eval()
    
    losses1 = AverageVarMeter()
    losses2 = AverageVarMeter()
    accs1 = AverageVarMeter()
    accs2 = AverageVarMeter()
    corrs = AverageVarMeter()
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)

        p_y1 = model1(x).detach().cpu()
        p_y2 = model2(x).detach().cpu()

        loss1 = criterion(p_y1,y)
        loss2 = criterion(p_y2,y)

        acc1 = accuracy(p_y1, y)
        acc2 = accuracy(p_y2, y)
        
        corr = correspondence_score(p_y1,p_y2)

        losses1.update(loss1,x.size(0))
        losses2.update(loss2,x.size(0))
        accs1.update(acc1[0],x.size(0))
        accs2.update(acc2[0],x.size(0))
        corrs.update(corr,x.size(0))
#         print(acc)
        del x, y, p_y1, p_y2, acc1,acc2, loss1, loss2, corr
        torch.cuda.empty_cache()
    loss1 = losses1.avg.detach().cpu()
    loss2 = losses2.avg.detach().cpu()
    acc1 = accs1.avg.detach().cpu()
    acc2 = accs2.avg.detach().cpu()
    corr = corrs.avg.detach().cpu()
    
    print("Accuracy/Loss 1: {:.2f}/{:.4f}".format(acc1,loss1))
    print("Accuracy/Loss 2: {:.2f}/{:.4f}".format(acc2,loss2))
    print("Correspondence: ", corr)
    
    del losses1, losses2, accs1, accs2, corrs

    return loss1, loss2, acc1, acc2, corr