import torch
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.optim as optim

import numpy as np
import logging
import os

import copy

from .swae_utils import sliced_wasserstein_distance
from .eval_utils import AverageVarMeter, accuracy, accuracy_b
from .net_utils import apply_taylor_softmax

SMALL = 1e-5

def test_fake_model_NLL(model, test_loader, criterion, device, worst_acc=100.0, 
                        save_dir='../results', save_model = "fake_ckpt.pth",pred_prob = True):
    """
        test fakenet if criterion is NLL
    """
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            if pred_prob:
                p_y = 1-model(x).detach().cpu()
            else:
                p_y = 1-model(x).softmax(dim=1).detach().cpu()
            loss = criterion(torch.log(torch.clamp(p_y,min=SMALL)),y)
            
            acc = accuracy(1-p_y, y.detach().cpu())
        
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
#         print(acc)
    if accs.avg<=worst_acc:
        torch.save(model.state_dict(),os.path.join(save_dir,save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu(), accs.std.detach().cpu()


def test_fake_model(model, test_loader, criterion, device, worst_acc=100.0, 
                    save_dir='../results', save_model = "fake_ckpt.pth",pred_prob = False):
    """
        test fake model if criterion directly uses mode output!
    """
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    for batch_idx, (x,y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        p_y = model(x)
        if pred_prob:
            p_y = torch.log(torch.clamp(p_y,min=SMALL))
        loss = criterion(p_y,y)
        
        acc = accuracy(p_y.detach().cpu(), y.detach().cpu())
    
        losses.update(loss,x.size(0))
        accs.update(acc[0],x.size(0))
#         print(acc)
    if accs.avg<worst_acc:
        torch.save(model.state_dict(),os.path.join(save_dir,save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu(), accs.std.detach().cpu()


def train_fake_model_NLL_swd(clf, train_loader, st_loader, optimizer, device, loss_clf, epochs, pred_prob,
                        swd_weight, test_loader, save_dir, save_model):
    clf.to(device)
    
    worst_acc = 100.0

    for epoch in range(epochs):
        losses = AverageVarMeter()
        losses1 = AverageVarMeter()
        losses2 = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        
        iterloader = iter(st_loader)
        
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()

            
#             fake_loss = 0
            try:
                batch = next(iterloader)
            except StopIteration:
                iterloader = iter(st_loader)
                batch = next(iterloader)
            
            pred = clf(x)
            if pred_prob:
                # out = torch.log(torch.clamp(1-pred,min=SMALL))
                outst = batch[0]
            else:
                pred = pred.softmax(dim=1)
                # out = torch.log(torch.clamp(1-pred.softmax(dim=1),min=SMALL))
                outst = batch[0].softmax(dim=1)
            out = torch.log(torch.clamp(1-pred,min=SMALL))

            fake_loss = loss_clf(out,y)

            swd_loss = sliced_wasserstein_distance(pred,outst.to(device),num_projections=50,p=2,device=device)
            # swd_loss = sliced_wasserstein_distance(pred.sort()[0],outst.sort()[0].to(device),num_projections=50,p=2,device=device)
            
            loss = fake_loss + swd_loss*swd_weight
            loss.backward()
            optimizer.step()
            acc = accuracy(pred.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            losses1.update(fake_loss.detach().cpu(),x.size(0))
            losses2.update(swd_loss.detach().cpu(),x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss,batch, pred, swd_loss, outst
            torch.cuda.empty_cache()
            

        if test_loader:
            _,acc_te,acc_std_te = test_fake_model_NLL(clf,test_loader,loss_clf,device,
                                                    100.0,save_dir, save_model,pred_prob)
            
            if worst_acc>acc_te:
                worst_acc = acc_te
        
            logging.info(
                    "\nEpoch [{}/{}]\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Loss1/Loss2 {:.4f}/{:.4f}\t"
                    "Prec@1 train {:.3f} ({:.3f})\t"
                    "Prec@1 test {:.3f} ({:.3f})   \t".format(
                        (epoch+1),
                        epochs,
                        losses.avg.detach().cpu(),
                        losses.std.detach().cpu(),
                        losses1.avg.detach().cpu(),
                        losses2.avg.detach().cpu(),
                        accs.avg.detach().cpu(),
                        accs.std.detach().cpu(),
                        acc_te,
                        acc_std_te,                   
                    )
            )
        else:
            torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
            logging.info(
                "\nEpoch [{}/{}]\t"
                "Loss {:.4f} ({:.4f})\t"
                "Loss1/Loss2 {:.4f}/{:.4f}\t"
                "Prec@1 train {:.3f} ({:.3f})   \t".format(
                    (epoch+1),
                    epochs,
                    losses.avg.detach().cpu(),
                    losses.std.detach().cpu(),
                    losses1.avg.detach().cpu(),
                    losses2.avg.detach().cpu(),
                    accs.avg.detach().cpu(),
                    accs.std.detach().cpu(),             
                )
            )
    clf.cpu()
    torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
    return clf



   

def train_fakenet_NLL(clf, train_loader, optimizer, device, loss_clf, epochs, pred_prob,
                        test_loader = None, save_dir = "../results",save_model="cifar_fakenet.pth"):
    
    """
        train fakenet with NLL loss without any regularization
    """
    
    
    clf.to(device)
    
    worst_acc = 100.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for x,y in train_loader:
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            if pred_prob:
                out = 1-clf(x)  
            else:
                out = 1-clf(x).softmax(dim=1)  
            #torch에서 logsoftmax 쓰듯이 log를 취해주면 학습이 더 잘됨
            loss = loss_clf(torch.log(torch.clamp(out,min=SMALL)),y)
            loss.backward()
            optimizer.step()
            
            acc = accuracy(1-out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss
            torch.cuda.empty_cache()
        if test_loader:
            _,acc_te,acc_std_te = test_fake_model_NLL(clf,test_loader,loss_clf,device,
                                                    100.0,save_dir, save_model,pred_prob)
            if worst_acc>acc_te:
                worst_acc = acc_te    
            logging.info(
                    "\nEpoch [{}/{}]\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 train {:.3f} ({:.3f})\t"
                    "Prec@1 test {:.3f} ({:.3f})   \t".format(
                        (epoch+1),
                        epochs,
                        losses.avg.detach().cpu(),
                        losses.std.detach().cpu(),
                        accs.avg.detach().cpu(),
                        accs.std.detach().cpu(),
                        acc_te,
                        acc_std_te,                   
                    )
            )
        else:
            torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
            logging.info(
                "\nEpoch [{}/{}]\t"
                "Loss {:.4f} ({:.4f})\t"
                "Prec@1 train {:.3f} ({:.3f})   \t".format(
                    (epoch+1),
                    epochs,
                    losses.avg.detach().cpu(),
                    losses.std.detach().cpu(),
                    accs.avg.detach().cpu(),
                    accs.std.detach().cpu(),             
                )
            )
    clf.cpu()
    torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
    return clf


def train_model_with_oe_KL(clf, train_loader, outlier_loader, optimizer, device, 
                          loss_in, loss_out, weight_out, epochs,pred_prob = True, test_loader = None,
                          save_dir = '../results', save_model="cifar_clf.pth"):
    clf.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        losses1 = AverageVarMeter()
        losses2 = AverageVarMeter()
        losses = AverageVarMeter()
        accs = AverageVarMeter()

        clf.train()
        for i,(in_set, out_set) in enumerate(zip(train_loader, outlier_loader)):
            x,y = in_set[0].to(device),in_set[1].to(device)
            x_out = out_set[0].to(device)
            
            clf.zero_grad()
            pred_in = clf(x)
            pred_out = clf(x_out)

            if pred_prob:
                pred_in = torch.log(torch.clamp(pred_in,min=SMALL))
            else:
                pred_out = pred_out.softmax(dim=1)

            loss1 = loss_in(pred_in,y)
            
                
            loss2 = weight_out*loss_out(pred_out.log(), torch.ones_like(pred_out)*0.1)
            loss = loss1+loss2
            
            loss.backward()
            optimizer.step()
            
            acc = accuracy(pred_in.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            losses1.update(loss1.detach().cpu(),x.size(0))
            losses2.update(loss2.detach().cpu(),x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, pred_in,x,y,loss, pred_out, x_out
            torch.cuda.empty_cache()

        if test_loader:
            _,acc_te,acc_std_te = test_model(clf,test_loader,loss_in,device,0.0,save_dir,save_model,pred_prob)
            if best_acc<acc_te:
                best_acc = acc_te    
            logging.info(
                    "\nEpoch [{}/{}]\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Loss1/Loss2 {:.4f}/{:.4f}\t"
                    "Prec@1 train {:.3f} ({:.3f})\t"
                    "Prec@1 test {:.3f} ({:.3f})   \t".format(
                        (epoch+1),
                        epochs,
                        losses.avg.detach().cpu(),
                        losses.std.detach().cpu(),
                        losses1.avg.detach().cpu(),
                        losses2.avg.detach().cpu(),
                        accs.avg.detach().cpu(),
                        accs.std.detach().cpu(),
                        acc_te,
                        acc_std_te,                   
                    )
            )
        else:
            torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
            logging.info(
                "\nEpoch [{}/{}]\t"
                "Loss {:.4f} ({:.4f})\t"
                "Loss1/Loss2 {:.4f}/{:.4f}\t"
                "Prec@1 train {:.3f} ({:.3f})   \t".format(
                    (epoch+1),
                    epochs,
                    losses.avg.detach().cpu(),
                    losses.std.detach().cpu(),
                    losses1.avg.detach().cpu(),
                    losses2.avg.detach().cpu(),
                    accs.avg.detach().cpu(),
                    accs.std.detach().cpu(),             
                )
            )
    clf.cpu()
    
    return clf

def train_attmodel_hard(clf, victim_clf, steal_loader, optimizer, device, loss_clf, epochs,
                    save_dir = "../results",save_model="cifar_clf.pth"):
    """
        steal network with soft label
    """
    
    clf.to(device)
    
    best_acc = 0.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for i,(x,y) in enumerate(steal_loader):
            x = x.to(device)
            victim_clf.to(device).eval()
            with torch.no_grad():
                fake_out = copy.deepcopy(victim_clf(x).detach().cpu())
                _, fake_label = torch.max(fake_out, dim=1)
            
            clf.zero_grad()
            out = clf(x)
            
            loss = loss_clf(out,fake_label.to(device))
            loss.backward()
            optimizer.step()

            acc = accuracy(out.detach().cpu(),fake_label.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,fake_out,loss, fake_label
            torch.cuda.empty_cache()
        
        logging.info(
            "\nEpoch [{}/{}]\t"
            "Loss {:.4f} ({:.4f})\t"
            "Prec@1 train {:.3f} ({:.3f})   \t".format(
                (epoch+1),
                epochs,
                losses.avg.detach().cpu(),
                losses.std.detach().cpu(),
                accs.avg.detach().cpu(),
                accs.std.detach().cpu(),             
            )
        )
    clf.cpu()
    return clf

def train_attmodel_soft(clf, victim_clf, steal_loader, optimizer, device, loss_clf, epochs, kl_loss = True,
                    save_dir = "../results",save_model="cifar_clf.pth"):
    """
        steal network with soft label
    """
    
    clf.to(device)
    
    best_acc = 0.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for i,(x,y) in enumerate(steal_loader):
            x = x.to(device)
            victim_clf.to(device).eval()
            with torch.no_grad():
                fake_out = copy.deepcopy(victim_clf(x).detach().cpu())                
            
            clf.zero_grad()
            out = clf(x)
            if kl_loss:
                out = out.log_softmax(dim=1)
            # print(y)
            loss = loss_clf(out,fake_out.to(device))
            loss.backward()
            optimizer.step()

            acc = accuracy(out.detach().cpu(),fake_out.max(dim=1)[1])
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,fake_out,loss
            torch.cuda.empty_cache()
        
        logging.info(
            "\nEpoch [{}/{}]\t"
            "Loss {:.4f} ({:.4f})\t"
            "Prec@1 train {:.3f} ({:.3f})   \t".format(
                (epoch+1),
                epochs,
                losses.avg.detach().cpu(),
                losses.std.detach().cpu(),
                accs.avg.detach().cpu(),
                accs.std.detach().cpu(),             
            )
        )
    clf.cpu()
    return clf


def train_model(clf, train_loader, optimizer, device, loss_clf, epochs, pred_prob=False, test_loader = None, 
                    save_dir = "../results",save_model="cifar_clf.pth"):
    """
        train network
    """
    
    clf.to(device)
    
    best_acc = 0.0
    for epoch in range(epochs):
        losses = AverageVarMeter()
        accs = AverageVarMeter()
        clf.train()
        for i,(x,y) in enumerate(train_loader):
            x,y = x.to(device),y.to(device)
            clf.zero_grad()
            out = clf(x)
            if pred_prob:
                out = torch.log(torch.clamp(out,min=SMALL))
            loss = loss_clf(out,y)
            loss.backward()
            optimizer.step()
            acc = accuracy(out.detach().cpu(),y.detach().cpu())
            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del acc, out,x,y,loss
            torch.cuda.empty_cache()
        
        if test_loader:
            _,acc_te,acc_std_te = test_model(clf,test_loader,loss_clf,device,0.0,save_dir,save_model,pred_prob)
            if best_acc<acc_te:
                best_acc = acc_te    
            logging.info(
                    "\nEpoch [{}/{}]\t"
                    "Loss {:.4f} ({:.4f})\t"
                    "Prec@1 train {:.3f} ({:.3f})\t"
                    "Prec@1 test {:.3f} ({:.3f})   \t".format(
                        (epoch+1),
                        epochs,
                        losses.avg.detach().cpu(),
                        losses.std.detach().cpu(),
                        accs.avg.detach().cpu(),
                        accs.std.detach().cpu(),
                        acc_te,
                        acc_std_te,                   
                    )
            )
        else:
            torch.save(clf.state_dict(),os.path.join(save_dir, save_model))
            logging.info(
                "\nEpoch [{}/{}]\t"
                "Loss {:.4f} ({:.4f})\t"
                "Prec@1 train {:.3f} ({:.3f})   \t".format(
                    (epoch+1),
                    epochs,
                    losses.avg.detach().cpu(),
                    losses.std.detach().cpu(),
                    accs.avg.detach().cpu(),
                    accs.std.detach().cpu(),             
                )
            )
    clf.cpu()
    return clf

def test_model(model, test_loader, criterion, device, best_acc=0.0,save_dir = "../results/",save_model = "ckpt.pth",pred_prob = False):
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            p_y = model(x)
            if pred_prob:
                p_y = torch.log(torch.clamp(p_y,min=SMALL))
            loss = criterion(p_y,y)

            acc = accuracy(p_y.detach().cpu(), y.detach().cpu())

            losses.update(loss,x.size(0))
            accs.update(acc[0],x.size(0))
            del x,y,p_y, loss, acc
            torch.cuda.empty_cache()
    #         print(acc)
        if accs.avg>=best_acc:
            torch.save(model.state_dict(),os.path.join(save_dir, save_model))
    return losses.avg.detach().cpu(), accs.avg.detach().cpu(),accs.std.detach().cpu()

def test_binary_model(model, test_loader, criterion, device, best_acc=0.0,save_dir = "../results/",save_model = "ckpt.pth",pred_prob = False):
    model.to(device)
    model.eval()
    losses = AverageVarMeter()
    accs = AverageVarMeter()
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)

            p_y = model(x)
            if pred_prob:
                p_y = torch.log(torch.clamp(p_y,min=SMALL))
            loss = criterion(p_y,y)

            acc = accuracy_b(p_y.detach().cpu(), y.detach().cpu())

            losses.update(loss,x.size(0))
            accs.update(torch.tensor(acc),x.size(0))
            del x,y,p_y, loss, acc
            torch.cuda.empty_cache()
    #         print(acc)
        if accs.avg>best_acc:
            torch.save(model.state_dict(),os.path.join(save_dir, save_model))
    return losses.avg.detach().cpu().item(), accs.avg

