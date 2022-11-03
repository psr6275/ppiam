import torch 
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(con_mat, labels, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'), normalize=False):
    plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    marks = np.arange(len(labels))
    nlabels = []
    for k in range(len(con_mat)):
        n = sum(con_mat[k])
        nlabel = '{0}(n={1})'.format(labels[k],n)
        nlabels.append(nlabel)
    plt.xticks(marks, labels)
    plt.yticks(marks, nlabels)

    thresh = con_mat.max() / 2.
    if normalize:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    else:
        for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
            plt.text(j, i, con_mat[i, j], horizontalalignment="center", color="white" if con_mat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def hist_multiple_dataloader(data_loaders, model, device, filename, plt_title = None,bins=10, title=True):
    model.to(device).eval()
    fig, axs = plt.subplots(len(data_loaders),2, sharex=True)
    
    for i, dloader in enumerate(data_loaders):
        cdloader = copy_dataloader_noshuffle(dloader)
        preds = get_prediction(model, cdloader, device)
        max_preds = torch.max(preds,axis=1)    
        
        axs[i,0].hist(max_preds[0].numpy(), bins = bins, rwidth=0.9)
        axs[i,1].hist(max_preds[1].numpy(), bins = bins, rwidth=0.9)
    if title:
        axs[0,0].set_title("maximum prediction value")
        axs[0,1].set_title("prediction class")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    model.cpu()

    


def maxclass_hist(data_loader, model, device, filename, plt_title = None,bins=10, 
                  return_val = False, clipping =False, clip_vals = [0.0,1.0]):
    model.to(device).eval()
    with torch.no_grad():
        for i, data in enumerate(data_loader,0):
            x = data[0].to(device)
            outs = model(x)
            if i==0:
                max_vals = outs.cpu().detach().numpy()
            else:
                max_vals = np.vstack((max_vals, outs.cpu().detach().numpy()))
            del data, outs,x
    max_vals = np.max(max_vals,axis=1)
    if clipping:
        max_vals = np.clip(max_vals, clip_vals[0], clip_vals[1])
    if plt_title:
        plt.title(plt_title)
    plt.hist(max_vals, bins=bins, rwidth=0.9)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    model.cpu()
    if return_val:
        return max_vals 

def prediction_hist(data_loader, model, device, filename, plt_title = None,bins=10):
    model.to(device).eval()
    max_idxs = torch.tensor([])
    with torch.no_grad():
        for data in data_loader:
            outs = model(data[0].to(device))
            _,idxs = outs.cpu().detach().max(axis=1)
            max_idxs = torch.cat((max_idxs,idxs))
            del data, outs,idxs
    
    if plt_title:
        plt.title(plt_title)
    plt.hist(max_idxs.numpy().flatten(), bins=bins, rwidth=0.9)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    model.cpu()

def maxclass_hist_comb(data_loader, comb_model, device, filename, plt_title = None,bins=None, 
                  return_val = False, clipping =False, clip_vals = [0.0,1.0]):
    comb_model.to(device).eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader,0):
            x = data[0].to(device)
            outs = comb_model(x)
            outs_f = (torch.max(comb_model.net_orig(x).cpu().detach(),dim=1).values<=comb_model.tau).numpy().flatten()
            if i==0:
                max_vals = outs.cpu().detach().numpy()
                f_idxs = outs_f
            else:
                max_vals = np.vstack((max_vals, outs.cpu().detach().numpy()))
#                 print(f_idxs.shape, outs_f.shape)
                f_idxs = np.hstack((f_idxs,outs_f))
                
            del data,x,outs, outs_f
    max_vals = np.max(max_vals,axis=1)
    if clipping:
        max_vals = np.clip(max_vals, clip_vals[0], clip_vals[1])
    if plt_title:
        plt.title(plt_title)
    if bins==None:
        bins = [0.1*i for i in range(11)]
    plt.hist(max_vals, bins=bins,label="combined network", rwidth=0.9)
    plt.hist(max_vals[f_idxs], bins=bins, label="fake network", rwidth=0.9)
    plt.legend(loc ='upper left')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    comb_model.cpu()
    if return_val:
        return max_vals,f_idxs     

def prediction_hist_comb(data_loader, comb_model, device, filename=None, plt_title = None,bins=None, return_val = False):
    comb_model.to(device).eval()
    max_idxs = torch.tensor([])
    f_idxs = np.array([])
    with torch.no_grad():
        for data in data_loader:
            x = data[0].to(device)
            outs = comb_model(x)
            outs_f = (torch.max(comb_model.net_orig(x).cpu().detach(),dim=1).values<=comb_model.tau).numpy().flatten()
            _,idxs = outs.cpu().detach().max(axis=1)
            max_idxs = torch.cat((max_idxs,idxs))
            f_idxs = np.concatenate((f_idxs, outs_f))
            del data, outs,idxs, outs_f
    if bins==None:
        bins = [i for i in range(11)]
    f_idxs = f_idxs.astype(int)
    print("maxidxs shape: ",max_idxs.shape)
    print("fidxs shape: ",f_idxs.shape)
    print("fidxs sum: ", np.sum(f_idxs))
    if plt_title:
        plt.title(plt_title)
    plt.hist(max_idxs.numpy().flatten(), bins=bins, label="combined network", rwidth=0.9)
    plt.hist(max_idxs.numpy().flatten()[f_idxs==1], bins=bins, label="fake network", rwidth=0.9)
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    comb_model.cpu()
    if return_val:
        return max_idxs, f_idxs

def prediction_count_comb(data_loader, comb_model, device, plt_title = None,bins=None, return_val = False):
    comb_model.to(device).eval()
    print(comb_model.tau)
    max_idxs = torch.tensor([])
    f_idxs = np.array([])
    with torch.no_grad():
        for data in data_loader:
            x = data[0].to(device)
            outs = comb_model(x)
            outs_f = (torch.max(comb_model.net_orig(x).cpu().detach(),dim=1).values<=comb_model.tau).numpy().flatten()
            _,idxs = outs.cpu().detach().max(axis=1)
            max_idxs = torch.cat((max_idxs,idxs))
            f_idxs = np.concatenate((f_idxs, outs_f))
            del data, outs,idxs, outs_f
    if bins==None:
        bins = [i for i in range(11)]
    f_idxs = f_idxs.astype(int)
    print("maxidxs shape: ",max_idxs.shape)
    print("fidxs shape: ",f_idxs.shape)
    print("fidxs sum: ", np.sum(f_idxs))
    if plt_title:
        plt.title(plt_title)
    max_idxs = max_idxs.numpy().flatten().astype(int)
    plt.plot(np.arange(10), np.bincount(max_idxs), label="combined network", color='blue')
    plt.plot(np.arange(10), np.bincount(max_idxs[f_idxs==1]), label="fake network", color='red')
    #plt.hist(max_idxs.numpy().flatten(), bins=bins, label="combined network", rwidth=0.9)
    #plt.hist(max_idxs.numpy().flatten()[f_idxs==1], bins=bins, label="fake network", rwidth=0.9)
    plt.show()
    comb_model.cpu()
    if return_val:
        return max_idxs, f_idxs

def maxclass_hist_comb_by_class(data_loader, comb_model, device, bins=None, 
                  return_val = False, clipping =False, clip_vals = [0.0,1.0]):
    comb_model.to(device).eval()

    with torch.no_grad():
        for i, data in enumerate(data_loader,0):
            x = data[0].to(device)
            outs = comb_model(x)
            outs_f = (torch.max(comb_model.net_orig(x).cpu().detach(),dim=1).values<=comb_model.tau).numpy().flatten()
            if i==0:
                max_vals = outs.cpu().detach().numpy()
                f_idxs = outs_f
            else:
                max_vals = np.vstack((max_vals, outs.cpu().detach().numpy()))
#                 print(f_idxs.shape, outs_f.shape)
                f_idxs = np.hstack((f_idxs,outs_f))
                
            del data,x,outs, outs_f
    max_ind = np.argmax(max_vals, axis=1)
    max_vals = np.max(max_vals,axis=1)
    if clipping:
        max_vals = np.clip(max_vals, clip_vals[0], clip_vals[1])

    if bins==None:
        bins = [0.1*i for i in range(11)]
    
   
    for i in range(10):
        #print(max_ind==i)
        #print(f_idxs)
        plt.hist(max_vals[max_ind==i], bins=bins,label="combined network", rwidth=0.9)
        plt.hist(max_vals[(max_ind==i)&(f_idxs)], bins=bins, label="fake network", rwidth=0.9)
        plt.legend(loc ='upper left')
        plt.show()
    comb_model.cpu()
    if return_val:
        return max_vals,f_idxs     

def test_net(net, loader, device):
    net = net.to(device).eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.3f %%' % (
        100 * correct / total))
    net = net.cpu()
    return 100 * correct / total

def test_net_by_class(net, loader, device):
    net = net.to(device)
    correct = torch.zeros(10)
    total = torch.zeros(10)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            for i in range(10):
                total[i]+=labels[labels==i].size(0)
                correct[i]+=(predicted==labels)[labels==i].sum().item()
    net = net.cpu()
    return 100 * correct / total

def get_prediction(model, testloader, device):
    model.to(device).eval()
    preds = []
    with torch.no_grad():
        for data in testloader:
            preds.append(model(data[0].to(device)).detach().cpu())
        preds = torch.cat(preds,dim=0)
    return preds

def test_fakenet_ratio(model,testloader,tau,device,combnet=True):
    if combnet:
        net = model.net_orig
    else:
        net = model
    net.to(device).eval()
    ssc = 0.0
    ss = 0.0
    with torch.no_grad():
        for x,_ in testloader:
            out = net(x.to(device)).detach().cpu()
#             print(torch.max(out,1))
#             print(torch.sum(torch.max(out,1)[0]>tau))
            ssc += torch.sum(torch.max(out,1)[0]<tau).item()
            ss += len(x)
            del x, out
    net.cpu()
    return (ssc/ss)*100
    
class AverageVarMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0
        self.sum2 = 0
        self.std = 0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val = val
#         print(val)
        self.sum2 += (val**2)*n
        self.sum +=val*n
        self.count +=n
        self.avg = self.sum / self.count
        self.std = torch.sqrt(self.sum2/self.count-self.avg**2)



def accuracy(output, target, topk=(1,)):
    '''Compute the top1 and top k accuracy'''
    maxk = max(topk)
    batch_size = target.size(0)
    _,pred = output.topk(maxk,1,True,True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res

def accuracy_b(output, target, thres = 0.5):
    '''Compute the binary accuracy'''
    assert output.ndim == 1 and target.size() == output.size()
    y_prob = output>thres 
    return (target == y_prob).sum().item() / target.size(0)

def correspondence_score(output1, output2):
    _, pred1 = torch.max(output1.data, dim=1)
    _, pred2 = torch.max(output2.data, dim=1)
    correct = pred1.eq(pred2)
    return 100*correct.view(-1).float().sum(0)/pred1.size(0)

def _is_shuffle(dloader):
    return type(dloader.sampler) == torch.utils.data.sampler.RandomSampler

def copy_dataloader_noshuffle(dloader):
#     dloader_ns = copy.deepcopy(dloader)
    from torch.utils.data.dataset import Subset
    if type(dloader.dataset) == Subset:
        dset = copy.deepcopy(Subset(dloader.dataset.dataset,dloader.dataset.indices))
    else:
        dset = copy.deepcopy(dloader.dataset)
    return torch.utils.data.DataLoader(dset, batch_size = dloader.batch_size, sampler = dloader.sampler)

def obtain_confusion_matrix(clf, dloader,device, num_classes=10):
    from torchmetrics import ConfusionMatrix
    if _is_shuffle(dloader):
        dloader = copy_dataloader_noshuffle(dloader)
    if type(dloader.dataset) == torch.utils.data.dataset.Subset:
        sidx = dloader.dataset.indices
        target = torch.tensor(dloader.dataset.dataset.targets)[sidx]
    else:
        target = torch.tensor(dloader.dataset.targets)
    print(target)
    preds = get_prediction(clf, dloader, device)
    confmat = ConfusionMatrix(num_classes=num_classes)
    return confmat(preds, target)

def split_by_tau(comb, dloader, device):
    from torch.utils.data import Subset
    if type(dloader.sampler) == torch.utils.data.sampler.RandomSampler:
        cdloader = copy_dataloader_noshuffle(dloader)
    else:
        cdloader = dloader
    preds = get_prediction(comb.net_orig, cdloader, device)
    predm = torch.max(preds,axis=1)
    idxlist = torch.arange(len(predm[0]))
    tidx = list(idxlist[predm[0]>comb.tau].numpy())
    fidx = list(idxlist[predm[0]<=comb.tau].numpy())

    if type(dloader.dataset) == torch.utils.data.dataset.Subset:
        dset = cdloader.dataset.dataset
        sidx = np.array(cdloader.dataset.indices)
        dset_target = Subset(dset, list(sidx[tidx]))
        dset_fake = Subset(dset, list(sidx[fidx]))
    else:
        dset = cdloader.dataset        
        dset_target = Subset(dset, tidx)
        dset_fake = Subset(dset, fidx)
    dloader_target = torch.utils.data.DataLoader(dset_target, batch_size = dloader.batch_size, shuffle = False)
    dloader_fake = torch.utils.data.DataLoader(dset_fake, batch_size = dloader.batch_size, shuffle = False)
    

    return dloader_target, dloader_fake    