import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import SubsetRandomSampler
import copy

def approx_invsig(deg):
    x_int = np.arange(0.001,1.0,0.001)
    y_int = np.log(x_int)-np.log(1-x_int)
    A = np.zeros((len(x_int), deg+1))
    A[:,0]=1
    for i in range(deg):
        A[:,i+1]=x_int**(i+1)
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y_int)

def approx_sig(deg, gamma):
    x_int = np.arange(-4,4.01,0.01)*gamma
    y_int = 1/(1+np.exp(-1*x_int))
    A = np.zeros((len(x_int), deg+1))
    A[:,0]=1
    for i in range(deg):
        A[:,i+1]=x_int**(i+1)
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(y_int)

def deceptive_perturb(score, beta, gamma, logit=False):
    if logit:
        score = nn.Softmax(1)(score)
    rs = torch.log(score)-torch.log(1-score)
    rs *= gamma
    rs = 1/(1+torch.exp(-1*rs))
    rs -= 1/2
    rs *= beta
    rs = score-rs
    sum_rs = rs.sum(1)
    rs/=sum_rs.reshape(-1,1)
    return rs

def deceptive_perturb_he(score, beta, gamma, t1, t2):
    appr_invsig = approx_invsig(t1)
    rs = torch.zeros_like(score)
    for i in range(t1+1):
        rs+=appr_invsig[i]*(score**i)
    rs*=gamma
    appr_sig = approx_sig(t2, gamma)
    rs2 = torch.zeros_like(rs)
    for i in range(t2+1):
        rs2+=appr_sig[i]*(rs**i)
    rs2 -= 1/2
    rs2 *= beta
    rs2 = score-rs2
    sum_rs = rs2.sum(1)
    rs2/=sum_rs.reshape(-1,1)
    return rs2
    
def inv_apprx(x, t, m):
    a = 2-2/m*x
    b = 1-2/m*x
    for i in range(t):
        b = b*b
        a = a*(1+b)
    return 2/m*a

def comp_max_tau(output, tau, t1, t2):
    device = output.device
    res = copy.copy(output)
    
    res = torch.cat((res, tau*torch.ones(len(res)).view(-1,1).to(device)), 1)
    for i in range(t1):
        res = res*res
        sum_res = res.sum(1)
        if i==0:
            inv = inv_apprx(sum_res, t2, 2+tau*tau)
        else:
            inv = inv_apprx(sum_res, t2, 2)
        res *= inv.reshape(-1,1)
    return res[:,-1]

def apply_taylor_softmax(x,emph=1.0):
    x = (1+x+0.5*x**2)**emph                                   
    x /= torch.sum(x,axis=1).view(-1,1)
    return x

class Net_deceptive_perturb(nn.Module):
    def __init__(self, model, beta, gamma):
        super(Net_deceptive_perturb, self).__init__()
        self.model = model
        self.beta = beta
        self.gamma = gamma
    def forward(self,x):
        x = self.model(x)
        x = deceptive_perturb(x, self.beta, self.gamma)
        return x
        
class Net_deceptive_perturb_HE(nn.Module):
    def __init__(self, model, beta, gamma, t1=3,t2=3):
        super(Net_deceptive_perturb_HE, self).__init__()
        self.model = model
        self.beta = beta
        self.gamma = gamma
        self.t1 = t1
        self.t2 = t2
    def forward(self,x):
        x = self.model(x)
        x = deceptive_perturb_he(x, self.beta, self.gamma, self.t1, self.t2)
        return x        
class Net_tsoftmax(nn.Module):
    def __init__(self, model,temp = 1000.0):
        super(Net_tsoftmax, self).__init__()
        self.model = model
        self.temp = temp
    def forward(self,x):
        x = self.model(x)
        x = apply_taylor_softmax(x/self.temp)
        return x
    
class Net_softmax(nn.Module):
    def __init__(self, model):
        super(Net_softmax, self).__init__()
        self.model = model
    def forward(self,x):
        x = self.model(x)
        x = F.softmax(x,dim=1)
        return x

class Net_logsoftmax(nn.Module):
    def __init__(self, model):
        super(Net_logsoftmax, self).__init__()
        self.model = model
    def forward(self,x):
        x = self.model(x)
        x = F.log_softmax(x,dim=1)
        return x

class Net_log(nn.Module):
    def __init__(self, model):
        super(Net_log, self).__init__()
        self.model = model
    def forward(self,x):
        x = self.model(x)
        x = torch.log(x)
        return x


class CombNet(nn.Module):
    def __init__(self, net_orig, net_fake, tau=0.5):
        super(CombNet, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        cond_in = torch.max(x1, dim=1).values>self.tau
        out = (x1*cond_in.view(-1,1)+x2*(~cond_in.view(-1,1)))
        return out
    
class CombNet_logit(nn.Module):
    def __init__(self, net_orig, net_fake, tau=0.5):
        super(CombNet_logit, self).__init__()
        self.net_orig = Net_softmax(net_orig)
        self.net_fake = Net_softmax(net_fake)
        self.tau = tau
        
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        cond_in = torch.max(x1, dim=1).values>self.tau
        out = (x1*cond_in.view(-1,1)+x2*(~cond_in.view(-1,1)))
        return out

class CombNetHE(nn.Module):
    def __init__(self, net_orig, net_fake, tau=0.5, t1=3, t2=3):
        super(CombNetHE, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        self.t1 = t1
        self.t2 = t2
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        #cond_in = torch.max(x1, dim=1).values>self.tau
        cond_in = comp_max_tau(x1, self.tau, self.t1, self.t2)
        out = (x1*(1-cond_in.view(-1,1))+x2*cond_in.view(-1,1))
        return out
    
class CombNet_smooth(nn.Module): 
    def __init__(self, net_orig, net_fake, tau=0.5,nu=10):
        super(CombNet_smooth, self).__init__()
        self.net_orig = net_orig
        self.net_fake = net_fake
        self.tau = tau
        self.nu = nu
        
    def forward(self, x):
        x1 = self.net_orig(x)
        x2 = self.net_fake(x)
        cond_in = torch.sigmoid(self.nu*(self.tau-torch.max(x1, dim=1).values))
        out = (x1*(1-cond_in.view(-1,1))+x2*(cond_in.view(-1,1)))
        return out

