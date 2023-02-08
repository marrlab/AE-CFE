import torch
import torch.nn as nn
#mmd [1 0 0]

def calculate_mmd(z,ds):
    zs = z.squeeze()
    source = zs[ds[:,0] == 1]
    target1 = zs[ds[:,1] == 1]
    target2 = zs[ds[:,2] == 1]

    loss1 = mmd(source, target1)
    loss2 = mmd(source, target2)

    return loss1, loss2


def mmd(source, target):
    mseloss = nn.MSELoss()
    kldiv = nn.KLDivLoss()
    

    source_mu = torch.mean(source, axis=0)
    target_mu = torch.mean(target, axis=0)
    mu_dis = mseloss(source_mu, target_mu)


    source_cov = torch.cov(source.T)
    target_cov = torch.cov(target.T)
    cov_loss = 0.5*(
        kldiv(torch.nn.LogSoftmax(dim=0)(source_cov), torch.nn.Softmax(dim=0)(target_cov)) +
        kldiv(torch.nn.LogSoftmax(dim=0)(target_cov), torch.nn.Softmax(dim=0)(source_cov)) )



    loss = mu_dis + cov_loss
    return loss