import torch

def get_gradient(model, x):
    model.zero_grad()
    y = model(x.reshape(1, *x.shape)).mean()
    y.backward()
    return torch.cat([param.grad.flatten() for param in model.parameters()])

def get_ntk(model, x):
    shape = x.shape
    feats = torch.stack([
        get_gradient(model, _x).detach().cpu().clone() for _x in x
    ], dim=0)
    return feats.reshape(shape[0], -1).detach().clone()
