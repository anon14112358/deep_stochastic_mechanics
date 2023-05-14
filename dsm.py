import numpy as np 
import torch 
from torch import nn
from utils_physics import *
from tqdm import tqdm
from torch.autograd import grad


def get_jacobian(f, x):
    """Computes the Jacobian of f w.r.t x.

    This is according to the reverse mode autodiff rule,

    sum_i v^b_i dy^b_i / dx^b_j = sum_i x^b_j R_ji v^b_i,

    where:
    - b is the batch index from 0 to B - 1
    - i, j are the vector indices from 0 to N-1
    - v^b_i is a "test vector", which is set to 1 column-wise to obtain the correct
        column vectors out ot the above expression.

    :param f: function R^N -> R^N
    :param x: torch.tensor of shape [B, N]
    :return: Jacobian matrix (torch.tensor) of shape [B, N, N]
    """

    B, N = x.shape
    y = f(x)
    jacobian = list()
    for i in range(y.shape[-1]):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = grad(y,
                       x,
                       grad_outputs=v,
                       retain_graph=True,
                       create_graph=True,
                       allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)

    jacobian = torch.stack(jacobian, dim=2).requires_grad_()

    return jacobian


class NN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim) 
        self.relu1 = nn.Tanh()
        self.relu2 = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, hid_dim)  
        self.fc4 = nn.Linear(hid_dim, output_dim) 

    def forward(self, x, t):
        out = torch.hstack((x, t))
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc4(out)
        return out

def train_dsm(net_u, net_v, d, mu, sig2, device, optimizer, time_splits,
              h=1e-2, m=1, T=1,
              batch_size=10, n_iter=10, N=1000, iter_threshhold=4, 
              alpha_later=0.2, alpha_start=0.6, 
              use_Laplacian=False, x0_phase=0,
              alpha=0.8, gamma=1, beta=1):
    """Train DSM model
    Inputs:
    net_u: a neural net u
    net_v: a neural net v
    net_v: a neural net v
    batch_size: batch size
    n_iter: the number of epochs
    iter_threshhold: for how many epochs regenerate alpha_start fraction of batches
    alpha_later: fraction of batches to regenerate after  iter_threshhold
    alpha_start: fraction of batches to regenerate before  iter_threshhold
    N: number of time steps
    use_Laplacian: use the Laplacian operator or not

    Return:
    net_u, net_v                                 : trained models
    losses, losses_sm, losses_newton, losses_init: losses per epoch
    """
    criterion = nn.MSELoss()
    losses = []
    losses_sm = []
    losses_newton = []
    losses_init = []

    # start = time.time()
    with tqdm(range(n_iter), unit="iter") as tepoch:
        for tau in tepoch:
            if tau == 0: 
                l_sm = 0
                l_nl = 0
                X_0 = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu, 
                                                                (sig2**2)*np.eye(d), batch_size)).to(device)
                X_0.requires_grad = True
                u0_val = torch.Tensor(u_0(X_0, sig2)).to(device)
                v0_val = torch.Tensor(v_0(X_0, x0_phase)).to(device)
                optimizer.zero_grad()
                eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), batch_size) for i in range(N+1)]
                for i in range(1, N+1): # iterate over time steps
                    t_prev = time_splits[i-1].clone()
                    if i == 1:
                        X_prev = X_0.clone().to(device) # X_{i-1}
                    else:
                        X_prev = X_i.clone().to(device) # X_{i-1}

                    t_i = time_splits[i].clone()
                    
                    t_prev_batch = time_transform(t_prev).expand(batch_size, 1).to(device).clone()
                    X_i = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                    net_v(X_prev, t_prev_batch)) \
                            + torch.Tensor(np.sqrt(h*T/(m*N)) * eps[i-1]).to(device)
    
                    if i > 1: 
                        Xs = torch.hstack((Xs, X_i))
                    else:
                        Xs = X_i.clone()
    
                X_0_iter0 = X_0.clone() # collect X_0 from the initial iter
                Xs = torch.concat((X_0_iter0, Xs), axis=1)
            elif tau > iter_threshhold: 
                BATCH_size = int(alpha_later * batch_size) 
                
                X_0_iter = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu, 
                                                                (sig2**2)*np.eye(d), BATCH_size)).to(device)
                X_0_iter.requires_grad = True
                u0_val = torch.Tensor(u_0(X_0_iter, sig2)).to(device)
                v0_val = torch.Tensor(v_0(X_0_iter, x0_phase)).to(device)
                optimizer.zero_grad()
                eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), BATCH_size) for i in range(N+1)]
                # get one trajectory
                for i in range(1, N+1): 
                    t_prev = time_splits[i-1].clone()
                    if i == 1:
                        X_prev = X_0_iter.clone().to(device) 
                    else:
                        X_prev = X_i_iter.clone().to(device) 

                    t_i = time_splits[i].clone()
                    
                    t_prev_batch = time_transform(t_prev).expand(BATCH_size, 1).to(device)
                    X_i_iter = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                    net_v(X_prev, t_prev_batch)) \
                            + torch.Tensor(np.sqrt(h*T/(m*N)) * eps[i-1]).to(device)
    
                    if i > 1: 
                        Xs_iter = torch.hstack((Xs_iter, X_i_iter))
                    else:
                        Xs_iter = X_i_iter.clone()
                Xs_iter = torch.concat((X_0_iter, Xs_iter), axis=1)
                
                # replace the old batch with the new one
                Xs_all = Xs_all.reshape((batch_size, (N+1)*d))
                X_0_all = X_0_all.reshape((batch_size, d))
                Xs_all = torch.roll(Xs_all, -BATCH_size, 0)
                X_0_all = torch.roll(X_0_all, -BATCH_size, 0)
                Xs_all = torch.vstack((Xs_all[BATCH_size:].detach(), Xs_iter.detach()) )
                Xs_all.requires_grad = True

                X_0_all = torch.vstack((X_0_all[BATCH_size:].detach(), X_0_iter.detach()) )
                X_0_all.requires_grad = True 
                
                Xs_all = Xs_all.reshape(batch_size*(N+1), d)
                X_0_all = X_0_all.reshape(batch_size, d)
                
                time_splits_batches = time_splits.repeat(batch_size).reshape(batch_size, N+1).reshape(batch_size*(N+1), 1).to(device)
                time_splits_batches.requires_grad = True
                
                out_u = net_u(Xs_all, time_splits_batches)
                out_v = net_v(Xs_all, time_splits_batches)
                du_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1 

                    dudt = grad(out_u, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    du_dt[:, i] = dudt[:, 0]  
                    
                dv_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1 

                    dvdt = grad(out_v, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    dv_dt[:, i] = dvdt[:, 0]
                
                d_norm = torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_u(x, time_splits_batches), Xs_all), 
                                    net_u(Xs_all, time_splits_batches)) - \
                        torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_v(x, time_splits_batches), Xs_all), 
                                    net_v(Xs_all, time_splits_batches))
        
                dv_ddx = get_jacobian(lambda x: torch.einsum("jii", 
                                                    get_jacobian(lambda x: 
                                                                 net_v(x, time_splits_batches), 
                                                                 x))[:, None], Xs_all)[:, :, 0]
   
                if not use_Laplacian:
                    du_ddx = get_jacobian(lambda x: torch.einsum("jii", 
                                                        get_jacobian(lambda x: 
                                                                     net_u(x, time_splits_batches), 
                                                                     x))[:, None], Xs_all)[:, :, 0]
                else: 
                    du_ddx = torch.zeros_like(dv_ddx)
                    for j in range(d):
                        du_ddx[:, j] = torch.einsum("jii", get_jacobian(lambda x: 
                                                    get_jacobian(lambda x: net_u(x, time_splits_batches)[:, j][:, None], x)[:, :, 0], Xs_all))
                
                out_uv = net_v(Xs_all, time_splits_batches) * net_u(Xs_all, time_splits_batches)
                dvu_dx = grad(out_uv, Xs_all, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]
                L_sm = criterion(du_dt, -(h/(2*m)) * dv_ddx - dvu_dx) #/ N

                L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx \
                                -V_x_i(Xs_all).to(device)/m) #/ N 
                
                u0_val = torch.Tensor(u_0(X_0_all, sig2)).to(device)
                v0_val = torch.Tensor(v_0(X_0_all, x0_phase)).to(device)
                L_ic = criterion(net_u(X_0_all, 
                                    time_splits[0].expand(batch_size, 1).to(device)), u0_val) \
                        + criterion(net_v(X_0_all, 
                                        time_splits[0].expand(batch_size, 1).to(device)), v0_val)
                
                loss = (alpha * L_sm + beta * L_nl + gamma * L_ic) / 3.0
                losses.append(loss.item())
                losses_newton.append(L_nl.item())
                losses_sm.append(L_sm.item())
                losses_init.append(L_ic.item())
                tepoch.set_postfix(loss_iter=loss.item(), loss_mean=np.mean(losses[-10:]), loss_std=np.std(losses[-10:]))

                loss.backward()
                optimizer.step()
                
            elif tau <= iter_threshhold:
                BATCH_size = int(alpha_start * batch_size)

                X_0_iter = torch.Tensor(np.random.multivariate_normal(np.ones(d) * mu, 
                                                                (sig2**2)*np.eye(d), BATCH_size)).to(device)
                X_0_iter.requires_grad = True
                u0_val = torch.Tensor(u_0(X_0_iter, sig2)).to(device)
                v0_val = torch.Tensor(v_0(X_0_iter, x0_phase)).to(device)
                optimizer.zero_grad()
                eps = [np.random.multivariate_normal(np.zeros(d), np.eye(d), BATCH_size) for i in range(N+1)]
                # get one trajectory
                for i in range(1, N+1): 
                    t_prev = time_splits[i-1].clone()
                    if i == 1:
                        X_prev = X_0_iter.clone().to(device) 
                    else:
                        X_prev = X_i_iter.clone().to(device) 

                    t_i = time_splits[i].clone()

                    t_prev_batch = time_transform(t_prev).expand(BATCH_size, 1).to(device)
                    X_i_iter = torch.Tensor(X_prev).to(device) + T / N * (net_u(X_prev, t_prev_batch) + \
                                                                    net_v(X_prev, t_prev_batch)) \
                            + torch.Tensor(np.sqrt(h*T/(m*N)) * eps[i-1]).to(device)

                    if i > 1: 
                        Xs_iter = torch.hstack((Xs_iter, X_i_iter))
                    else:
                        Xs_iter = X_i_iter.clone()
                        
                Xs_iter = torch.concat((X_0_iter, Xs_iter), axis=1)
                
                # replace the old batch with the new one
                if tau == 1:
                    Xs_all = torch.vstack((Xs[BATCH_size:].detach(), Xs_iter.detach()) )
                    Xs_all.requires_grad = True

                    X_0_all = torch.vstack((X_0_iter0[BATCH_size:].detach(), X_0_iter.detach()) )
                    X_0_all.requires_grad = True
                else:
                    Xs_all = Xs_all.reshape((batch_size, (N+1)*d))
                    X_0_all = X_0_all.reshape((batch_size, d))
                    Xs_all = torch.roll(Xs_all, -BATCH_size, 0)
                    X_0_all = torch.roll(X_0_all, -BATCH_size, 0)
                    Xs_all = torch.vstack((Xs_all[BATCH_size:].detach(), Xs_iter.detach()) )
                    Xs_all.requires_grad = True

                    X_0_all = torch.vstack((X_0_all[BATCH_size:].detach(), X_0_iter.detach()) )
                    X_0_all.requires_grad = True

                Xs_all = Xs_all.reshape(batch_size*(N+1), d)
                X_0_all = X_0_all.reshape(batch_size, d)

                time_splits_batches = time_splits.repeat(batch_size).reshape(batch_size, N+1).reshape(batch_size*(N+1), 1).to(device)
                time_splits_batches.requires_grad = True

                out_u = net_u(Xs_all, time_splits_batches)
                out_v = net_v(Xs_all, time_splits_batches)
                du_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1 

                    dudt = grad(out_u, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    du_dt[:, i] = dudt[:, 0]  
                    
                dv_dt = torch.zeros((batch_size*(N+1), d)).to(device)
                for i in range(d):
                    vector = torch.zeros_like(out_u)
                    vector[:, i] = 1 

                    dvdt = grad(out_v, time_splits_batches, grad_outputs=vector, create_graph=True)[0]
                    dv_dt[:, i] = dvdt[:, 0]
                    
                d_norm = torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_u(x, time_splits_batches), Xs_all), 
                                    net_u(Xs_all, time_splits_batches)) - \
                        torch.einsum("ijk,ik->ij", get_jacobian(lambda x: net_v(x, time_splits_batches), Xs_all), 
                                    net_v(Xs_all, time_splits_batches))
        
                dv_ddx = get_jacobian(lambda x: torch.einsum("jii", 
                                                    get_jacobian(lambda x: net_v(x, time_splits_batches), x))[:, None], 
                                    Xs_all)[:, :, 0]
            
                if not use_Laplacian:
                    du_ddx = get_jacobian(lambda x: torch.einsum("jii", 
                                                        get_jacobian(lambda x: 
                                                                     net_u(x, time_splits_batches), 
                                                                     x))[:, None], Xs_all)[:, :, 0]
                else: 
                    du_ddx = torch.zeros_like(dv_ddx)
                    for j in range(d):
                        du_ddx[:, j] = torch.einsum("jii", get_jacobian(lambda x: 
                                                    get_jacobian(lambda x: net_u(x, time_splits_batches)[:, j][:, None], x)[:, :, 0], Xs_all))
                
            
                out_uv = net_v(Xs_all, time_splits_batches) * net_u(Xs_all, time_splits_batches)
                dvu_dx = grad(out_uv, Xs_all, grad_outputs=torch.ones_like(out_uv), create_graph=True)[0]
                
                L_sm = criterion(du_dt, -(h/(2*m)) * dv_ddx - dvu_dx) #/N

                L_nl = criterion(dv_dt, d_norm + (h / 2 / m) * du_ddx - V_x_i(Xs_all).to(device)/m) #/N

                u0_val = torch.Tensor(u_0(X_0_all, sig2)).to(device)
                v0_val = torch.Tensor(v_0(X_0_all, x0_phase)).to(device)
                L_ic = criterion(net_u(X_0_all, 
                                    time_splits[0].expand(batch_size, 1).to(device)), u0_val) \
                        + criterion(net_v(X_0_all, 
                                        time_splits[0].expand(batch_size, 1).to(device)), v0_val)

                loss = (alpha * L_sm + beta * L_nl + gamma * L_ic) / 3.0
                losses.append(loss.item())
                losses_newton.append(L_nl.item())
                losses_sm.append(L_sm.item())
                losses_init.append(L_ic.item())
                tepoch.set_postfix(loss_iter=loss.item(), loss_mean=np.mean(losses[-10:]), loss_std=np.std(losses[-10:]))
                
                loss.backward(retain_graph=True)
                optimizer.step()
            else:
                print('NO CHOICE')
                
    # end = time.time()
    # time_run = end - start
    return net_u, net_v, losses, losses_sm, losses_newton, losses_init

