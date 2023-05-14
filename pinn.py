import numpy as np
import torch
import os
from scipy import sparse
from scipy import integrate
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from pyDOE import lhs
from tqdm import tqdm
from torch import nn
from torch.autograd import grad


class NN_pinn(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super(NN_pinn, self).__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim) 
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(hid_dim, hid_dim) 
        self.relu2 = nn.Tanh()
        self.fc3 = nn.Linear(hid_dim, output_dim) 

    def forward(self, x):
#         out = torch.hstack((x, t))
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    

def psi_0(x, x0=0.0, sigma=np.sqrt(0.1), x0_phase=0):
    A = 1.0 / (sigma * np.sqrt(2*np.pi)) 
    return np.sqrt(A) * np.exp(-(x-x0)**2 / (4.0 * (sigma**2))) * np.exp(1j * x0_phase * x)


# potential for the PINN loss
def potential_V(x):
    return 0.5*(x - 0.1)**2

def train_pinn(model, n_epochs, f_colloc, b_colloc, ic_colloc, ic, device,
               lambda_data=0, lambda_pde=1, sample_idx=False, h=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_list = np.zeros(n_epochs)
    loss_physics_list = np.zeros(n_epochs)
    loss_b_list = np.zeros(n_epochs)
    loss_ic_list = np.zeros(n_epochs)
    loss_regr_list = np.zeros(n_epochs)

    with tqdm(range(n_epochs), unit="epoch") as tepoch:
        for i in tepoch:
            optimizer.zero_grad()
            N_subs_regr = 20000
            N_subs_colloc = 20000
            if lambda_data > 0:
                if sample_idx:
                    idx_subsample = np.random.choice(train_x.shape[0], N_subs_regr, replace=False)
                    train_x_sub = train_x[idx_subsample]
                    train_y_sub = train_y[idx_subsample]
                    y_pred = model(train_x_sub.to(device))
                    loss1 = torch.mean((y_pred - train_y_sub.to(device))**2)
                else:
                    y_pred = model(train_x.to(device))
            else:
                loss1 = torch.tensor(0) 
                
            # calculate loss on colloc points
            if sample_idx:
                idx_subsample = np.random.choice(f_colloc.shape[0], N_subs_colloc, replace=False)
                f_colloc_sub = f_colloc[idx_subsample]
                f_colloc_sub = f_colloc_sub.to(device)
                y_pred = model(f_colloc_sub)
                u = y_pred[:, 0]
                v = y_pred[:, 1]
                grad_u = torch.autograd.grad(u, f_colloc_sub, torch.ones_like(u), create_graph=True)[0]
                grad_v = torch.autograd.grad(v, f_colloc_sub, torch.ones_like(v), create_graph=True)[0] 
                du_dt = grad_u[:, [1]]
                dv_dt = grad_v[:, [1]]
                du_dx = grad_u[:, [0]]
                dv_dx = grad_v[:, [0]]
                du_dxx = torch.autograd.grad(du_dx, f_colloc_sub, torch.ones_like(du_dx), create_graph=True)[0][:, [0]]
                dv_dxx = torch.autograd.grad(dv_dx, f_colloc_sub, torch.ones_like(dv_dx), create_graph=True)[0][:, [0]]

                loss_u = -h*dv_dt + (h**2 / 2) * du_dxx - potential_V(f_colloc_sub[:, 0])[:, None] * u.view(-1, 1)
#                 loss_u = h*dv_dt - (h**2 / 2) * du_dxx - potential_V(f_colloc_sub[:, 0])[:, None] * u.view(-1, 1)
                loss_v = h*du_dt + (h**2 / 2) * dv_dxx - potential_V(f_colloc_sub[:, 0])[:, None] * v.view(-1, 1)
#                 loss_v = h*du_dt + (h**2 / 2) * dv_dxx + potential_V(f_colloc_sub[:, 0])[:, None] * v.view(-1, 1)
            else:
                f_colloc = f_colloc.to(device) # to(device)
                y_pred = model(f_colloc)
                u = y_pred[:, 0]
                v = y_pred[:, 1]

                grad_u = torch.autograd.grad(u, f_colloc, torch.ones_like(u), create_graph=True)[0]
                grad_v = torch.autograd.grad(v, f_colloc, torch.ones_like(v), create_graph=True)[0] 
                du_dt = grad_u[:, [1]]
                dv_dt = grad_v[:, [1]]
                du_dx = grad_u[:, [0]]
                dv_dx = grad_v[:, [0]]
                du_dxx = torch.autograd.grad(du_dx, f_colloc, torch.ones_like(du_dx), create_graph=True)[0][:, [0]]
                dv_dxx = torch.autograd.grad(dv_dx, f_colloc, torch.ones_like(dv_dx), create_graph=True)[0][:, [0]]
                loss_u = -h*dv_dt + (h**2 / 2) * du_dxx - potential_V(f_colloc[:, 0])[:, None] * u.view(-1, 1)
                loss_v = h*du_dt + (h**2 / 2) * dv_dxx - potential_V(f_colloc[:, 0])[:, None] * v .view(-1, 1)          

            loss_physics = (loss_u**2 + loss_v**2).mean() #torch.concatenate((loss_u, loss_v), axis=1)
            y_pred_b = model(b_colloc.to(device))
            y_pred_ic = model(ic_colloc.to(device))
    
            loss_b = torch.mean(y_pred_b**2)
            loss_ic = torch.mean((y_pred_ic - ic.to(device))**2)

            loss2 = loss_physics + loss_b + loss_ic #(torch.mean(loss_physics**2) + loss_b + loss_ic)

            loss = lambda_data * loss1 + lambda_pde * loss2 # add two loss terms together
            loss_list[i] = loss.detach().cpu().numpy()
            loss_regr_list[i] = loss1.item()
            loss_physics_list[i] = torch.mean(loss_physics**2).item()
            loss_b_list[i] = loss_b.item()
            loss_ic_list[i] = loss_ic.item()
            tepoch.set_postfix(loss_iter=loss.item(), loss_mean=np.mean(loss_list[i-10:]), 
                               loss_std=np.std(loss_list[i-10:]))

            loss.backward()
            optimizer.step()
            
            # save model checkpoint
            torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, 'pinn_checkp_test.pth')

    return model, loss_list, loss_regr_list, loss_physics_list, loss_b_list, loss_ic_list


def get_dens_pinn(y_pred, num_t=1000, num_x=1132):
    return (np.abs(y_pred[:, 0] + 1j*y_pred[:, 1])**2).reshape(num_t, num_x)


