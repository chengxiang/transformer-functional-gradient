###########################################
# This file contains the following:
# 1. Linear Transformer Model
# 2. Function for clipping gradient
# 3. Function for generating random data
#
# The notation for linear attention follows
# the paper at https://arxiv.org/pdf/2306.00297.pdf
###########################################


import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definition of a single linear attention unit for linear-regression data
# P is the value matrix
# Q is the product of key,query matrices
# the dimensions of the input are
# B: batch-size of prompts
# N: context length (excluding query)
# d: covariate dimension
# P,Q,K are d x d matrices
# Z is a B x (N+1) + (d+1) matrix
# Output is also B x (N+1) + (d+1)

# For linear attention, activation = None
# For standard attention, activation(x) = torch.nn.functional.softmax(x, dim = 2)
# For ReLU attention, activation(x) = torch.nn.relu(x)

def softmax_activation(mask, Attn):
    Attn = Attn.masked_fill(mask, float('-inf'))
    return torch.nn.functional.softmax(Attn, dim = 1)

def relu_activation(mask, Attn):
    Attn = Attn.masked_fill(mask, float('0'))
    return torch.nn.functional.relu(Attn)

def exp_activation(mask, Attn):
    Attn = torch.exp(Attn)
    Attn = Attn.masked_fill(mask, float('0'))
    return Attn

def linear_activation(mask, Attn):
    Attn = Attn.masked_fill(mask, float('0'))
    return Attn

def attention(P,Q,K,Z,mask, activation = 'exp'):
    B= Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    QK = torch.einsum('ji,jk->ik', (Q,K))
    Attn = torch.einsum('BNi, ij, BMj -> BNM', (Z,QK,Z))
    
    PZ = torch.einsum('ij, BNj -> BNi', (P,Z))
    
    if activation == 'exp':
        Attn = exp_activation(mask, Attn)
    elif activation == 'linear':
        Attn = linear_activation(mask, Attn)
    elif activation == 'relu':
        Attn = relu_activation(mask, Attn)
    elif activation == 'softmax':
        Attn = softmax_activation(mask, Attn)
    else:
        assert(False)
    Output = torch.einsum('BNi, BNM -> BMi', (PZ,Attn))
    return Output /N



# The Transformer network
# n_layer denotes the number of layers
# n_head denotes the number of heads. In most of our experiments, n_head = 1
# d denotes the dimension of covariates
# var denotes the variance of initialization. It needs to be sufficiently small, but exact value is not important
# allparam: contains all the parameters, has dimension n_layer x n_head x 2 x d x d
# For example
# - P matrix at layer i, head j is allparam[i,j,0,:,:]
# - Q matrix at layer i, head j is allparam[i,j,1,:,:]
# - K matrix at layer i, head j is allparam[i,j,1,:,:]
class Transformer_F(nn.Module):
    def __init__(self, n_layer, n_head, d, var, N=20):
        super(Transformer_F, self).__init__()
        self.register_parameter('allparam', torch.nn.Parameter(torch.zeros(n_layer, n_head, 3, d+1, d+1)))
        with torch.no_grad():
            self.allparam.normal_(0,var)
        self.n_layer = n_layer
        self.n_head = n_head
        self.register_buffer('mask',torch.zeros([1,N+1,N+1], dtype=torch.bool))
        self.mask[:,-1, :] = True
        self.register_buffer('param_mask',torch.zeros([1,1,3,d+1,d+1], dtype=torch.bool))
        self.param_mask[0,0,0,d,:d]=True
        self.param_mask[0,0,0,:d,d]=True
        self.param_mask[0,0,1,:d+1,d]=True
        self.param_mask[0,0,1,d,:d+1]=True
        self.param_mask[0,0,2,:d+1,d]=True
        self.param_mask[0,0,2,d,:d+1]=True
        

    def forward(self, Z, activation):
        for i in range(self.n_layer):
            Zi = Z
            residues = 0
            # the forwarad map of each layer is given by F(Z) = Z + attention(Z)
            for j in range(self.n_head):
                Pij = self.allparam[i,j,0,:,:]
                Qij = self.allparam[i,j,1,:,:]
                Kij = self.allparam[i,j,2,:,:]
                residues = residues + attention(Pij,Qij,Kij, Zi, self.mask, activation)
            Z = Zi + residues
            
        if Z.norm() > 1e10:
            Z = 1e10* Z/Z.norm()
        return Z
    
    def zero_row_col(self):
        with torch.no_grad():
            self.allparam.data = self.allparam.data.masked_fill(self.param_mask, 0)
    #enforces top-left-dxd-block sparsity on p
    def zero_block_P(self):
        d = self.allparam.shape[4]
        for i in range(self.n_layer):
            for j in range(self.n_head):
                with torch.no_grad():
                    self.allparam[i,j,0,:d-1,:d-1].zero_()

# evaluate the loss of model, given data (Z,y)
def in_context_loss(model, Z, y, activation):
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    output = model(Z, activation)
    diff = output[:,N,d]+y
    loss = ((diff)**2).mean() 
    return loss

def euclidean_kernel(X):
    kernel_mat = torch.einsum('BNi,BMi->BNM', (X,X))
    return kernel_mat

def relu_kernel(X):
    kernel_mat = torch.einsum('BNi,BMi->BNM', (X,X))
    return torch.nn.functional.relu(kernel_mat)

def exp_kernel(X,sigma=1):
    kernel_mat = torch.einsum('BNi,BMi->BNM', (X,X))
    return torch.exp(1/(2*sigma**2)*kernel_mat)

def combination_kernel(X,sigma=1):
    kernel_mat0 = torch.einsum('BNi,BMi->BNM', (X[:,:,0:2],X[:,:,0:2]))
    kernel_mat1 = torch.einsum('BNi,BMi->BNM', (X[:,:,2:-1],X[:,:,2:-1]))
    return torch.exp(1/(2*sigma**2)*kernel_mat0) + kernel_mat1

def generate_data_inplace(Z, kernel, U=None, D=None):
    B = Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    X = Z[:,:,0:-1]
    X.normal_(0, 1).cuda()
    
    X.div_(X.norm(p=2,dim=2)[:,:,None])
    
    
    if kernel=='euclidean':
        kernel_matrices = euclidean_kernel(X) + torch.eye(N+1,N+1).unsqueeze(0).cuda() * 1e-8 #regularization for cholesky
    elif kernel=='exp':
        kernel_matrices = exp_kernel(X)
    elif kernel=='relu':
        kernel_matrices = relu_kernel(X)
    elif kernel=='comb':
        kernel_matrices = combination_kernel(X)
    else:
        print(kernel)
        assert False
    L, Q = torch.linalg.eigh(kernel_matrices)
    
    Z[:,:,-1].normal_(0,1)
    Z[:,:,-1] = torch.einsum('BNM,BM,BM-> BN', (Q, L.abs()**0.5, Z[:,:,-1]))
    
    
    y_test = Z[:,-1,-1].detach().clone()
    Z[:,-1,-1].zero_()
    if U is not None:
        U = U.to(device)
        D = D.to(device)
        Z[:,:,0:-1] = torch.einsum('ij, jk, BNk -> BNi', (U,D,X))
    return Z.to(device),y_test.to(device)

def bayes_prediction(Z, kernel, U, D):
    B = Z.shape[0]
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    
    X = Z[:,:,0:-1]
    y = Z[:,0:-1,-1]
    
    if kernel=='euclidean':
        kernel_matrices = euclidean_kernel(X) + torch.eye(N+1,N+1).unsqueeze(0).cuda() * 1e-4 #regularization for cholesky
    elif kernel=='exp':
        kernel_matrices = exp_kernel(X) 
    elif kernel=='relu':
        kernel_matrices = relu_kernel(X)
    elif kernel=='comb':
        kernel_matrices = combination_kernel(X)
    else:
        assert False
        
    
    L, Q = torch.linalg.eigh(kernel_matrices)
    kernel_matrices = torch.einsum('BNM,BM,BOM-> BNO', (Q, L.abs(), Q))
    v = kernel_matrices[:,0:-1,-1]
    
    tt = torch.linalg.solve(kernel_matrices[:,0:-1,0:-1],y)
    bayes_pred = torch.einsum('Bi,Bi->B',(v,tt))
    
    return bayes_pred

def bayes_loss(Z, y, activation, U, D):
    N = Z.shape[1]-1
    d = Z.shape[2]-1
    output = bayes_prediction(Z, activation, U, D)
    diff = output-y
    loss = ((diff)**2).mean() 
    
    #print(diff)
    return loss

class Transformer_C(nn.Module):
    def __init__(self, n_layer, n_head, d, var, N=20):
        super(Transformer_C, self).__init__()
        self.register_parameter('allparam', torch.nn.Parameter(torch.zeros(n_layer, n_head, 3, d+1, d+1)))
        with torch.no_grad():
            self.allparam.normal_(0,var)
        self.n_layer = n_layer
        self.n_head = n_head
        self.register_buffer('mask',torch.zeros([1,N+1,N+1], dtype=torch.bool))
        self.mask[:,-1, :] = True
        self.register_buffer('param_mask',torch.zeros([1,1,3,d+1,d+1], dtype=torch.bool))
        self.param_mask[0,0,0,d,:d]=True
        self.param_mask[0,0,0,:d,d]=True
        self.param_mask[0,0,1,:d+1,d]=True
        self.param_mask[0,0,1,d,:d+1]=True
        self.param_mask[0,0,2,:d+1,d]=True
        self.param_mask[0,0,2,d,:d+1]=True
        assert n_head == 2
        

    def forward(self, Z, activation):
        #ignore activation
        for i in range(self.n_layer):
            Zi = Z
            residues = 0
            # the forwarad map of each layer is given by F(Z) = Z + attention(Z)
            j=0
            Pij = self.allparam[i,j,0,:,:]
            Qij = self.allparam[i,j,1,:,:]
            Kij = self.allparam[i,j,2,:,:]
            residues = residues + attention(Pij,Qij,Kij, Zi, self.mask, 'exp')
            j=1
            Pij = self.allparam[i,j,0,:,:]
            Qij = self.allparam[i,j,1,:,:]
            Kij = self.allparam[i,j,2,:,:]
            residues = residues + attention(Pij,Qij,Kij, Zi, self.mask, 'linear')
            Z = Zi + residues
            
        if Z.norm() > 1e10:
            Z = 1e10* Z/Z.norm()
        return Z
    
    def zero_row_col(self):
        with torch.no_grad():
            self.allparam.data = self.allparam.data.masked_fill(self.param_mask, 0)