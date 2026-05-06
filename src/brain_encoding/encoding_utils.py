import torch
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneOut

def standardise_features(x_train, x_test):
        
        x_train_mean = x_train.mean(dim=0, keepdim=True)
        x_train_std = x_train.std(dim=0, unbiased=False, keepdim=True)
        x_train_std = torch.where(x_train_std == 0, torch.tensor(1.0, device=x_train.device), x_train_std)
        x_train_scaled = (x_train - x_train_mean) / x_train_std
        x_test_scaled = (x_test - x_train_mean) / x_train_std

        return x_train_scaled, x_test_scaled

def pca(x_train, x_test):
    
        
    n_samples, n_features = x_train.shape
    if n_samples < n_features:
        # compute SVD on X (n_samples x n_features) directly
        # X = U S Vt
        U, S, Vt = torch.linalg.svd(x_train, full_matrices=False)
        # explained variance ratio = S^2 / sum(S^2)
        explained = S ** 2
        cum = torch.cumsum(explained, dim=0) / torch.sum(explained)
        # find number of components to reach 0.99
        k = int(torch.searchsorted(cum, 0.99).item()) + 1
        V = Vt[:k, :].T  # n_features x k
        # transform
        train_proj = x_train @ V
        test_proj = x_test @ V
    else:
        # n_samples >= n_features: compute SVD on X^T X may be heavy; instead do SVD on X directly as well
        U, S, Vt = torch.linalg.svd(x_train, full_matrices=False)
        explained = S ** 2
        cum = torch.cumsum(explained, dim=0) / torch.sum(explained)
        k = int(torch.searchsorted(cum, 0.99).item()) + 1
        V = Vt[:k, :].T
        train_proj = x_train @ V
        test_proj = x_test @ V

    return train_proj, test_proj


def compute_correlations(Y_pred: torch.Tensor, Y_true: torch.Tensor) -> torch.Tensor:
    # (n_samples, n_voxels)
    n = Y_pred.shape[0]

    pred_mean = Y_pred.mean(dim=0, keepdim=True)
    true_mean = Y_true.mean(dim=0, keepdim=True)

    xm = Y_pred - pred_mean
    ym = Y_true - true_mean

    corr = F.cosine_similarity(xm, ym, dim=0)

    return corr


def compute_batch_correlations(predictions, y_test):
        """
        Compute correlations between predictions and targets for all permutations
        predictions: (n_permutations, n_test, n_voxels)
        y_test: (n_test, n_voxels)
        """
        
        # Center the data
        pred_centered = predictions - predictions.mean(dim=1, keepdim=True)
        y_test_centered = y_test - y_test.mean(dim=0, keepdim=True)
        
        # Compute correlations vectorized
        numerator = torch.einsum('ptv,tv->pv', pred_centered, y_test_centered)
        
        pred_std = torch.sqrt(torch.sum(pred_centered**2, dim=1))
        y_test_std = torch.sqrt(torch.sum(y_test_centered**2, dim=0))
        
        denominator = pred_std * y_test_std.unsqueeze(0)
        
        # Handle division by zero
        correlations = numerator / (denominator + 1e-8)
        
        return correlations
    

def TensorRidge(x_train, x_test, y_train, alphas, batch_size=1000):
     
    n_samples, n_features = x_train.size()
    n_samples, n_voxels = y_train.size()
    device = x_train.device
    dtype = torch.float64

    x_train = x_train.to(device=device, dtype=dtype)
    x_test = x_test.to(device=device, dtype=dtype)
    y_train = y_train.to(device=device, dtype=dtype)
    alphas = alphas.to(device=device, dtype=dtype)
     
    XtX = x_train.T @ x_train  # [n_features, n_features]

    # Precompute X^T Y
    XtY = x_train.T @ y_train  # [n_features, n_voxels]
    W_list = []

    for start in range(0, n_voxels, batch_size):
        end = min(start + batch_size, n_voxels)
        batch_alphas = alphas[start:end]  # [batch_size]
        batch_XtY = XtY[:, start:end]     # [n_features, batch_size]

        # Build batched XtX + alpha*I: [batch_size, n_features, n_features]
        eye = torch.eye(n_features, device='cuda', dtype=dtype)
        XtX_batch = XtX.unsqueeze(0) + batch_alphas.view(-1, 1, 1) * eye.unsqueeze(0)

        # Prepare XtY for batched solve: [batch_size, n_features, 1]
        XtY_batch = batch_XtY.T.unsqueeze(-1)

        # Solve in batch
        W_batch = torch.linalg.solve(XtX_batch, XtY_batch)  # [batch_size, n_features, 1]
        W_list.append(W_batch.squeeze(-1).T)               # [n_features, batch_size]

    # Concatenate all batches
    W_all = torch.cat(W_list, dim=1)  # [n_features, n_voxels]
    # Get predictions
    Y_pred = torch.matmul(x_test, W_all)    # slightly faster for large tensors

    return Y_pred

def TensorRidgeCV(x_train, y_train, alphas):

    device = x_train.device
    n_samples, n_voxels = y_train.size()
    n_samples, n_features = x_train.size()
    dtype = torch.float64

    x_train = x_train.to(device=device, dtype=dtype)
    y_train = y_train.to(device=device, dtype=dtype)
    alphas = alphas.to(device=device, dtype=dtype)

    # Cross-validation setup
    k_folds = 5
    fold_size = n_samples // k_folds
    indices = torch.arange(n_samples)

    # To store scores: [n_voxels, n_alphas]
    scores = torch.zeros(n_voxels, len(alphas), device='cuda')

    # ---------------------------
    # Batched RidgeCV with Pearson correlation
    # ---------------------------
    for k in range(k_folds):
        # Train/test split
        test_idx = indices[k*fold_size:(k+1)*fold_size]
        train_idx = torch.cat([indices[:k*fold_size], indices[(k+1)*fold_size:]])

        X_train, X_test = x_train[train_idx], x_train[test_idx]
        Y_train, Y_test = y_train[train_idx], y_train[test_idx]

        # Precompute XtX and XtY
        XtX = X_train.T @ X_train            
        XtY = X_train.T @ Y_train            

        # Build batched XtX + alpha*I: [n_alphas, n_features, n_features]
        eye = torch.eye(n_features, device='cuda', dtype=torch.float32)
        XtX_batch = XtX.unsqueeze(0) + alphas.view(-1, 1, 1) * eye.unsqueeze(0)

        # XtY_batch: [n_alphas, n_features, n_voxels]
        XtY_batch = XtY.unsqueeze(0).expand(len(alphas), -1, -1)

        # Solve for weights: [n_alphas, n_features, n_voxels]
        W = torch.linalg.solve(XtX_batch, XtY_batch)

        # Predictions: [n_alphas, n_test, n_voxels]
        X_test_batch = X_test.unsqueeze(0)
        Y_pred = torch.matmul(X_test_batch, W)

        # Pearson correlation per voxel and alpha
        # Center data
        Y_test_mean = Y_test.mean(dim=0, keepdim=True)          
        Y_pred_mean = Y_pred.mean(dim=1, keepdim=True)          
        Y_test_centered = Y_test.unsqueeze(0) - Y_test_mean     
        Y_pred_centered = Y_pred - Y_pred_mean                  

        # Compute numerator and denominator
        numerator = (Y_pred_centered * Y_test_centered).sum(dim=1)      
        denominator = torch.sqrt((Y_pred_centered**2).sum(dim=1) * (Y_test_centered**2).sum(dim=1))  

        corr = numerator / (denominator + 1e-8)  # Add small eps to avoid division by zero
        scores += corr.T  # [n_voxels, n_alphas]

    # Average across folds
    scores /= k_folds

    # Best alpha per voxel
    best_alpha_idx = scores.argmax(dim=1)
    best_alphas = alphas[best_alpha_idx]

    return best_alphas
     


