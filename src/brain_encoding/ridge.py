import torch
from sklearn.model_selection import KFold


def compute_correlations(Y_pred, Y_true):
    xm = Y_pred - Y_pred.mean(0, keepdim=True)
    ym = Y_true - Y_true.mean(0, keepdim=True)
    corr = (xm * ym).sum(0) / (xm.norm(dim=0) * ym.norm(dim=0) + 1e-12)
    return corr


class MultiRidge:
    """Ridge model for multiple outputs and regularization strengths.
    Assumes X is already scaled (no centering or scaling is done)."""

    def __init__(self, ls):
        self.ls = ls
        self.scoring = compute_correlations
        self.X_t = None
        self.e = None
        self.Q = None
        self.Y = None
        self.Ym = None
        # Cached intermediates
        self._r = None
        self._p = None

    def fit(self, X, Y):
        """
        Arguments:
            X: 2D tensor (n, d) of pre-scaled features
            Y: 2D tensor (n, m) of targets
        """
        self.X_t = X.t()
        _, S, V = self.X_t.svd()
        self.e = S.pow_(2)
        self.Q = self.X_t @ V

        self.Y = Y
        self.Ym = Y.mean(dim=0)

        # Precompute intermediates (cache for reuse)
        Y_centered = self.Y - self.Ym 
        self._p = self.X_t @ Y_centered      
        self._r = self.Q.t() @ self._p       
        return self

    def _compute_pred_interms(self):
        """Return cached matrices used for prediction."""
        return self.Ym, self._r, self._p

    def _compute_single_beta(self, l, y_idx):
        Y_j, Ym_j = self.Y[:, y_idx], self.Ym[y_idx]
        beta = (1 / l) * (
            self.X_t @ (Y_j - Ym_j)
            - self.Q / (self.e + l) @ self.Q.t() @ self.X_t @ (Y_j - Ym_j)
        )
        return beta

    def get_model_weights_and_bias(self, l_idxs):
        betas = torch.zeros((self.X_t.shape[0], len(l_idxs)))
        for j, l_idx in enumerate(l_idxs):
            l = self.ls[l_idx]
            betas[:, j] = self._compute_single_beta(l, j)
        return betas, self.Ym

    @torch.no_grad()
    def get_prediction_scores(self, X_te, Y_te):
        """Compute correlations for all regularization values."""
        M_te = X_te @ self.Q
        Ym, r, p = self._compute_pred_interms()
        N_te = X_te @ p  

        E = self.e[:, None]
        ls = self.ls
        inv_denoms = 1.0 / (E + ls[None, :])
        r_scaled = r[:, :, None] * inv_denoms[:, None, :]
        M_r = torch.einsum("nd,dml->nml", M_te, r_scaled)

        N_expanded = N_te[:, :, None]
        Ym_expanded = Ym[None, :, None]
        Yhat_te = (1.0 / ls)[None, None, :] * (N_expanded - M_r) + Ym_expanded

        Y_te_centered = Y_te - Y_te.mean(0, keepdim=True)
        Yhat_te_centered = Yhat_te - Yhat_te.mean(0, keepdim=True)
        num = torch.einsum("nm,nml->ml", Y_te_centered, Yhat_te_centered)
        denom_true = Y_te_centered.norm(dim=0, keepdim=True)
        denom_pred = Yhat_te_centered.norm(dim=0)
        denom = denom_true.t() * denom_pred
        return num / (denom + 1e-12)

    def predict_single(self, X_te, l_idxs):
        """Predict using one lambda per output (vectorized version)."""
        M_te = X_te @ self.Q
        Ym, r, p = self._compute_pred_interms()
        N_te = X_te @ p
        ls = self.ls[l_idxs]

        # Vectorized prediction (no Python loop)
        E = self.e[:, None]
        inv_terms = 1.0 / (E + ls[None, :]) 
        M_r = M_te @ (r * inv_terms)        
        Yhat_te = (1.0 / ls)[None, :] * (N_te - M_r) + Ym[None, :]
        return Yhat_te


class RidgeCVEstimator:
    """Cross-validated ridge regression (X assumed pre-scaled)."""

    def __init__(self, ls, n_permutations = 1000):
        self.ls = ls
        self.cv = KFold(n_splits=5)
        self.scoring = compute_correlations
        self.base_ridge = None
        self.mean_cv_scores = None
        self.best_l_scores = None
        self.best_l_idxs = None
        self.n_permutations = n_permutations


    def fit(self, X, Y, groups=None, X_test=None, Y_test=None, permutation_indices=None, run_permutations=False):
        """
        Fit the model with cross-validation and optional permutation testing.
        
        Arguments:
            X: 2D tensor (n, d) of pre-scaled training features
            Y: 2D tensor (n, m) of training targets
            groups: Optional grouping for cross-validation splits
            X_test: Optional test features for permutation testing
            Y_test: Optional test targets for permutation testing
            run_permutations: Whether to run permutation tests (default: True)
        """
        # Standard cross-validation to find best alphas
        cv_scores = []

        for idx_tr, idx_val in self.cv.split(X, Y):
            X_tr, X_val = X[idx_tr], X[idx_val]
            Y_tr, Y_val = Y[idx_tr], Y[idx_val]

            ridge = MultiRidge(self.ls)
            ridge.fit(X_tr, Y_tr)
            split_scores = ridge.get_prediction_scores(X_val, Y_val)
            cv_scores.append(split_scores)

        cv_scores = torch.stack(cv_scores)
        self.mean_cv_scores = cv_scores.mean(dim=0)
        self.best_l_scores, self.best_l_idxs = self.mean_cv_scores.max(dim=1)

        # Fit final model on full data
        self.base_ridge = MultiRidge(self.ls).fit(X, Y)
        
        # Run permutation tests if requested
        if run_permutations:
            if X_test is None or Y_test is None:
                raise ValueError("X_test and Y_test must be provided for permutation testing")
            self._run_permutation_tests(X, Y, X_test, Y_test, permutation_indices)
        
        return self

    def _run_permutation_tests(self, X_train, Y_train, X_test, Y_test, permutation_indices):
        """
        Run batch permutation tests using the best alphas identified during CV.
        Train on permuted training labels, but predict on non-permuted test set.
        Fully vectorized implementation processing all permutations at once.
        
        Arguments:
            X_train: Training features
            Y_train: Training labels (will be permuted)
            X_test: Test features
            Y_test: Test labels (NOT permuted)
        """
        
        # Create all permuted Y matrices at once: (n_permutations, n_train, n_outputs)
        Y_perm_batch = Y_train[permutation_indices] # should work thanks to 'advanced pytorch indexing'
        
        # Get best lambdas for each output
        best_ls = self.ls[self.best_l_idxs]  # (n_outputs,)
        
        # Precompute X_train-related terms (shared across all permutations)
        X_train_t = X_train.t()  # (d, n_train)
        _, S, V = X_train_t.svd()
        e = S.pow(2)  # (d,)
        Q = X_train_t @ V  # (d, d)

        # Batch compute all intermediates: shape (n_permutations, ...)
        Ym_perm = Y_perm_batch.mean(dim=1) 
        Y_centered = Y_perm_batch - Ym_perm[:, None, :]  
        
        # p: X_train_t @ Y_centered for each permutation
        p = torch.einsum('dn,pno->pdo', X_train_t, Y_centered)  
        
        # r: Q.t() @ p for each permutation
        r = torch.einsum('dd,pdo->pdo', Q.t(), p)  
        
        # Compute predictions on TEST set for all permutations at once
        M_test = X_test @ Q 
        N_test = torch.einsum('nd,pdo->pno', X_test, p)  
        
        # Compute M_r: M_test @ (r * inv_terms) for each permutation
        E = e[:, None]  
        inv_terms = 1.0 / (E + best_ls[None, :])  
        r_scaled = r * inv_terms[None, :, :]  
        M_r_test = torch.einsum('nd,pdo->pno', M_test, r_scaled)  
        
        # Final predictions on test set: (n_permutations, n_test, n_outputs)
        Y_pred_batch = (1.0 / best_ls)[None, None, :] * (N_test - M_r_test) + Ym_perm[:, None, :]
        
        # Compute correlations against NON-PERMUTED test labels
        Y_test_centered = Y_test - Y_test.mean(0, keepdim=True)
        Yhat_centered = Y_pred_batch - Y_pred_batch.mean(dim=1, keepdim=True)  
        
        num = torch.einsum('no,pno->po', Y_test_centered, Yhat_centered)
        denom_true = Y_test_centered.norm(dim=0)  
        denom_pred = Yhat_centered.norm(dim=1)  
        perm_scores = num / (denom_true[None, :] * denom_pred + 1e-12)
        
        self.permutation_scores = perm_scores
      
        
        return self

    def predict(self, X):
        if self.best_l_idxs is None:
            raise RuntimeError("Cannot predict before fitting.")
        return self.base_ridge.predict_single(X, self.best_l_idxs)

    def get_model_weights_and_bias(self):
        if self.best_l_idxs is None:
            raise RuntimeError("Cannot return weights before fitting.")
        return self.base_ridge.get_model_weights_and_bias(self.best_l_idxs)
    
    def get_permutation_results(self):
        """
        Return permutation test results.
        
        Returns:
            dict with keys:
                - 'p_values': p-value for each output
                - 'permutation_scores': all permutation scores (n_permutations, n_outputs)
                - 'actual_scores': actual CV scores for best alphas
        """
        if self.permutation_scores is None:
            raise RuntimeError("Permutation tests have not been run.")
        
        return self.permutation_scores

    def predict(self, X):
        if self.best_l_idxs is None:
            raise RuntimeError("Cannot predict before fitting.")
        return self.base_ridge.predict_single(X, self.best_l_idxs)

    def get_model_weights_and_bias(self):
        if self.best_l_idxs is None:
            raise RuntimeError("Cannot return weights before fitting.")
        return self.base_ridge.get_model_weights_and_bias(self.best_l_idxs)
    
    def run_permutation_tests(self, X, Y):
            """
            Run batch permutation tests using the best alphas identified during CV.
            Fully vectorized implementation processing all permutations at once.
            """
            n_samples, n_outputs = Y.shape
            
            # Create all permuted Y matrices at once: (n_permutations, n_samples, n_outputs)
            Y_perm_batch = torch.zeros((self.n_permutations, n_samples, n_outputs)).to(X.device)
            for output_idx in range(n_outputs):
                for perm_idx in range(self.n_permutations):
                    perm_indices = torch.randperm(n_samples)
                    Y_perm_batch[perm_idx, :, output_idx] = Y[perm_indices, output_idx]
    
            
            # Get best lambdas for each output
            best_ls = self.ls[self.best_l_idxs]  # (n_outputs,)
            
            # Precompute X-related terms (shared across all permutations)
            X_t = X.t()  
            _, S, V = X_t.svd()
            e = S.pow(2)  
            Q = X_t @ V  
            
            # Batch compute all intermediates: shape (n_permutations, ...)
            Ym_perm = Y_perm_batch.mean(dim=1) 
            Y_centered = Y_perm_batch - Ym_perm[:, None, :]  
            
        
            # Reshape to (n_permutations, d, n_outputs)
            p = torch.einsum('dn,pno->pdo', X_t, Y_centered)  
            
            # r: Q.t() @ p for each permutation
            r = torch.einsum('dd,pdo->pdo', Q.t(), p) 
            
            # Compute predictions for all permutations at once
            M = X @ Q  # (n, d) - shared
            N = torch.einsum('nd,pdo->pno', X, p)  
            
            # Compute M_r: M @ (r * inv_terms) for each permutation
            E = e[:, None]  # (d, 1)
            inv_terms = 1.0 / (E + best_ls[None, :]) 
            # r * inv_terms: (n_permutations, d, n_outputs) * (d, n_outputs)
            r_scaled = r * inv_terms[None, :, :]  
            M_r = torch.einsum('nd,pdo->pno', M, r_scaled) 
            
            # Final predictions: (n_permutations, n, n_outputs)
            Y_pred_batch = (1.0 / best_ls)[None, None, :] * (N - M_r) + Ym_perm[:, None, :]
            
            # Compute correlations for all permutations at once
            Y_centered = Y_perm_batch - Y_perm_batch.mean(dim=1, keepdim=True)
            Yhat_centered = Y_pred_batch - Y_pred_batch.mean(dim=1, keepdim=True)
            
            num = torch.einsum('pno,pno->po', Y_centered, Yhat_centered)
            denom_true = Y_centered.norm(dim=1)  
            denom_pred = Yhat_centered.norm(dim=1) 
            perm_scores = num / (denom_true * denom_pred + 1e-12)
            
            permutation_scores = perm_scores
            
            return permutation_scores

