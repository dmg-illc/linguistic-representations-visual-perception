import torch
import numpy as np

def compute_permutation_indices(n_images, n_permutations):

    np.random.seed(23)
    arr = np.arange((n_images*n_images-n_images) / 2)
    indices = np.stack([np.random.permutation(arr) for i in range(n_permutations)])

    return indices.astype(int)



def spearman_rowwise(A: np.array, B: np.array, chunk_size: int = 500) -> torch.Tensor:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    A = torch.from_numpy(A).to(device).to(torch.float32)
    B = torch.from_numpy(B).to(device).to(torch.float32)
    n_rows, n_cols = A.shape

    rank_b = _rankdata_1d(B)
    b_repeated = rank_b.unsqueeze(0).expand(n_rows, -1)  # (n_rows, n_cols)

    rhos = torch.empty(n_rows, dtype=torch.float32, device=A.device)

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)

        a_chunk = A[start:end]          # (chunk, n_cols)
        b_chunk = b_repeated[start:end] # (chunk, n_cols)
        rank_a = _rankdata_2d(a_chunk)  # rank each row

        # Center ranks
        ra_mean = rank_a.mean(dim=1, keepdim=True)
        rb_mean = b_chunk.mean(dim=1, keepdim=True)
        xa = rank_a - ra_mean
        xb = b_chunk - rb_mean

        # Numerator: sum(xa * xb) per row
        num = (xa * xb).sum(dim=1)

        # Denominator: sqrt(sum(xa^2) * sum(xb^2)) per row
        sa2 = (xa ** 2).sum(dim=1)
        sb2 = (xb ** 2).sum(dim=1)
        denom = torch.sqrt(sa2 * sb2)

        # Avoid division by zero for constant rows
        rho = torch.where(denom == 0, torch.tensor(float("nan"), device=A.device), num / denom)
        rhos[start:end] = rho

    return rhos.cpu().numpy()


def _rankdata_1d(x: torch.Tensor) -> torch.Tensor:
    """Assign average ranks to a 1D tensor (mirrors scipy.stats.rankdata)."""
    # Reuse the 2D implementation on a single-row view
    return _rankdata_2d(x.unsqueeze(0)).squeeze(0)


def _rankdata_2d(x: torch.Tensor) -> torch.Tensor:
    """Assign average ranks row-wise to a 2D tensor, fully vectorized."""
    n_rows, n_cols = x.shape
    device = x.device

    order = torch.argsort(x, dim=1)                         
    ranks = torch.empty_like(x)
    row_idx = torch.arange(n_rows, device=device).unsqueeze(1).expand_as(order)
    col_ranks = torch.arange(1, n_cols + 1, dtype=torch.float32, device=device) \
                     .unsqueeze(0).expand(n_rows, -1)       
    ranks[row_idx, order] = col_ranks

    sorted_x = x.gather(1, order)                           
    
    new_group = torch.cat([
        torch.ones(n_rows, 1, dtype=torch.bool, device=device),
        sorted_x[:, 1:] != sorted_x[:, :-1]
    ], dim=1)                                               

    # Early exit: if no ties at all, skip averaging
    if new_group.all():
        return ranks

    group_id = new_group.cumsum(dim=1) - 1                  

    row_offsets = (torch.arange(n_rows, device=device) * n_cols).unsqueeze(1)
    flat_group_id = (group_id + row_offsets).view(-1)      

    flat_ranks = ranks[row_idx, order].view(-1)            
    n_groups = n_rows * n_cols                              

    group_sum = torch.zeros(n_groups, dtype=torch.float32, device=device)
    group_count = torch.zeros(n_groups, dtype=torch.float32, device=device)
    ones = torch.ones_like(flat_ranks)

    group_sum.scatter_add_(0, flat_group_id, flat_ranks)
    group_count.scatter_add_(0, flat_group_id, ones)

    avg_rank = group_sum / group_count.clamp(min=1)         
    averaged_sorted_ranks = avg_rank[flat_group_id].view(n_rows, n_cols)

    # Write averaged ranks back into original (unsorted) positions
    ranks[row_idx, order] = averaged_sorted_ranks

    return ranks