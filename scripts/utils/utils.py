import torch
import torch.nn.functional as F

def desc_l2norm(desc: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize descriptors with shape [N, C] or [N, C, H, W]
    """
    return F.normalize(desc, p=2, dim=1, eps=1e-10)
    
def matching_points(grd_desc, grd_scrs, sat_desc, sat_scrs, k=1024, temperature=0.1):
    """
    Matches points based on top-k scoring descriptors.

    Args:
        grd_desc (Tensor): Ground descriptors of shape [B, D, H, W] or [B, D, N].
        grd_scrs (Tensor): Ground scores of shape [B, 1, H, W] or [B, 1, N].
        sat_desc (Tensor): Satellite descriptors of shape [B, D, H, W] or [B, D, N].
        sat_scrs (Tensor): Satellite scores of shape [B, 1, H, W] or [B, 1, N].
        k (int): Number of top-scoring points to select.
        temperature (float): Scaling factor for similarity computation.

    Returns:
        similarity (Tensor): Similarity matrix of shape [B, K, K].
        sat_indices_topk (Tensor): Top-k satellite indices, shape [B, 1, K].
        grd_indices_topk (Tensor): Top-k ground indices, shape [B, 1, K].
    """

    # Flatten descriptors & scores to [B, D, N] and [B, 1, N]    
    grd_desc = grd_desc.reshape(grd_desc.shape[0], grd_desc.shape[1], -1)  
    grd_scrs = grd_scrs.reshape(grd_scrs.shape[0], 1, -1)  
    sat_desc = sat_desc.reshape(sat_desc.shape[0], sat_desc.shape[1], -1)  
    sat_scrs = sat_scrs.reshape(sat_scrs.shape[0], 1, -1)  

    # Ensure k is within valid bounds
    _, num_features, num_points_grd = grd_desc.shape  # [B, D, N]
    _, num_features, num_points_sat = sat_desc.shape  # [B, D, N]
    assert k <= num_points_grd and k <= num_points_sat, (f"k must be ≤ the number of points in both grids: "f"ground={num_points_grd}, satellite={num_points_sat}")

    # Get the top-k indices and scores
    sat_scrs_topk, sat_indices_topk = torch.topk(sat_scrs, k, dim=-1, sorted=True)  # [B, 1, K]
    grd_scrs_topk, grd_indices_topk = torch.topk(grd_scrs, k, dim=-1, sorted=True)  # [B, 1, K]
    
    # Gather the top-k descriptors
    sat_desc_topk = torch.gather(sat_desc, 2, sat_indices_topk.expand(-1, num_features, -1))  # Shape: [B, D, K]
    grd_desc_topk = torch.gather(grd_desc, 2, grd_indices_topk.expand(-1, num_features, -1))  # Shape: [B, D, K]

    # Compute similarity (scaled dot product)
    similarity = torch.matmul(sat_desc_topk.transpose(1, 2), grd_desc_topk) / temperature  # [B, K, K]

    # Full similarity for analysis (optional)
    full_similarity = torch.matmul(sat_desc.transpose(1, 2), grd_desc) / temperature  # [B, N_sat, N_grd]
    return similarity, sat_indices_topk, grd_indices_topk, full_similarity



def weighted_procrustes_2d(A, B, w=None, use_weights=True, use_mask=False, eps=1e-16, check_rank=True):
    """
    Computes 2D Weighted Procrustes Alignment.
    
    Args:
        A (Tensor): Source points [B, N, 2].
        B (Tensor): Target points [B, N, 2].
        w (Tensor): Weights [B, N].
    
    Returns:
        R (Tensor): Estimated rotation matrix [B, 2, 2].
        t (Tensor): Estimated translation vector [B, 1, 2].
        valid (bool): Indicates whether transformation is valid.
    """

    assert len(A) == len(B)

    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)
        A_mean, B_mean = (w_norm * A).sum(1, keepdim=True), (w_norm * B).sum(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean

        H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c) if use_mask else A_c.transpose(1, 2) @ (w_norm * B_c)
    else:
        A_mean, B_mean = A.mean(1, keepdim=True), B.mean(1, keepdim=True)
        A_c, B_c = A - A_mean, B - B_mean
        H = A_c.transpose(1, 2) @ B_c

    if check_rank and (torch.linalg.matrix_rank(H) == 1).sum() > 0:
        return None, None, False

    U, S, V = torch.svd(H)
    print('S', S)
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))

    R = V @ Z @ U.transpose(1, 2)
    t = B_mean - A_mean @ R.transpose(1, 2)

    return R, t, True


def soft_inlier_counting_bev(X0, X1, R, t, th=50):
    """
    Computes soft inlier count for BEV.
    """
    beta = 5 / th
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return torch.sigmoid(beta * (th - dist)).sum(-1, keepdim=True)


def inlier_counting_bev(X0, X1, R, t, th=50):
    """
    Computes binary inlier count for BEV.
    """
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return ((th - dist) >= 0).float()


