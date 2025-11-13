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
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))

    R = V @ Z @ U.transpose(1, 2)
    t = B_mean - A_mean @ R.transpose(1, 2)

    return R, t, True

def weighted_procrustes_2d_with_scale(A, B, w=None, use_weights=True, use_mask=False, eps=1e-16, check_rank=True):
    """
    Computes 2D Weighted Procrustes Alignment with scale estimation.

    Args:
        A (Tensor): Source points [B, N, 2].
        B (Tensor): Target points [B, N, 2] (up to scale).
        w (Tensor): Weights [B, N].

    Returns:
        R (Tensor): Estimated rotation matrix [B, 2, 2].
        t (Tensor): Estimated translation vector [B, 1, 2].
        s (Tensor): Estimated scale [B, 1, 1].
        valid (bool): Indicates whether transformation is valid.
    """
    assert len(A) == len(B)

    if use_weights:
        W1 = torch.abs(w).sum(1, keepdim=True)
        w_norm = (w / (W1 + eps)).unsqueeze(-1)  # [B, N, 1]

        A_mean = (w_norm * A).sum(1, keepdim=True)  # [B, 1, 2]
        B_mean = (w_norm * B).sum(1, keepdim=True)
        A_c = A - A_mean
        B_c = B - B_mean

        H = A_c.transpose(1, 2) @ (w.unsqueeze(-1) * B_c) if use_mask else A_c.transpose(1, 2) @ (w_norm * B_c)
    else:
        A_mean = A.mean(1, keepdim=True)
        B_mean = B.mean(1, keepdim=True)
        A_c = A - A_mean
        B_c = B - B_mean
        H = A_c.transpose(1, 2) @ B_c

    if check_rank and (torch.linalg.matrix_rank(H) == 1).sum() > 0:
        return None, None, None, False

    U, S, V = torch.svd(H)
    Z = torch.eye(2, device=A.device).unsqueeze(0).repeat(A.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))

    R = V @ Z @ U.transpose(1, 2)

    # Estimate scale s
    A_c_rot = A_c @ R.transpose(1, 2)
    if use_weights:
        numerator = (w_norm * B_c * A_c_rot).sum(dim=1, keepdim=True).sum(-1, keepdim=True)  # [B,1,1]
        denominator = (w_norm * A_c_rot ** 2).sum(dim=1, keepdim=True).sum(-1, keepdim=True) + eps  # [B,1,1]
    else:
        numerator = (B_c * A_c_rot).sum(dim=1, keepdim=True).sum(-1, keepdim=True)
        denominator = (A_c_rot ** 2).sum(dim=1, keepdim=True).sum(-1, keepdim=True) + eps

    s = numerator / denominator  # [B,1,1]
    # Adjust translation to include scale
    t = B_mean - s * (A_mean @ R.transpose(1, 2))  # [B,1,2]
    

    return R, t, s, True


def soft_inlier_counting_bev(X0, X1, R, t, th=50):
    """
    Computes soft inlier count for BEV.
    """
    beta = 5 / th
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return torch.sigmoid(beta * (th - dist)).sum(-1, keepdim=True)

def soft_inlier_counting_bev_with_scale(X0, X1, R, t, scale, th=50):
    """
    Computes soft inlier count for BEV.
    """
    beta = 5 / th
    X0_to_1 = scale * (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return torch.sigmoid(beta * (th - dist)).sum(-1, keepdim=True)


def inlier_counting_bev(X0, X1, R, t, th=50):
    """
    Computes binary inlier count for BEV.
    """
    X0_to_1 = (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return ((th - dist) >= 0).float()


def inlier_counting_bev_with_scale(X0, X1, R, t, scale, th=50):
    """
    Computes binary inlier count for BEV.
    """
    X0_to_1 = scale * (R @ X0.transpose(2, 1)).transpose(2, 1) + t
    dist = (((X0_to_1 - X1).pow(2).sum(-1) + 1e-6).sqrt())
    return ((th - dist) >= 0).float()

    
def get_top_points_in_pillars(
    bev_coord_grd, 
    metric_coord_grd, 
    mask, 
    max_depth, 
    pillar_size=0.1
):
    """
    Selects the top (highest z) point inside each XY pillar.

    Args:
        bev_coord_grd: (B, H*W, 2) tensor, BEV coordinates (x, y) in meters
        metric_coord_grd: (B, H*W, 3) tensor, full metric coordinates (x, y, z)
        mask: (B, H*W) tensor, valid points mask (True for valid)
        max_depth: float, maximum depth range
        pillar_size: float, size of each pillar in meters (default 0.1m)

    Returns:
        top_mask: (B, H*W) boolean tensor, True at top points
    """
    B, N, _ = bev_coord_grd.shape

    # Define pillar grid
    grid_range = max_depth
    min_x, min_y = -grid_range, -grid_range
    max_x, max_y = grid_range, grid_range
    num_pillars_x = int((max_x - min_x) / pillar_size)
    num_pillars_y = int((max_y - min_y) / pillar_size)
    num_total_pillars = num_pillars_x * num_pillars_y

    # Extract coordinates
    x, y = bev_coord_grd[..., 0], bev_coord_grd[..., 1]  # (B, H*W)
    z = metric_coord_grd[..., 2]                         # (B, H*W)

    # Compute pillar indices
    pillar_x = ((x - min_x) / pillar_size).floor().long()
    pillar_y = ((y - min_y) / pillar_size).floor().long()
    pillar_x = pillar_x.clamp(0, num_pillars_x - 1)
    pillar_y = pillar_y.clamp(0, num_pillars_y - 1)

    pillar_idx = pillar_y * num_pillars_x + pillar_x  # (B, H*W)
    pillar_idx[~mask] = -1

    # Flatten for batch processing
    batch_idx = torch.arange(B, device=bev_coord_grd.device).reshape(B, 1).expand(B, N)

    pillar_idx_flat = pillar_idx.reshape(-1)  # (B*N,)
    z_flat = z.reshape(-1)
    mask_flat = mask.reshape(-1)
    batch_idx_flat = batch_idx.reshape(-1)

    # Combined batch+pillar index
    combined_idx = batch_idx_flat * num_total_pillars + pillar_idx_flat

    # Keep only valid points
    valid = (pillar_idx_flat >= 0) & (mask_flat)
    combined_idx_valid = combined_idx[valid]
    z_valid = z_flat[valid]

    # Create tensor to hold max z per pillar
    max_z_per_pillar = torch.full(
        (B * num_total_pillars,), 
        -1e6, 
        dtype=z.dtype, 
        device=z.device
    )

    # Scatter reduce to find max per (batch, pillar)
    max_z_per_pillar.scatter_reduce_(
        0, 
        combined_idx_valid, 
        z_valid, 
        reduce='amax'
    )

    # Lookup corresponding max_z for each point
    corresponding_max_z = max_z_per_pillar[
        (batch_idx_flat * num_total_pillars + pillar_idx_flat).clamp(min=0)
    ]

    # Points matching their pillar's top z and valid
    is_top = (z_flat == corresponding_max_z) & (pillar_idx_flat >= 0) & (mask_flat)

    # Reshape
    top_mask = is_top.reshape(B, N)

    return top_mask

class e2eProbabilisticProcrustesSolver():
    """
    e2eProbabilisticProcrustesSolver computes the metric relative pose estimation during test time.
    Note that contrary to the training solver, here, the solver only refines the best pose hypothesis.
    Also, parameters are different during training and testing.
    """
    def __init__(self, it_RANSAC, it_matches, num_samples_matches, num_corr_2d_2d, num_refinements, th_inlier, th_soft_inlier, metric_coord_sat_B, metric_coord_grd_B):

        # Populate Procrustes RANSAC parameters
        self.it_RANSAC = it_RANSAC
        self.it_matches = it_matches
        self.num_samples_matches = num_samples_matches
        self.num_corr_2d_2d = num_corr_2d_2d
        self.num_refinements = num_refinements
        self.th_inlier = th_inlier
        self.th_soft_inlier = th_soft_inlier
        self.metric_coord_sat_B = metric_coord_sat_B
        self.metric_coord_grd_B = metric_coord_grd_B

    def estimate_pose(self, matching_score, return_inliers=False):
        '''
            Given 3D coordinates and matching matrices, estimate_pose computes the metric pose between query and reference images.
            args:
                return_inliers: Optional argument that indicates if a list of the inliers should be returned.
        '''
        device = matching_score.device
        matches = matching_score.detach()

        B, num_kpts_sat, num_kpts_grd = matches.shape

        matches_row = matches.reshape(B, num_kpts_sat*num_kpts_grd)
        batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, self.num_samples_matches]).reshape(B, self.num_samples_matches)
        batch_idx_ransac = torch.tile(torch.arange(B).view(B, 1), [1, self.num_corr_2d_2d]).reshape(B, self.num_corr_2d_2d)

        num_valid_h = 0
        Rs = torch.zeros((B, 0, 2, 2)).to(device)
        ts = torch.zeros((B, 0, 1, 2)).to(device)
        scales = torch.zeros((B, 0, 1, 1)).to(device)
        scores_ransac = torch.zeros((B, 0)).to(device)

        # Keep track of X and Y correspondences subset
        it_matches_ids = []
        dict_corr = {}

        for i_i in range(self.it_matches):

            try:
                sampled_idx = torch.multinomial(matches_row, self.num_samples_matches)
            except:
                print('[Except Reached]: Invalid matching matrix! ')
                break

            sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_idx_grd = (sampled_idx % num_kpts_grd)

            # # Sample the positions according to the sample ids
            X = self.metric_coord_sat_B[batch_idx, sampled_idx_sat, :]
            Y = self.metric_coord_grd_B[batch_idx, sampled_idx_grd, :]
            
            weights = matches_row[batch_idx, sampled_idx]

            dict_corr[i_i] = {'X': X, 'Y': Y, 'weights': weights}

            for kk in range(self.it_RANSAC):

                sampled_idx_ransac = torch.multinomial(weights, self.num_corr_2d_2d)

                X_k = X[batch_idx_ransac, sampled_idx_ransac, :]
                Y_k = Y[batch_idx_ransac, sampled_idx_ransac, :]
                weights_k = weights[batch_idx_ransac, sampled_idx_ransac]
                
                # get relative pose in grid space
                R, t, scale, ok_rank = weighted_procrustes_2d_with_scale(Y_k, X_k, use_weights=False)
                # R, t, scale, ok_rank = weighted_procrustes_2d_with_scale(Y_k, X_k, use_weights=True, use_mask=True, w=weights_k) 

                if not ok_rank:
                    continue

                invalid_t = (torch.isnan(t).any() or torch.isinf(t).any())
                invalid_R = (torch.isnan(R).any() or torch.isinf(R).any())

                if invalid_t or invalid_R:
                    continue

                # Compute hypothesis score
                score_k = soft_inlier_counting_bev_with_scale(Y, X, R, t, scale, th=self.th_soft_inlier)
                

                Rs = torch.cat([Rs, R.unsqueeze(1)], 1)
                ts = torch.cat([ts, t.unsqueeze(1)], 1)
                scales = torch.cat([scales, scale.unsqueeze(1)], 1)
                scores_ransac = torch.cat([scores_ransac, score_k], 1)
                it_matches_ids.append(i_i)
                num_valid_h += 1

        if num_valid_h > 0:
            max_ind = torch.argmax(scores_ransac, dim=1)
            R = Rs[batch_idx_ransac[:, 0], max_ind]
            t_metric = ts[batch_idx_ransac[:, 0], max_ind]
            scale_metric = scales[batch_idx_ransac[:, 0], max_ind]
            best_inliers = scores_ransac[batch_idx_ransac[:, 0], max_ind]

            # Use subset of correspondences that generated the hypothesis with maximum score
            X_best = torch.zeros_like(X)
            Y_best = torch.zeros_like(Y)
            for i_b in range(len(max_ind)):
                X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
            inliers_ref = torch.zeros((B, self.num_samples_matches)).to(device)

            # inliers:
            th_ref = self.num_refinements*[self.th_inlier]
            inliers_pre = self.num_corr_2d_2d * torch.ones_like(best_inliers)
            for i_ref in range(len(th_ref)):
                inliers = inlier_counting_bev_with_scale(Y_best, X_best, R, t_metric, scale_metric, th=th_ref[i_ref])

                do_ref = (inliers.sum(-1) >= self.num_corr_2d_2d) * (inliers.sum(-1) > inliers_pre)
                inliers_pre[do_ref] = inliers.sum(-1)[do_ref]

                # Check whether any refinements need to be done
                if (do_ref.sum().float() == 0.).item():
                    break
                inliers_ref[do_ref] = inliers[do_ref]
                R[do_ref], t_metric[do_ref], _, _ = weighted_procrustes_2d_with_scale(Y_best[do_ref], X_best[do_ref],
                                                                     use_weights=True, use_mask=True,
                                                                     check_rank=False,
                                                                     w=inliers_ref[do_ref])
            best_inliers = soft_inlier_counting_bev_with_scale(X_best, Y_best, R, t_metric, scale_metric, th=self.th_inlier)
        
        else:
            R = torch.zeros((B, 2, 2)).to(matches.device)
            t_metric = torch.zeros((B, 1, 2)).to(matches.device)
            scale_metric = torch.zeros((B, 1, 1)).to(matches.device)
            best_inliers = torch.zeros((B)).to(matches.device)
            
        inliers = None
        if return_inliers:
            if num_valid_h > 0:
    
                # Use subset of correspondences that generated the hypothesis with maximum score
                X_best = torch.zeros_like(X)
                Y_best = torch.zeros_like(Y)
                scale_best = torch.zeros_like(scale)
                weights_best = torch.zeros_like(weights)
                for i_b in range(len(max_ind)):
                    X_best[i_b], Y_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['X'][i_b], dict_corr[it_matches_ids[max_ind[i_b]]]['Y'][i_b]
                    scale_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['scales'][i_b]
                    weights_best[i_b] = dict_corr[it_matches_ids[max_ind[i_b]]]['weights'][i_b]
                    
                # Compute inliers from latest sampled set of correspondences
                inliers_idxs = inlier_counting_bev_with_scale(X_best, Y_best, R, t_metric, scale_best, th=self.th_inlier)
                inliers = []
                for idx_b in range(len(inliers_idxs)):
                    X_inliers = X_best[idx_b, inliers_idxs[idx_b]==1.]
                    Y_inliers = Y_best[idx_b, inliers_idxs[idx_b]==1.]
                    score_inliers = weights_best[idx_b, inliers_idxs[idx_b]==1.]
                    order_corr = torch.argsort(score_inliers, descending=True)
                    inliers_b = torch.cat([X_inliers[order_corr], Y_inliers[order_corr], score_inliers[order_corr].unsqueeze(-1)], dim=1)
                    inliers.append(inliers_b)
            
        return R, t_metric, scale_metric, best_inliers, inliers