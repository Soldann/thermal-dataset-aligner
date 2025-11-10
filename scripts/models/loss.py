import torch
import torch.nn.functional as F
import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
import matplotlib.pyplot as plt

sat_bev_res = config.getint("Model", "sat_bev_res")

def scale_loss_log_l1(s_pred, s_gt, eps=1e-8):
    """
    Computes a robust scale loss using log-L1:
        loss = |log(s_pred) - log(s_gt)|
    
    Args:
        s_pred: Tensor of shape [B, 1, 1] — predicted scale from Procrustes
        s_gt:   Tensor of shape [B, 1, 1] — ground truth scale (used for data augmentation)
    
    Returns:
        Scalar tensor — average scale loss
    """
    s_pred_clamped = s_pred.clamp(min=eps)
    s_gt_clamped = s_gt.clamp(min=eps)
    loss = torch.abs(torch.log(s_pred_clamped) - torch.log(s_gt_clamped))
    return loss.mean()

def compute_infonce_weight(pose_loss, a=2.0, center=5.0):
    """
    pose_loss: scalar tensor
    a: sharpness of the sigmoid (2.0 is usually good)
    center: where to switch (~5.0)
    """
    return 1.0 / (1.0 + torch.exp(a * (pose_loss - center)))

def entropy_loss(matching_score, eps=1e-8):
    """
    matching_score: (B, N_g, N_s), already after bi-softmax normalization
    Returns: scalar entropy loss
    """
    # Row-wise entropy (Ground -> Satellite)
    row_entropy = - (matching_score * (matching_score + eps).log()).sum(dim=-1).mean()

    # Column-wise entropy (Satellite -> Ground)
    col_entropy = - (matching_score * (matching_score + eps).log()).sum(dim=-2).mean()

    # Total entropy loss
    return row_entropy + col_entropy

def mutual_nn_loss(matching_score):
    """
    matching_score: (B, N_g, N_s), after bi-softmax
    Returns: scalar mutual NN loss
    """
    # Find maximum score per ground point (row-wise max)
    row_max, _ = matching_score.max(dim=-1)  # (B, N_g)

    # Find maximum score per satellite point (column-wise max)
    col_max, _ = matching_score.max(dim=-2)  # (B, N_s)

    # Mutual NN loss: encourage high max in both directions
    # We want to maximize, so loss is negative
    return -0.5 * (row_max.mean() + col_max.mean())
    

def loss_bev_space(X0, Rgt, tgt, R, t):
    """
    Computes BEV reprojection loss between ground-truth and predicted transformations.

    Args:
        X0 (Tensor): Initial 3D coordinates [B, N, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].

    Returns:
        loss (Tensor): Mean reprojection error per batch.
    """
    B = X0.shape[0]

    # Transform points using ground-truth and predicted transformations
    X1_gt = Rgt @ X0.repeat(B,1,1).transpose(2, 1) + tgt.transpose(2, 1) 
    X1_pred = R @ X0.repeat(B,1,1).transpose(2, 1) + t.transpose(2, 1) 

    # Compute L2 distance for reprojection error
    loss = torch.mean(torch.sqrt(((X1_gt - X1_pred)**2).sum(dim=1)), dim=-1)

    return loss


def trans_l1_loss(t, tgt):
    """
    Computes L1 loss for translation vector.

    Args:
        t (Tensor): Predicted translation vector [B, 1, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].

    Returns:
        loss (Tensor): L1 loss for translation.
    """
    return torch.abs(t - tgt).sum(dim=-1)


def trans_l2_loss(t, tgt):
    """
    Computes L2 loss for translation vector.

    Args:
        t (Tensor): Predicted translation vector [B, 1, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].

    Returns:
        loss (Tensor): L2 loss for translation.
    """
    return ((t[:, :, :2] - tgt[:, :, :2]) ** 2).sum(dim=-1)

def translation_direction_loss(t_pred, t_gt, s_pred, eps=1e-8):
    """
    Translation loss: cosine similarity between predicted translation and GT (up to scale).
    
    Args:
        t_pred: Predicted translation [B, 1, 2]
        t_gt: Ground-truth translation [B, 1, 2]
        s_pred: Predicted scale [B, 1, 1]
        
    Returns:
        loss: scalar
    """

    # Remove scale from ground-truth
    s_pred_detached = s_pred.detach().squeeze(-1).squeeze(-1)  # [B]
    t_gt_scaled = t_gt.squeeze(1) / (s_pred_detached.unsqueeze(-1) + eps)  # [B, 2]

    # Normalize both translations
    t_pred_norm = F.normalize(t_pred.squeeze(1), dim=-1)  # [B, 2]
    t_gt_norm = F.normalize(t_gt_scaled, dim=-1)          # [B, 2]

    # Cosine similarity loss
    loss = 1 - (t_pred_norm * t_gt_norm).sum(dim=-1).mean()
    return loss

def rot_angle_loss(R, Rgt):
    """
    Computes rotation loss using residual rotation angle [radians].

    Args:
        R (Tensor): Predicted rotation matrix [B, 2, 2].
        Rgt (Tensor): Ground-truth rotation matrix [B, 2, 2].

    Returns:
        loss (Tensor): Rotation error (L1 loss).
        R_err (Tensor): Rotation error in radians.
    """
    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = torch.clip((trace - 1) / 2, -0.99999, 0.99999)  # Prevent NaNs
    R_err = torch.acos(cosine)
    return R_err.mean()


def compute_pose_loss(R, t, Rgt, tgt, soft_clipping=True):
    """
    Computes total pose estimation loss (rotation + translation).

    Args:
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        soft_clipping (bool): Whether to apply soft clipping using tanh.

    Returns:
        loss (Tensor): Combined loss.
        loss_rot (Tensor): Rotation loss.
        loss_trans (Tensor): Translation loss.
    """
    loss_rot, rot_err = rot_angle_loss(R, Rgt)
    loss_trans = trans_l1_loss(t, tgt)

    if soft_clipping:
        loss = torch.tanh(loss_rot / 0.9) + torch.tanh(loss_trans / 0.9)
    else:
        loss = loss_rot + loss_trans

    return loss, loss_rot, loss_trans


def vcre_loss(R, t, Tgt, K0, H=720):
    """
    Computes Virtual Correspondences Reprojection Error (VCRE).

    Args:
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].
        Tgt (Tensor): Ground-truth transformation matrix [B, 4, 4].
        K0 (Tensor): Intrinsic camera matrix.
        H (int): Image height.

    Returns:
        repr_err (Tensor): Reprojection error per batch.
    """
    B = R.shape[0]
    Rgt, tgt = Tgt[:, :3, :3], Tgt[:, :3, 3:].transpose(1, 2)

    eye_coords = torch.from_numpy(eye_coords_glob).to(R.device, dtype=torch.float32).unsqueeze(0)[:, :, :3]
    eye_coords = eye_coords.expand(B, -1, -1)

    # Ground-truth 2D projections
    uv_gt = project_2d(eye_coords, K0)

    # Predicted transformations
    eye_coord_tmp = (R @ eye_coords.transpose(2, 1)) + t.transpose(2, 1)
    eyes_residual = (Rgt.transpose(2, 1) @ eye_coord_tmp - Rgt.transpose(2, 1) @ tgt.transpose(2, 1)).transpose(2, 1)

    uv_pred = project_2d(eyes_residual, K0)

    # Clip values to prevent invalid pixel locations
    uv_gt = torch.clip(uv_gt, 0, H)
    uv_pred = torch.clip(uv_pred, 0, H)

    # Compute reprojection error
    repr_err = torch.mean(torch.norm(uv_gt - uv_pred, dim=-1), dim=-1, keepdim=True)

    return repr_err


def compute_vcre_loss(R, t, Rgt, tgt, K=None, soft_clipping=True):
    """
    Computes Virtual Correspondences Reprojection Error (VCRE) loss.

    Args:
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        K (Tensor, optional): Camera intrinsic matrix.
        soft_clipping (bool): Whether to apply soft clipping using tanh.

    Returns:
        loss (Tensor): VCRE loss.
        loss_rot (Tensor): Rotation loss.
        loss_trans (Tensor): Translation loss.
    """
    B = R.shape[0]
    Tgt = torch.zeros((B, 4, 4), device=R.device, dtype=torch.float32)
    Tgt[:, :3, :3] = Rgt
    Tgt[:, :3, 3:] = tgt.transpose(2, 1)

    loss = vcre_loss(R, t, Tgt, K)

    if soft_clipping:
        loss = torch.tanh(loss / 80)

    loss_rot, rot_err = rot_angle_loss(R, Rgt)
    loss_trans = trans_l1_loss(t, tgt)

    return loss, loss_rot, loss_trans

def compute_similarity_loss(Rgt, tgt, sat_desc, grd_desc, sat_points_selected, grd_points_selected, sat_indices_sampled, grd_indices_sampled, coord_sat, coord_grd):
    """
    Compute similarity loss between satellite and ground features after transformation.

    Args:
        Rgt (Tensor): Ground-truth rotation matrix (B, 2, 2).
        tgt (Tensor): Ground-truth translation vector (B, 1, 2).
        sat_desc (Tensor): Satellite descriptors (B, C, N).
        grd_desc (Tensor): Ground descriptors (B, C, M).
        sat_points_selected (Tensor): Selected satellite points (B, P, 2).
        grd_points_selected (Tensor): Selected ground points (B, P, 2).
        sat_indices_sampled (Tensor): Sampled satellite indices (B, P).
        grd_indices_sampled (Tensor): Sampled ground indices (B, P).
        coord_sat (Tensor): Sampled ground indices (B, sat_bev_res*sat_bev_res, 2).
        coord_grd (Tensor): Sampled ground indices (B, grd_bev_res*grd_bev_res, 2).

    Returns:
        similarity_loss (Tensor): Similarity loss value per batch (B,).
    """
    
    B, num_points, _ = sat_points_selected.shape
    device = sat_desc.device  # Ensure everything is on the correct device

    # Add homogeneous coordinates (convert 2D to 3D)
    ones = torch.ones(B, num_points, 1, device=device)
    sat_points_h = torch.cat((sat_points_selected, ones), dim=-1)  # (B, P, 3)
    grd_points_h = torch.cat((grd_points_selected, ones), dim=-1)  # (B, P, 3)

    # Construct transformation matrix (B, 3, 3)
    T_s2g = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_s2g[:, :2, :2] = Rgt
    T_s2g[:, :2, 2] = tgt[:, 0, :]

    # ------------------------
    # Satellite to Ground Mapping
    # ------------------------
    grd_points_mapped = (T_s2g @ sat_points_h.transpose(2, 1)).permute(0, 2, 1)
    grd_points_mapped[..., :2] /= grd_points_mapped[..., 2:3]  # Normalize by homogeneous coord
    grd_points_mapped = grd_points_mapped[..., :2]  # (B, P, 2)

    # Find nearest ground metric coordinates
    distances = torch.cdist(grd_points_mapped, coord_grd)  # (B, P, R)
    min_distances, grd_indices_mapped = distances.min(dim=-1)  # (B, P)

    keep_index = min_distances <= (grid_size_h / grd_bev_res)  # Mask valid matches

    # Compute similarity loss for Satellite-to-Ground (S2G)
    similarity_loss_s2g = torch.zeros(B, device=device)
    for b in range(B):
        if keep_index[b].sum() > 0:
            sat_indices_kept = sat_indices_sampled[b, keep_index[b]]
            grd_indices_kept = grd_indices_mapped[b, keep_index[b]]

            desc_similarity = torch.matmul(sat_desc[b, :, sat_indices_kept].T, grd_desc[b, :, grd_indices_kept])
            similarity_loss_s2g[b] = keep_index[b].sum() - torch.diagonal(desc_similarity).sum()

    # print('sat_points_selected', sat_points_selected)
    # print('grd_points_mapped', grd_points_mapped)
    # print('coord_grd[0,grd_indices_mapped[0,0],:]', coord_grd[0,grd_indices_mapped[0,0],:])
    
    # ------------------------
    # Ground to Satellite Mapping
    # ------------------------
    T_g2s = torch.linalg.inv(T_s2g)  # Inverse transformation (Ground -> Satellite)
    sat_points_mapped = (T_g2s @ grd_points_h.transpose(2, 1)).permute(0, 2, 1)
    sat_points_mapped[..., :2] /= sat_points_mapped[..., 2:3]  # Normalize by homogeneous coord
    sat_points_mapped = sat_points_mapped[..., :2]  # (B, P, 2)
    
    # Find nearest satellite metric coordinates
    distances = torch.cdist(sat_points_mapped, coord_sat)  # (B, P, R)
    min_distances, sat_indices_mapped = distances.min(dim=-1)  # (B, P)
    keep_index = min_distances <= (grid_size_h / sat_bev_res)  # Mask valid matches
    
    # print('grd_points_selected', grd_points_selected)
    # print('sat_points_mapped', sat_points_mapped)
    # print('coord_sat[0,sat_indices_mapped[0,0],:]', coord_sat[0,sat_indices_mapped[0,0],:])

    # Compute similarity loss for Ground-to-Satellite (G2S)
    similarity_loss_g2s = torch.zeros(B, device=device)
    for b in range(B):
        if keep_index[b].sum() > 0:
            sat_indices_kept = sat_indices_mapped[b, keep_index[b]]
            grd_indices_kept = grd_indices_sampled[b, keep_index[b]]

            desc_similarity = torch.matmul(sat_desc[b, :, sat_indices_kept].T, grd_desc[b, :, grd_indices_kept])
            similarity_loss_g2s[b] = keep_index[b].sum() - torch.diagonal(desc_similarity).sum()

    # Return average loss
    # return (similarity_loss_s2g + similarity_loss_g2s) / 2
    return similarity_loss_s2g


def compute_infonce_loss_match_all(
    Rgt, tgt,
    sat_points_selected, grd_points_selected,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    mask, grid_size_h
):
    
    B, num_sat, num_grd = matching_score_original.shape
    
    matches_row = matching_score_original.flatten(1)


    # scale = torch.ones_like(scale)
    ### satellite to ground
    grd_points_mapped = (sat_points_selected - tgt) @ Rgt 

    distances = torch.cdist(grd_points_mapped, grd_coord)  # (B, P, R)
    min_distances, col_mapped = distances.min(dim=-1)               # (B, P)

    infoNCE_loss_s2g = torch.zeros(B)
    for b in range(B):
        keep_index_b_distance = min_distances[b] <= (grid_size_h / sat_bev_res) 
        
        keep_index_b_depth = mask[b][col_mapped[b]]
        keep_index_b = keep_index_b_distance.squeeze(0) * keep_index_b_depth

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_row = sampled_row[b, valid_indices]
            valid_col_mapped = col_mapped[b, valid_indices]

            unique, idx, counts = torch.unique(valid_row, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score_original.device), cum_sum[:-1]))
            unique_row = valid_row[ind_sorted[cum_sum]]
            unique_col_mapped = valid_col_mapped[ind_sorted[cum_sum]]

            selected_matching_indices = unique_row*num_grd + unique_col_mapped
            
            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            demoninator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]), dim=1)
            
            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / demoninator))

    ### ground to satellite
    
    sat_points_mapped = grd_points_selected @ Rgt.transpose(1, 2) + tgt
    
    distances = torch.cdist(sat_points_mapped, sat_coord)
    min_distances, row_mapped = distances.min(dim=-1)
        
    infoNCE_loss_g2s = torch.zeros(B)
    for b in range(B):
        keep_index_b = min_distances[b] <= (grid_size_h / sat_bev_res)

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_col = sampled_col[b, valid_indices]
            valid_row_mapped = row_mapped[b, valid_indices]
            
            
            unique, idx, counts = torch.unique(valid_col, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score_original.device), cum_sum[:-1]))
            unique_col = valid_col[ind_sorted[cum_sum]]
            unique_row_mapped = valid_row_mapped[ind_sorted[cum_sum]]

            
            selected_matching_indices = unique_row_mapped*num_grd + unique_col
            
            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            demoninator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)
            
            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / demoninator))
            
    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2



def compute_infonce_loss_match_all_with_scale(
    Rgt, tgt,
    sat_points_selected, grd_points_selected, scale,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    mask, grid_size_h
):

    B, num_sat, num_grd = matching_score_original.shape
    scale = scale.to(torch.float32)
    scale = scale.detach()

    matches_row = matching_score_original.flatten(1)

    ### satellite to ground
    grd_points_mapped = ((sat_points_selected - tgt) / scale) @ Rgt 

    distances = torch.cdist(grd_points_mapped, grd_coord)  # (B, P, R)
    min_distances, col_mapped = distances.min(dim=-1)               # (B, P)

    infoNCE_loss_s2g = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b_distance = min_distances[b] <= (grid_size_h / sat_bev_res) / scale[b] 
        keep_index_b_depth = mask[b][col_mapped[b]]
        # print('mask[b]', mask[b].shape)
        keep_index_b = keep_index_b_distance.squeeze(0) * keep_index_b_depth

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_row = sampled_row[b, valid_indices]
            valid_col_mapped = col_mapped[b, valid_indices]

            unique, idx, counts = torch.unique(valid_row, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score_original.device), cum_sum[:-1]))
            unique_row = valid_row[ind_sorted[cum_sum]]
            unique_col_mapped = valid_col_mapped[ind_sorted[cum_sum]]

            selected_matching_indices = unique_row*num_grd + unique_col_mapped

            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            demoninator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]), dim=1)
            
            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / demoninator))

    ### ground to satellite
    
    sat_points_mapped = scale * (grd_points_selected @ Rgt.transpose(1, 2)) + tgt
    
    distances = torch.cdist(sat_points_mapped, sat_coord)
    min_distances, row_mapped = distances.min(dim=-1)
        
    infoNCE_loss_g2s = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b = min_distances[b] <= (grid_size_h / sat_bev_res)

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_col = sampled_col[b, valid_indices]
            valid_row_mapped = row_mapped[b, valid_indices]
            
            unique, idx, counts = torch.unique(valid_col, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score_original.device), cum_sum[:-1]))
            unique_col = valid_col[ind_sorted[cum_sum]]
            unique_row_mapped = valid_row_mapped[ind_sorted[cum_sum]]

            
            selected_matching_indices = unique_row_mapped*num_grd + unique_col
            
            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            demoninator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)
            
            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / demoninator))
            
    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2

def compute_infonce_loss_match_all_with_scale_select_negatives(
    Rgt, tgt,
    sat_points_selected, grd_points_selected, scale,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    mask, grid_size_h
):

    B, num_sat, num_grd = matching_score_original.shape
    scale = scale.to(torch.float32)

    matches_row = matching_score_original.flatten(1)
    # scale = scale.detach()
    # scale = torch.ones_like(scale)

    ### satellite to ground
    grd_points_mapped = ((sat_points_selected - tgt) / scale) @ Rgt 

    distances = torch.cdist(grd_points_mapped, grd_coord)  # (B, P, R)
    min_distances, col_mapped = distances.min(dim=-1)               # (B, P)

    grd_pointwise_distance = torch.cdist(grd_coord, grd_coord)
    

    infoNCE_loss_s2g = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b_distance = min_distances[b] <= (grid_size_h / sat_bev_res) / scale[b] 
        keep_index_b_depth = mask[b][col_mapped[b]]
        # print('mask[b]', mask[b].shape)
        keep_index_b = keep_index_b_distance.squeeze(0) * keep_index_b_depth

        negative_mask = grd_pointwise_distance[b] > 0.05 # (grid_size_h / sat_bev_res) / scale[b] / 10

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_row = sampled_row[b, valid_indices]
            valid_col_mapped = col_mapped[b, valid_indices]

            unique, idx, counts = torch.unique(valid_row, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score_original.device), cum_sum[:-1]))
            unique_row = valid_row[ind_sorted[cum_sum]]
            unique_col_mapped = valid_col_mapped[ind_sorted[cum_sum]]
            
            negative_mask_for_unique_row = (negative_mask[unique_col_mapped]).to(torch.float32)

            selected_matching_indices = unique_row*num_grd + unique_col_mapped
            
            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            demoninator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]) * negative_mask_for_unique_row, dim=1) + positives
            
            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / demoninator))

    ### ground to satellite
    
    sat_points_mapped = scale * (grd_points_selected @ Rgt.transpose(1, 2)) + tgt
    
    distances = torch.cdist(sat_points_mapped, sat_coord)
    min_distances, row_mapped = distances.min(dim=-1)
        
    infoNCE_loss_g2s = torch.zeros(B, device=matching_score_original.device)
    for b in range(B):
        keep_index_b = min_distances[b] <= (grid_size_h / sat_bev_res)

        if keep_index_b.sum() > 0:
            valid_indices = keep_index_b.nonzero(as_tuple=True)[0]
            valid_col = sampled_col[b, valid_indices]
            valid_row_mapped = row_mapped[b, valid_indices]
            
            unique, idx, counts = torch.unique(valid_col, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat((torch.tensor([0]).to(matching_score_original.device), cum_sum[:-1]))
            unique_col = valid_col[ind_sorted[cum_sum]]
            unique_row_mapped = valid_row_mapped[ind_sorted[cum_sum]]

            
            selected_matching_indices = unique_row_mapped*num_grd + unique_col
            
            positives = torch.exp(torch.index_select(matches_row[b], 0, selected_matching_indices))
            demoninator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)
            
            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / demoninator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2


def compute_infonce_loss_direction_only(
    Rgt, tgt,
    sat_points_selected, grd_points_selected,
    sampled_row, sampled_col,
    matching_score_original,
    sat_coord, grd_coord,
    angle_threshold_deg=10.0,
    eps=1e-6
):
    
    B, num_sat, num_grd = matching_score_original.shape
    matching_score_exp = torch.exp(matching_score_original)

    angle_threshold_rad = angle_threshold_deg * torch.pi / 180.0

    ### Satellite to Ground
    infoNCE_loss_s2g = torch.zeros(B, device=matching_score_original.device)

    sat_vectors = (sat_points_selected - tgt) @ Rgt 
    sat_vectors = sat_vectors / (sat_vectors.norm(dim=-1, keepdim=True) + eps)

    grd_vectors = F.normalize(grd_coord, dim=-1)
    grd_vectors = grd_vectors / (grd_vectors.norm(dim=-1, keepdim=True) + eps)

    # Cosine similarity between satellite and ground directions
    cos_sim = sat_vectors @ grd_vectors.transpose(1, 2)
    angles = torch.acos(torch.clamp(cos_sim, -1 + eps, 1 - eps))

    for b in range(B):        

        positive_mask = (angles[b] <= angle_threshold_rad) 
        negative_mask = ~positive_mask

        if positive_mask.sum() > 0:
            sampled_row_b = sampled_row[b]
            
            pos_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)
            neg_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)

            # Scatter positive_mask into new_mask
            pos_index_mask[sampled_row_b] = positive_mask
            neg_index_mask[sampled_row_b] = negative_mask

            pos_scores = (pos_index_mask * matching_score_exp[b])[torch.unique(sampled_row_b), :]
            neg_scores = (neg_index_mask * matching_score_exp[b])[torch.unique(sampled_row_b), :]
            
            
            positives, _ = torch.max(pos_scores, dim=1)
            negatives = torch.sum(neg_scores, dim=1)
            
            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / (positives + negatives)))
        else:
            infoNCE_loss_s2g[b] = 0.0
        
    ### Ground to Satellite
    infoNCE_loss_g2s = torch.zeros(B, device=matching_score_original.device)
    grd_vectors = grd_points_selected @ Rgt.transpose(1, 2)
    grd_vectors = grd_vectors / (grd_vectors.norm(dim=-1, keepdim=True) + eps)

    # Compute satellite vectors from GT translation
    sat_vectors = sat_coord - tgt
    sat_vectors = sat_vectors / (sat_vectors.norm(dim=-1, keepdim=True) + eps)

    cos_sim = sat_vectors @ grd_vectors.transpose(1, 2)
    angles = torch.acos(torch.clamp(cos_sim, -1 + eps, 1 - eps))

    for b in range(B):
        positive_mask = (angles[b] <= angle_threshold_rad) 
        negative_mask = ~positive_mask

        all_zero_col_mask = positive_mask.sum(dim=0) == 0  
        
        if positive_mask.sum() > 0:
            sampled_col_b = sampled_col[b]
                                    
            pos_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)
            neg_index_mask = torch.zeros((num_sat, num_grd), dtype=positive_mask.dtype, device=positive_mask.device)

            pos_index_mask[:, sampled_col_b] = positive_mask  
            neg_index_mask[:, sampled_col_b] = negative_mask
            
            pos_scores = (pos_index_mask * matching_score_exp[b])[:, torch.unique(sampled_col_b[~all_zero_col_mask])]
            neg_scores = (neg_index_mask * matching_score_exp[b])[:, torch.unique(sampled_col_b[~all_zero_col_mask])]
            
            positives, _ = torch.max(pos_scores, dim=0)
            negatives = torch.sum(neg_scores, dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / (positives + negatives)))
        else:
            infoNCE_loss_g2s[b] = 0.0

    return (infoNCE_loss_s2g.mean() + infoNCE_loss_g2s.mean()) / 2




def compute_infonce_loss(
    Rgt, tgt,
    sat_points_selected, grd_points_selected,
    sampled_row, sampled_col,
    sat_indices_topk, grd_indices_topk,
    sat_keypoint_coord, grd_keypoint_coord,
    matching_score_original
):
    """
    Compute InfoNCE loss between satellite and ground features after transformation.

    Args:
        Rgt (Tensor): Ground-truth rotation matrix (B, 2, 2).
        tgt (Tensor): Ground-truth translation vector (B, 1, 2).
        sat_points_selected (Tensor): Selected satellite points (B, P, 2).
        grd_points_selected (Tensor): Selected ground points (B, P, 2).
        sampled_row (Tensor): Ground descriptor row indices for sampled points (B, P).
        sampled_col (Tensor): Satellite descriptor column indices for sampled points (B, P).
        sat_indices_topk, grd_indices_topk: Not used, but assumed available for expansion.
        sat_keypoint_coord (Tensor): Satellite keypoint coordinates (B, R, 2).
        grd_keypoint_coord (Tensor): Ground keypoint coordinates (B, R, 2).
        matching_score_original (Tensor): Original matching score map (B, M, N).

    Returns:
        Tensor: InfoNCE similarity loss (B,)
    """
    B, P, _ = sat_points_selected.shape
    device = matching_score_original.device

    # Homogeneous coordinates
    ones = torch.ones(B, P, 1, device=device)
    sat_points_h = torch.cat([sat_points_selected, ones], dim=-1)  # (B, P, 3)
    grd_points_h = torch.cat([grd_points_selected, ones], dim=-1)  # (B, P, 3)

    # Satellite to Ground Transformation
    T_s2g = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    T_s2g[:, :2, :2] = Rgt
    T_s2g[:, :2, 2] = tgt[:, 0, :]

    # ------------------------
    # Satellite → Ground
    # ------------------------
    grd_points_mapped = (T_s2g @ sat_points_h.transpose(2, 1)).permute(0, 2, 1)
    grd_points_mapped[..., :2] /= grd_points_mapped[..., 2:3]
    grd_points_mapped = grd_points_mapped[..., :2]

    distances = torch.cdist(grd_points_mapped, grd_keypoint_coord)  # (B, P, R)
    min_distances, col_mapped = distances.min(dim=-1)               # (B, P)
    keep_index = min_distances <= (grid_size_h / grd_bev_res)

    infoNCE_loss_s2g = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = sampled_row[b, keep_index[b]]
            col_kept = col_mapped[b, keep_index[b]]

            unique, idx, counts = torch.unique(row_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            unique_row = row_kept[ind_sorted[cum_sum]]
            coorespond_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = unique_row * num_keypoints + coorespond_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]), dim=1)

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    # ------------------------
    # Ground → Satellite
    # ------------------------
    T_g2s = torch.linalg.inv(T_s2g)
    sat_points_mapped = (T_g2s @ grd_points_h.transpose(2, 1)).permute(0, 2, 1)
    sat_points_mapped[..., :2] /= sat_points_mapped[..., 2:3]
    sat_points_mapped = sat_points_mapped[..., :2]

    distances = torch.cdist(sat_points_mapped, sat_keypoint_coord)
    min_distances, row_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / sat_bev_res)

    infoNCE_loss_g2s = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = row_mapped[b, keep_index[b]]
            col_kept = sampled_col[b, keep_index[b]]

            unique, idx, counts = torch.unique(col_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            coorespond_row = row_kept[ind_sorted[cum_sum]]
            unique_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = coorespond_row * num_keypoints + unique_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2

def compute_infonce_loss_with_mask(
    Rgt, tgt,
    sat_points_selected, grd_points_selected, scale,
    sampled_row, sampled_col,
    sat_indices_topk, grd_indices_topk,
    sat_keypoint_coord, grd_keypoint_coord,
    matching_score_original,
    sampled_mask
):
    """
    Compute InfoNCE loss between satellite and ground features after transformation.

    Args:
        Rgt (Tensor): Ground-truth rotation matrix (B, 2, 2).
        tgt (Tensor): Ground-truth translation vector (B, 1, 2).
        sat_points_selected (Tensor): Selected satellite points (B, P, 2).
        grd_points_selected (Tensor): Selected ground points (B, P, 2).
        sampled_row (Tensor): Ground descriptor row indices for sampled points (B, P).
        sampled_col (Tensor): Satellite descriptor column indices for sampled points (B, P).
        sat_indices_topk, grd_indices_topk: Not used, but assumed available for expansion.
        sat_keypoint_coord (Tensor): Satellite keypoint coordinates (B, R, 2).
        grd_keypoint_coord (Tensor): Ground keypoint coordinates (B, R, 2).
        matching_score_original (Tensor): Original matching score map (B, M, N).

    Returns:
        Tensor: InfoNCE similarity loss (B,)
    """
    B, P, _ = sat_points_selected.shape
    device = matching_score_original.device

    

    # ------------------------
    # Satellite → Ground
    # ------------------------
    grd_points_mapped = scale * sat_points_selected @ Rgt.transpose(1, 2) + tgt

    distances = torch.cdist(grd_points_mapped, grd_keypoint_coord)  # (B, P, R)
    min_distances, col_mapped = distances.min(dim=-1)               # (B, P)
    keep_index = min_distances <= (grid_size_h / sat_bev_res) 

    
    keep_index = keep_index * sampled_mask
    # print('keep_index', keep_index.shape, keep_index.sum())
    # print('sampled_row', sampled_row.shape)

    infoNCE_loss_s2g = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = sampled_row[b, keep_index[b]]
            col_kept = col_mapped[b, keep_index[b]]

            unique, idx, counts = torch.unique(row_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            unique_row = row_kept[ind_sorted[cum_sum]]
            coorespond_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = unique_row * num_keypoints + coorespond_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, unique_row, :]), dim=1)

            infoNCE_loss_s2g[b] = -torch.mean(torch.log(positives / denominator))

    # ------------------------
    # Ground → Satellite
    # ------------------------
    sat_points_mapped = ((grd_points_selected - tgt) @ Rgt) / scale

    distances = torch.cdist(sat_points_mapped, sat_keypoint_coord)
    min_distances, row_mapped = distances.min(dim=-1)
    keep_index = min_distances <= (grid_size_h / sat_bev_res)
    keep_index = keep_index * sampled_mask


    infoNCE_loss_g2s = torch.zeros(B, device=device)

    for b in range(B):
        if keep_index[b].sum() > 0:
            row_kept = row_mapped[b, keep_index[b]]
            col_kept = sampled_col[b, keep_index[b]]

            unique, idx, counts = torch.unique(col_kept, sorted=True, return_inverse=True, return_counts=True)
            _, ind_sorted = torch.sort(idx, stable=True)
            cum_sum = counts.cumsum(0)
            cum_sum = torch.cat([torch.tensor([0], device=device), cum_sum[:-1]])
            coorespond_row = row_kept[ind_sorted[cum_sum]]
            unique_col = col_kept[ind_sorted[cum_sum]]

            selected_indices = coorespond_row * num_keypoints + unique_col
            positives = torch.exp(torch.index_select(matching_score_original[b].flatten(), 0, selected_indices))
            denominator = torch.sum(torch.exp(matching_score_original[b, :, unique_col]), dim=0)

            infoNCE_loss_g2s[b] = -torch.mean(torch.log(positives / denominator))

    return (infoNCE_loss_s2g + infoNCE_loss_g2s) / 2

def topology_direction_loss(Rgt, tgt,
    sat_points_selected, grd_points_selected, w, eps=1e-6):

    B, N, _ = sat_points_selected.shape

    sat_points_selected = sat_points_selected - tgt
    grd_points_selected = grd_points_selected @ Rgt.transpose(1, 2)


    # topology
    W1 = torch.abs(w).sum(1, keepdim=True)
    w_norm = (w / (W1 + eps)).unsqueeze(-1)  # [B, N, 1]
    
    sat_mean = (w_norm * sat_points_selected).sum(1, keepdim=True)  # [B, 1, 2]
    grd_mean = (w_norm * grd_points_selected).sum(1, keepdim=True)
    sat_centered = sat_points_selected - sat_mean
    grd_centered = grd_points_selected - grd_mean

    sat_norm = sat_centered / (sat_centered.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1) + eps)  # (B, N, 2)
    grd_norm = grd_centered / (grd_centered.norm(dim=2).mean(dim=1, keepdim=True).unsqueeze(-1) + eps)  # (B, N, 2)
    
    D_sat = torch.cdist(sat_norm, sat_norm)  # (B, N, N)
    D_grd = torch.cdist(grd_norm, grd_norm)  # (B, N, N)

    w_pair = torch.bmm(w.squeeze(-1).unsqueeze(2), w.squeeze(-1).unsqueeze(1))  # (B, N, N), outer product of weights
    topo_error = (D_sat - D_grd) ** 2  # (B, N, N)

    weighted_topo_error = (w_pair * topo_error).sum(dim=[1,2]) / (w_pair.sum(dim=[1,2]) + eps)  # (B,)
    topology_loss = weighted_topo_error.mean()  # scalar over batch

    # direction
    # v_sat = F.normalize(sat_points_selected, dim=-1, eps=eps)
    # v_grd = F.normalize(grd_points_selected, dim=-1, eps=eps)
    v_sat = sat_points_selected / (sat_points_selected.norm(dim=-1, keepdim=True) + eps)
    v_grd = grd_points_selected / (grd_points_selected.norm(dim=-1, keepdim=True) + eps)
    cos_sim = (v_sat * v_grd).sum(dim=-1)  # (B, N)

    direction_error = 1 - cos_sim  # (B, N)    
    weighted_direction_error = (w * direction_error).sum(dim=1) / (w.sum(dim=1) + eps)  # (B,)
    direction_loss = weighted_direction_error.mean()

    
    return topology_loss, direction_loss

def topology_ratio_direction_loss(Rgt, tgt,
    sat_points_selected, grd_points_selected, w, eps=1e-6, num_triplets=1000):

    B, N, _ = sat_points_selected.shape

    sat_points_selected = sat_points_selected - tgt
    grd_points_selected = grd_points_selected @ Rgt.transpose(1, 2)


    # topology
    # Normalize weights (B, N)
    w = w.squeeze(-1)
    w_probs = w / (w.sum(dim=1, keepdim=True) + eps)

    ratio_errors = []

    for b in range(B):
        x_sat = sat_points_selected[b]  # (N, 2)
        x_grd = grd_points_selected[b]  # (N, 2)
        w_b = w_probs[b]                # (N,)

        # Sample triplets (i, j, k)
        i_idx = torch.multinomial(w_b, num_triplets, replacement=True)
        j_idx = torch.multinomial(w_b, num_triplets, replacement=True)
        k_idx = torch.multinomial(w_b, num_triplets, replacement=True)

        # Filter valid triplets (i ≠ j ≠ k)
        mask_valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        if mask_valid.sum() == 0:
            continue  # skip if no valid triplets
        i_idx = i_idx[mask_valid]
        j_idx = j_idx[mask_valid]
        k_idx = k_idx[mask_valid]

        xi_sat, xj_sat, xk_sat = x_sat[i_idx], x_sat[j_idx], x_sat[k_idx]
        xi_grd, xj_grd, xk_grd = x_grd[i_idx], x_grd[j_idx], x_grd[k_idx]

        # Compute distances
        d_ij_sat = (xi_sat - xj_sat).norm(dim=-1)
        d_ik_sat = (xi_sat - xk_sat).norm(dim=-1)
        d_ij_grd = (xi_grd - xj_grd).norm(dim=-1)
        d_ik_grd = (xi_grd - xk_grd).norm(dim=-1)

        # Compute ratios
        ratio_sat = d_ij_sat / (d_ik_sat + eps)
        ratio_grd = d_ij_grd / (d_ik_grd + eps)

        # L2 error on ratio difference
        ratio_error = (ratio_sat - ratio_grd) ** 2
        ratio_errors.append(ratio_error.mean())

    if len(ratio_errors) > 0:
        topology_loss = torch.stack(ratio_errors).mean()
    else:
        topology_loss = torch.tensor(0.0, device=sat_points_selected.device)

    # direction
    # v_sat = F.normalize(sat_points_selected, dim=-1, eps=eps)
    # v_grd = F.normalize(grd_points_selected, dim=-1, eps=eps)
    v_sat = sat_points_selected / (sat_points_selected.norm(dim=-1, keepdim=True) + eps)
    v_grd = grd_points_selected / (grd_points_selected.norm(dim=-1, keepdim=True) + eps)
    cos_sim = (v_sat * v_grd).sum(dim=-1)  # (B, N)

    direction_error = 1 - cos_sim  # (B, N)    
    weighted_direction_error = (w * direction_error).sum(dim=1) / (w.sum(dim=1) + eps)  # (B,)
    direction_loss = weighted_direction_error.mean()

    
    return topology_loss, direction_loss