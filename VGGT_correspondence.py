import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, project_world_points_to_cam
from plotting import create_interactive_correspondence_plot, make_matching_figure
import matplotlib.cm as cm

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["/home/landson/RGBT-Scenes/Building/rgb/test/img_249.jpg", "/home/landson/RGBT-Scenes/Building/rgb/test/img_257.jpg"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))
    

    print(point_map_by_unprojection.shape)  # (num_cameras, H, W, 3)
    print(extrinsic.shape)  # (num_cameras, 3, 4)
    print(intrinsic.shape)  # (num_cameras, 3, 3)

    print(extrinsic[:, 1].shape)  # (1, 3, 4)

    # Project 3D Points to Other Camera Views
    projected_points_12 = project_world_points_to_cam(torch.from_numpy(point_map_by_unprojection[0].reshape(-1, 3)).double().cuda(),
                                                    extrinsic[:, 1].double(),
                                                    intrinsic[:, 1].double())

    projected_points_21 = project_world_points_to_cam(torch.from_numpy(point_map_by_unprojection[1].reshape(-1, 3)).double().cuda(),
                                                    extrinsic[:, 0].double(),
                                                    intrinsic[:, 0].double())
    
    # # make a 3d scatter plot of the 3d points using matplotlib
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(point_map_by_unprojection[0][:, :, 0].flatten(), 
    #         point_map_by_unprojection[0][:, :, 1].flatten(), 
    #         point_map_by_unprojection[0][:, :, 2].flatten(), s=1)
    # plt.show()

    projected_points_12_2D = projected_points_12[0].cpu()
    projected_points_21_2D = projected_points_21[0].cpu()

    H, W = point_map_by_unprojection[0].shape[0], point_map_by_unprojection[0].shape[1]

pixel_map_12 = torch.round(projected_points_12_2D[0].reshape(point_map_by_unprojection[0].shape[0], point_map_by_unprojection[0].shape[1], 2)).int()
pixel_map_21 = torch.round(projected_points_21_2D[0].reshape(point_map_by_unprojection[1].shape[0], point_map_by_unprojection[1].shape[1], 2)).int()

# Compute Bijection of Pixel Maps
Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
XY = torch.stack([X, Y], axis=-1)  # shape (H, W, 2)

# Check bijection: map2[map1[i,j]] == (i,j)
u, v = pixel_map_12[..., 0], pixel_map_12[..., 1]
# Step 2: Create valid mask
valid_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)
# Step 3: Initialize output with NaNs
mapped_back = torch.full_like(pixel_map_12, -1)
# Step 4: Safely assign only valid indices
mapped_back[valid_mask] = pixel_map_21[v[valid_mask], u[valid_mask]]
# mapped_back = pixel_map_21[pixel_map_12[..., 0], pixel_map_12[..., 1]]  # shape (H, W, 2)
bijection_mask = torch.all(mapped_back == XY, axis=-1)
print(bijection_mask.sum(), " out of ", H*W, " pixels have bijection correspondence.")
print(mapped_back[193, 436])  # check specific pixel
# Extract valid (i,j) and their mapped (u,v)
valid_u1v1 = XY[bijection_mask]
valid_u2v2 = pixel_map_12[bijection_mask]


# Create new pixel maps with only bijection correspondences
valid_pixel_map_12 = torch.full_like(pixel_map_12, -1)
valid_pixel_map_21 = torch.full_like(pixel_map_21, -1)
valid_pixel_map_12[bijection_mask] = pixel_map_12[bijection_mask]
valid_pixel_map_21[bijection_mask] = pixel_map_21[bijection_mask]

# color = cm.jet(depth_conf[0, 0].cpu().numpy()[valid_u1v1[:,1], valid_u2v2[:,0]] / depth_conf[0, 0].cpu().numpy().max())
# make_matching_figure(
#     images[0, 0].permute(1, 2, 0).cpu().numpy(),
#     images[0, 1].permute(1, 2, 0).cpu().numpy(),
#     valid_u1v1.cpu().numpy(),
#     valid_u2v2.cpu().numpy(),
#     color=color,
# )

create_interactive_correspondence_plot(images[0, 0].permute(1, 2, 0).cpu(), images[0, 1].permute(1, 2, 0).cpu(), valid_pixel_map_12.numpy(), valid_pixel_map_21.numpy(), depth_conf.cpu().numpy())
