import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
from typing import Tuple, Optional
from fused_ssim import fused_ssim

def imagegs_loss_fn(pred: torch.Tensor, target: torch.Tensor):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    return F.l1_loss(pred, target) +  0.1 * fused_ssim(pred, target)

def compute_gradient_pytorch(img_chunk: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute image gradients using PyTorch convolution (no cv2 dependency)"""
    # Sobel kernels for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], 
                           [-2, 0, 2], 
                           [-1, 0, 1]], 
                          dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    sobel_y = torch.tensor([[-1, -2, -1], 
                           [ 0,  0,  0], 
                           [ 1,  2,  1]], 
                          dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
    # Add batch and channel dimensions for conv2d
    img_chunk = img_chunk.unsqueeze(0).unsqueeze(0)
    
    # Compute gradients using conv2d
    grad_x = torch.nn.functional.conv2d(img_chunk, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(img_chunk, sobel_y, padding=1)
    
    # Remove batch and channel dimensions
    return grad_x.squeeze(), grad_y.squeeze()


def sample_points_pytorch(prob_map: torch.Tensor, num_points: int, H: int, W: int, 
                         device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Efficiently sample points using hierarchical sampling to avoid multinomial limits"""
    total_pixels = H * W
    
    # Check if we need hierarchical sampling (multinomial limit is 2^24)
    if total_pixels > 2**24:
        return sample_points_hierarchical(prob_map, num_points, H, W, device)
    else:
        # Use direct multinomial sampling for smaller images
        prob_flat = prob_map.flatten()
        sampled_indices = torch.multinomial(prob_flat, num_points, replacement=False)
        
        # Convert flat indices to 2D coordinates
        h_coords = (sampled_indices // W).float() / H
        w_coords = (sampled_indices % W).float() / W
        
        # Stack coordinates [x, y] format (normalized to [0,1])
        sampled_coords = torch.stack([w_coords, h_coords], dim=1)
        
        return sampled_indices, sampled_coords


def sample_points_hierarchical(prob_map: torch.Tensor, num_points: int, H: int, W: int, 
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hierarchical sampling for large images that exceed multinomial limits"""
    # Step 1: Downsample probability map for coarse sampling
    downsample_factor = 8  # Reduce resolution by 8x
    H_coarse = H // downsample_factor
    W_coarse = W // downsample_factor
    
    # Downsample using average pooling
    prob_map_coarse = torch.nn.functional.avg_pool2d(
        prob_map.unsqueeze(0).unsqueeze(0), 
        kernel_size=downsample_factor, 
        stride=downsample_factor
    ).squeeze()
    
    # Step 2: Sample coarse locations
    prob_flat_coarse = prob_map_coarse.flatten()
    
    # Sample more coarse points than needed for refinement
    num_coarse_points = min(num_points * 4, len(prob_flat_coarse))
    coarse_indices = torch.multinomial(prob_flat_coarse, num_coarse_points, replacement=False)
    
    # Step 3: Refine sampling in local neighborhoods
    sampled_indices_list = []
    sampled_coords_list = []
    
    points_per_coarse = num_points // num_coarse_points + 1
    neighborhood_size = downsample_factor * 2  # Local neighborhood size
    
    for coarse_idx in coarse_indices:
        if len(sampled_indices_list) >= num_points:
            break
            
        # Convert coarse index to coarse coordinates
        coarse_h = coarse_idx // W_coarse
        coarse_w = coarse_idx % W_coarse
        
        # Map to original image coordinates
        center_h = coarse_h * downsample_factor
        center_w = coarse_w * downsample_factor
        
        # Define local neighborhood
        h_start = max(0, center_h - neighborhood_size // 2)
        h_end = min(H, center_h + neighborhood_size // 2)
        w_start = max(0, center_w - neighborhood_size // 2)
        w_end = min(W, center_w + neighborhood_size // 2)
        
        # Extract local probability map
        local_prob = prob_map[h_start:h_end, w_start:w_end]
        local_prob_flat = local_prob.flatten()
        
        if local_prob_flat.sum() > 0:
            # Sample points in this neighborhood
            num_local_points = min(points_per_coarse, 
                                 len(local_prob_flat), 
                                 num_points - len(sampled_indices_list))
            
            if num_local_points > 0:
                local_indices = torch.multinomial(local_prob_flat, num_local_points, replacement=False)
                
                # Convert local indices to global indices
                local_h = local_indices // (w_end - w_start)
                local_w = local_indices % (w_end - w_start)
                global_h = local_h + h_start
                global_w = local_w + w_start
                global_indices = global_h * W + global_w
                
                sampled_indices_list.append(global_indices)
                
                # Convert to normalized coordinates
                h_coords = global_h.float() / H
                w_coords = global_w.float() / W
                coords = torch.stack([w_coords, h_coords], dim=1)
                sampled_coords_list.append(coords)
    
    # Concatenate all sampled points
    if sampled_indices_list:
        sampled_indices = torch.cat(sampled_indices_list)[:num_points]
        sampled_coords = torch.cat(sampled_coords_list)[:num_points]
    else:
        # Fallback: uniform random sampling
        print("Warning: Hierarchical sampling failed, using uniform sampling")
        sampled_indices = torch.randint(0, H * W, (num_points,), device=device)
        h_coords = (sampled_indices // W).float() / H
        w_coords = (sampled_indices % W).float() / W
        sampled_coords = torch.stack([w_coords, h_coords], dim=1)
    
    return sampled_indices, sampled_coords


def sample_points_grid_based(prob_map: torch.Tensor, num_points: int, H: int, W: int, 
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alternative grid-based sampling for very large images"""
    # Divide image into grid cells
    grid_size = int(np.sqrt(num_points)) * 2  # Oversample grid
    cell_h = H // grid_size
    cell_w = W // grid_size
    
    sampled_indices_list = []
    sampled_coords_list = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            if len(sampled_indices_list) >= num_points:
                break
                
            # Define cell boundaries
            h_start = i * cell_h
            h_end = min((i + 1) * cell_h, H)
            w_start = j * cell_w
            w_end = min((j + 1) * cell_w, W)
            
            # Extract cell probability
            cell_prob = prob_map[h_start:h_end, w_start:w_end]
            cell_prob_sum = cell_prob.sum()
            
            if cell_prob_sum > 0:
                # Sample one point from this cell
                cell_prob_flat = cell_prob.flatten()
                cell_idx = torch.multinomial(cell_prob_flat, 1, replacement=False)
                
                # Convert to global coordinates
                local_h = cell_idx // (w_end - w_start)
                local_w = cell_idx % (w_end - w_start)
                global_h = local_h + h_start
                global_w = local_w + w_start
                global_idx = global_h * W + global_w
                
                sampled_indices_list.append(global_idx)
                
                # Convert to normalized coordinates
                h_coord = global_h.float() / H
                w_coord = global_w.float() / W
                coord = torch.stack([w_coord, h_coord], dim=1)
                sampled_coords_list.append(coord)
    
    # Concatenate and pad if necessary
    if sampled_indices_list:
        sampled_indices = torch.cat(sampled_indices_list)
        sampled_coords = torch.cat(sampled_coords_list)
        
        # Pad with random samples if not enough points
        if len(sampled_indices) < num_points:
            remaining = num_points - len(sampled_indices)
            random_indices = torch.randint(0, H * W, (remaining,), device=device)
            random_h = (random_indices // W).float() / H
            random_w = (random_indices % W).float() / W
            random_coords = torch.stack([random_w, random_h], dim=1)
            
            sampled_indices = torch.cat([sampled_indices, random_indices])
            sampled_coords = torch.cat([sampled_coords, random_coords])
    else:
        # Fallback: uniform random sampling
        sampled_indices = torch.randint(0, H * W, (num_points,), device=device)
        h_coords = (sampled_indices // W).float() / H
        w_coords = (sampled_indices % W).float() / W
        sampled_coords = torch.stack([w_coords, h_coords], dim=1)
    
    return sampled_indices[:num_points], sampled_coords[:num_points]


def sample_colors_pytorch(target_image: torch.Tensor, sampled_indices: torch.Tensor, 
                         H: int, W: int, device: torch.device) -> torch.Tensor:
    """Sample colors from target image at specified locations using PyTorch"""
    # Convert flat indices to 2D coordinates
    h_indices = sampled_indices // W
    w_indices = sampled_indices % W
    
    # Sample colors using advanced indexing
    # target_image shape: [C, H, W]
    colors = target_image[:, h_indices, w_indices].T  # [num_points, C]
    
    return colors

class ImageGS_RS(nn.Module):
    def __init__(self, loss_type="L1+SSIM", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.total_num_points = kwargs["num_points"]  # Total budget N
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        )
        self.device = kwargs["device"]
        self.opt_type = kwargs.get("opt_type", "adam")
        
        # Content-adaptive parameters
        self.alpha_init = kwargs.get("alpha_init", 0.3)  # α parameter in paper
        # TODO Implement Top-K norm in gsplat
        self.K = kwargs.get("K", 10)  # Top-K parameter
        
        # Initialize with N/2 Gaussians
        self.init_num_points = self.total_num_points // 2
        self.current_num_points = self.init_num_points
        self.background = torch.ones(3, device=self.device)

        self.rotation_activation = torch.sigmoid
        
        # Learning rates from paper
        self.lr_xyz = kwargs.get("lr_xyz", 5e-4)
        self.lr_color = kwargs.get("lr_color", 5e-3)
        self.lr_scale = kwargs.get("lr_scale", 2e-3)
        self.lr_rot = kwargs.get("lr_rot", 2e-3)
        
        # Store target image for content-adaptive sampling
        # [1, C, H, W]
        self.content_adaptive_init(kwargs.get("init_image", None))

    def content_adaptive_init(self, target_image: torch.Tensor, chunk_size: int = 2048):
        """
        Initialize Gaussians based on content-adaptive sampling strategy
        Optimized for large 16K images using pure PyTorch operations
        
        Args:
            target_image: Input image tensor [1, C, H, W]
            chunk_size: Size of chunks for processing (default: 2048)
        """
        if target_image is None:
            print("Content Adaptive Init Failed........")
            return
        
        self.target_image = target_image.clone()
        
        # Handle input format [1, C, H, W]
        if target_image.dim() == 4:
            target_image = target_image.squeeze(0)  # Remove batch dimension -> [C, H, W]
        
        # Keep everything on GPU
        device = target_image.device
        C, H, W = target_image.shape

        print(f"Processing image of size: {H}x{W}")
        
        # Convert to grayscale on GPU
        if C == 3:
            # RGB to grayscale weights
            rgb_weights = torch.tensor([0.299, 0.587, 0.114], device=device).view(3, 1, 1)
            img_gray = (target_image * rgb_weights).sum(dim=0)
        elif C == 1:
            img_gray = target_image.squeeze(0)
        else:
            img_gray = target_image.mean(dim=0)
        
        # Process image in chunks to handle large 16K images
        gradient_magnitude = torch.zeros_like(img_gray, device=device)
        
        # Calculate optimal chunk overlap for gradient computation
        overlap = 16  # Overlap to handle edge effects
        
        num_chunks_h = (H + chunk_size - overlap - 1) // (chunk_size - overlap)
        num_chunks_w = (W + chunk_size - overlap - 1) // (chunk_size - overlap)
        
        print(f"Processing in {num_chunks_h}x{num_chunks_w} chunks of size {chunk_size}x{chunk_size}")
        
        for i, h_start in enumerate(range(0, H, chunk_size - overlap)):
            h_end = min(h_start + chunk_size, H)
            
            for j, w_start in enumerate(range(0, W, chunk_size - overlap)):
                w_end = min(w_start + chunk_size, W)
                
                # Extract chunk
                chunk = img_gray[h_start:h_end, w_start:w_end]
                
                # Compute gradients using PyTorch convolution
                grad_x, grad_y = compute_gradient_pytorch(chunk, device)
                chunk_grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
                
                # Handle overlapping regions by taking maximum
                actual_h_start = h_start + (overlap // 2 if h_start > 0 else 0)
                actual_h_end = h_end - (overlap // 2 if h_end < H else 0)
                actual_w_start = w_start + (overlap // 2 if w_start > 0 else 0)
                actual_w_end = w_end - (overlap // 2 if w_end < W else 0)
                
                chunk_h_start = actual_h_start - h_start
                chunk_h_end = chunk_h_start + (actual_h_end - actual_h_start)
                chunk_w_start = actual_w_start - w_start
                chunk_w_end = chunk_w_start + (actual_w_end - actual_w_start)
                
                gradient_magnitude[actual_h_start:actual_h_end, actual_w_start:actual_w_end] = \
                    torch.maximum(
                        gradient_magnitude[actual_h_start:actual_h_end, actual_w_start:actual_w_end],
                        chunk_grad_mag[chunk_h_start:chunk_h_end, chunk_w_start:chunk_w_end]
                    )
                
                # Clear intermediate tensors to save memory
                del chunk, grad_x, grad_y, chunk_grad_mag
        
        # Normalize gradient magnitude
        gradient_magnitude = gradient_magnitude / (gradient_magnitude.sum() + 1e-8)
        
        # Create sampling probabilities according to Equation 1
        uniform_prob = 1.0 / (H * W)
        prob_map = (1 - self.alpha_init) * gradient_magnitude + self.alpha_init * uniform_prob
        
        # Sample using efficient GPU-based approach with hierarchical sampling
        sampled_indices, sampled_coords = sample_points_pytorch(
            prob_map, self.init_num_points, H, W, device
        )
        
        # Initialize colors from target image at sampled locations
        colors = sample_colors_pytorch(target_image, sampled_indices, H, W, device)
        
        # Initialize parameters
        # Convert coordinates to [-1, 1] range
        self._xyz = nn.Parameter(sampled_coords * 2.0 - 1.0)
        self._rotation = nn.Parameter(torch.zeros(self.init_num_points, 1, dtype=torch.float32, device=device))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1), dtype=torch.float32, device=device))
        self._features_dc = nn.Parameter(colors)
        
        init_scale = 5.0  # 5 pixels
        self._scaling = nn.Parameter(torch.full((self.init_num_points, 2), 1.0/init_scale, 
                                            dtype=torch.float32, device=device))
        
        param_groups = [
            {'params': [self._xyz], 'lr': self.lr_xyz, "name": "xyz"},
            {'params': [self._features_dc], 'lr': self.lr_color, "name": "f_dc"},
            {'params': [self._scaling], 'lr': self.lr_scale, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.lr_rot, "name": "rotation"}
        ]
        
        self.optimizer = torch.optim.Adam(param_groups)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)
        
        # Free up intermediate tensors
        del gradient_magnitude, prob_map
        torch.cuda.empty_cache()
        
        print(f"Initialized {self.init_num_points} Gaussians successfully")

    @property
    def get_scaling(self):
        """Return actual scales from inverse scales"""
        return 1.0 / torch.abs(self._scaling + 1e-8)
    
    @property
    def get_rotation(self):
        """Convert rotation parameter to actual rotation angle"""
        # [0, 2*pi]
        return  self.rotation_activation(self._rotation) * 2 * math.pi
    
    @property
    def get_xyz(self):
        """Get normalized positions in [-1,1] range"""
        return torch.clamp(self._xyz, -0.9999, 0.9999)
    
    @property
    def get_features(self):
        return torch.clamp(self._features_dc, 0., 1.)
    
    @property
    def get_opacity(self):
        return self._opacity

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def add_gaussians_based_on_error(self, rendered_image, target_image):
        """
        Add new Gaussians to high-error regions according to Equation 2
        """
        if self.current_num_points >= self.total_num_points:
            return
        
        # Compute per-pixel error
        error = torch.abs(rendered_image - target_image).mean(dim=1)  # Average over RGB channels
        error_flat = error.flatten()
        
        # Determine how many Gaussians to add
        num_to_add = min(self.total_num_points // 8, self.total_num_points - self.current_num_points)
        
        if num_to_add <= 0:
            return
        
        # Use top-k sampling
        top_k = min(num_to_add * 100, len(error_flat))
        _, top_indices = torch.topk(error_flat, k=top_k, largest=True)
        
        # Randomly sample from top-k candidates
        selected_idx = torch.randperm(top_k, device=self.device)[:num_to_add]
        sampled_indices = top_indices[selected_idx]
        
        # Convert indices to pixel coordinates
        h_coords = (sampled_indices // self.W).float() / self.H
        w_coords = (sampled_indices % self.W).float() / self.W
        new_coords = torch.stack([w_coords, h_coords], dim=1)
        
        # Convert to tanh space
        new_xyz = 2.0 * new_coords - 1.0
        
        # Get colors from target image at sampled locations
        new_colors = []
        for idx in sampled_indices:
            h, w = idx // self.W, idx % self.W
            color = target_image[0, :, h, w]
            new_colors.append(color)
        new_colors = torch.stack(new_colors)
        
        # Initialize new Gaussian parameters
        init_scale = 5.0
        new_scaling = torch.full((num_to_add, 2), 1.0/init_scale, dtype=torch.float32, device=self.device)
        new_rotation = torch.zeros(num_to_add, 1, dtype=torch.float32, device=self.device)
    
        d = {"xyz": new_xyz,
            "f_dc": new_colors,
            "scaling" : new_scaling,
            "rotation" : new_rotation}
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # Update opacity buffer
        new_opacity = torch.ones((num_to_add, 1), dtype=torch.float32, device=self.device)
        self._opacity = torch.cat([self._opacity, new_opacity], dim=0)
        
        # Update current number of points
        self.current_num_points += num_to_add

    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(
            self.get_xyz, self.get_scaling, self.get_rotation, self.H, self.W, self.tile_bounds
        )
        out_img = rasterize_gaussians_sum(
            self.xys, depths, self.radii, conics, num_tiles_hit,
            self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W,
            background=self.background, return_alpha=False
        )
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def train_iter(self, gt_image, current_step):
        render_pkg = self.forward()
        image = render_pkg["render"]

        loss = imagegs_loss_fn(image, gt_image)
        loss.backward()
        
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        
        # Progressive Gaussian addition every 0.5K steps
        with torch.no_grad():
            if current_step % 500 == 0 and current_step > 0:
                self.add_gaussians_based_on_error(image, gt_image)
                print(f"Current Points: ", self.current_num_points)
            
        return loss, psnr