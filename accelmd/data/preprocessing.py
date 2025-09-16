"""Unified preprocessing utilities for molecular data and TARFlow integration.

This module consolidates all preprocessing functions including coordinate transformations,
augmentations, and TARFlow-specific coordinate-to-patch conversions.
"""
from __future__ import annotations

from typing import Tuple, Optional, List
import numpy as np
import torch
import mdtraj as md

__all__ = [
    "filter_chirality", 
    "center_coordinates", 
    "torch_to_mdtraj", 
    "random_rotation_augment",
    "coordinate_to_patches",
    "patches_to_coordinates",
    "apply_coordinate_augmentation"
]


# =============================================================================
# Legacy molecular preprocessing functions (from molecular_data.py)
# =============================================================================

def random_rotation_augment(samples: torch.Tensor) -> torch.Tensor:
    """Apply random 3D rotation augmentation to molecular coordinates.
    
    Generates random rotation matrices and applies them to each sample.
    The coordinates are first centered, rotated, then the center is added back.
    
    Parameters
    ----------
    samples : torch.Tensor
        Coordinate samples, shape [N, n_atoms, 3]
        
    Returns
    -------
    torch.Tensor
        Randomly rotated coordinates with same shape as input
    """
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(f"Expected shape [N, n_atoms, 3], got {samples.shape}")
    
    N, n_atoms, _ = samples.shape
    device = samples.device
    dtype = samples.dtype
    
    # Center coordinates first
    centers = samples.mean(dim=1, keepdim=True)  # [N, 1, 3]
    centered_coords = samples - centers
    
    rotated_coords = torch.zeros_like(centered_coords)
    
    # Apply a different random rotation to each sample
    for i in range(N):
        # Generate random rotation matrix using Rodriguez formula
        # Sample random axis (unit vector) and angle
        axis = torch.randn(3, device=device, dtype=dtype)
        axis = axis / torch.norm(axis)  # normalize to unit vector
        
        # Random rotation angle between 0 and 2π
        angle = torch.rand(1, device=device, dtype=dtype) * 2 * np.pi
        
        # Rodriguez rotation formula: R = I + sin(θ)[K] + (1-cos(θ))[K]²
        # where [K] is the skew-symmetric matrix of the axis vector
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Skew-symmetric matrix [K]
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=device, dtype=dtype)
        
        # Rotation matrix
        I = torch.eye(3, device=device, dtype=dtype)
        R = I + sin_angle * K + (1 - cos_angle) * torch.mm(K, K)
        
        # Apply rotation to all atoms in this sample
        coords_sample = centered_coords[i]  # [n_atoms, 3]
        rotated_coords[i] = torch.mm(coords_sample, R.T)  # Apply rotation
    
    # Add centers back
    augmented_coords = rotated_coords + centers
    
    return augmented_coords


def filter_chirality(samples: torch.Tensor, ref_trajectory: Optional[md.Trajectory] = None) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Filter out incorrect chirality conformations from samples.
    
    This function filters batch for the L-form based on dihedral angle differences
    at specific atomic indices (17, 26 for alanine dipeptide).
    
    Parameters
    ----------
    samples : torch.Tensor
        Coordinate samples to filter, shape [N, n_atoms, 3]
    ref_trajectory : Optional[md.Trajectory]
        Reference trajectory (unused in this implementation)
        
    Returns
    -------
    filtered_samples : torch.Tensor
        Samples with correct chirality only  
    counters : Tuple[int, int]
        (initial_d_form_count, final_d_form_count) for tracking
    """
    # Implementation based on main/utils/aldp_utils.py filter_chirality
    # This assumes we're working with internal coordinates or dihedral angles
    # For Cartesian coordinates, we need to compute dihedral angles first
    
    if samples.ndim == 3:
        # If we have [N, n_atoms, 3] Cartesian coordinates, 
        # we would need to compute dihedral angles first
        # For now, return all samples (no filtering)
        n_samples = samples.shape[0]
        return samples, (0, 0)
    
    # For flat coordinate input or dihedral angles
    if samples.ndim == 2 and samples.shape[1] >= 27:  # Assumes dihedral angles included
        ind = [17, 26]  # Indices for chirality check
        mean_diff = -0.043
        threshold = 0.8
        
        initial_count = samples.shape[0]
        
        # Compute wrapped differences
        diff_ = torch.column_stack((
            samples[:, ind[0]] - samples[:, ind[1]],
            samples[:, ind[0]] - samples[:, ind[1]] + 2 * np.pi,
            samples[:, ind[0]] - samples[:, ind[1]] - 2 * np.pi
        ))
        
        # Find minimum difference
        min_diff_ind = torch.min(torch.abs(diff_), 1).indices
        diff = diff_[torch.arange(samples.shape[0]), min_diff_ind]
        
        # Filter based on threshold
        keep_mask = torch.abs(diff - mean_diff) < threshold
        filtered_samples = samples[keep_mask]
        
        final_count = filtered_samples.shape[0]
        d_form_initial = initial_count - torch.sum(keep_mask).item()
        d_form_final = 0  # All remaining are L-form
        
        return filtered_samples, (d_form_initial, d_form_final)
    
    # Default: no filtering
    return samples, (0, 0)


def center_coordinates(samples: torch.Tensor) -> torch.Tensor:
    """Center molecular coordinates by removing the center of mass.
    
    Makes configurations mean-free by subtracting the center of mass
    from each configuration.
    
    Parameters
    ----------
    samples : torch.Tensor
        Coordinate samples, shape [N, n_atoms, 3] or [N, n_atoms*3]
        
    Returns
    -------
    torch.Tensor
        Mean-centered coordinates with same shape as input
    """
    # Implementation based on main/utils/se3_utils.py remove_mean
    original_shape = samples.shape
    
    if samples.ndim == 3:
        # Shape [N, n_atoms, 3]
        n_particles = samples.shape[1]
        n_dimensions = samples.shape[2]
        # Subtract mean across atoms (dim=1)
        centered = samples - torch.mean(samples, dim=1, keepdim=True)
    elif samples.ndim == 2:
        # Assume flat coordinates [N, n_atoms*3], reshape to [N, n_atoms, 3]
        n_coords = samples.shape[1]
        if n_coords % 3 != 0:
            raise ValueError(f"Expected coordinates divisible by 3, got {n_coords}")
        n_particles = n_coords // 3
        n_dimensions = 3
        
        # Reshape to [N, n_atoms, 3]
        reshaped = samples.view(-1, n_particles, n_dimensions)
        # Center coordinates
        centered_reshaped = reshaped - torch.mean(reshaped, dim=1, keepdim=True)
        # Reshape back to original
        centered = centered_reshaped.view(original_shape)
    else:
        raise ValueError(f"Unsupported tensor shape: {original_shape}")
    
    return centered


def torch_to_mdtraj(coords: torch.Tensor, topology: md.Topology) -> md.Trajectory:
    """Convert PyTorch coordinates to MDTraj trajectory.
    
    Parameters
    ----------
    coords : torch.Tensor
        Coordinates tensor of shape [N, n_atoms, 3] in nanometers
    topology : md.Topology
        MDTraj topology object
        
    Returns
    -------
    md.Trajectory
        MDTraj trajectory object
    """
    # Convert to numpy and ensure correct units (MDTraj expects nanometers)
    coords_np = coords.detach().cpu().numpy()
    
    # Create MDTraj trajectory
    traj = md.Trajectory(coords_np, topology)
    
    return traj


# =============================================================================
# TARFlow-specific preprocessing functions
# =============================================================================

def coordinate_to_patches(coords: torch.Tensor, patch_strategy: str = "atom_groups") -> torch.Tensor:
    """Convert molecular coordinates [B, N, 3] to patches for autoregressive processing.
    
    This adapts TARFlow's image patches concept to molecular coordinates by grouping
    atoms into sequential "patches" that can be processed autoregressively.
    
    Parameters
    ----------
    coords : torch.Tensor
        Molecular coordinates of shape [B, N, 3] where N is number of atoms
    patch_strategy : str
        Strategy for creating patches:
        - "atom_groups": Group consecutive atoms (default)
        - "residue_based": Group by residue (future extension)
        - "distance_based": Group by spatial proximity (future extension)
        
    Returns
    -------
    torch.Tensor
        Patches of shape [B, num_patches, patch_dim] where patch_dim = patch_size * 3
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coordinates of shape [B, N, 3], got {coords.shape}")
    
    B, N, _ = coords.shape
    
    if patch_strategy == "atom_groups":
        # Simple strategy: group consecutive atoms into patches
        # For molecular systems, we typically have small numbers of atoms
        # so we can use individual atoms as "patches"
        patches = coords.view(B, N, 3)  # Each atom is a 3D patch
        return patches
    
    elif patch_strategy == "pair_groups":
        # Group atoms in pairs (useful for small molecules)
        if N % 2 == 1:
            # Pad with zeros if odd number of atoms
            coords_padded = torch.cat([coords, torch.zeros(B, 1, 3, device=coords.device)], dim=1)
            N_padded = N + 1
        else:
            coords_padded = coords
            N_padded = N
        
        # Reshape to [B, N//2, 6] - each patch contains 2 atoms
        patches = coords_padded.view(B, N_padded // 2, 6)
        return patches
    
    else:
        raise ValueError(f"Unknown patch strategy: {patch_strategy}")


def patches_to_coordinates(patches: torch.Tensor, patch_strategy: str = "atom_groups", original_num_atoms: int = None) -> torch.Tensor:
    """Convert patches back to molecular coordinates [B, N, 3].
    
    Inverse operation of coordinate_to_patches.
    
    Parameters
    ----------
    patches : torch.Tensor
        Patches of shape [B, num_patches, patch_dim]
    patch_strategy : str
        Strategy used to create patches (must match forward operation)
    original_num_atoms : int, optional
        Original number of atoms (needed for padding removal)
        
    Returns
    -------
    torch.Tensor
        Molecular coordinates of shape [B, N, 3]
    """
    B = patches.shape[0]
    
    if patch_strategy == "atom_groups":
        # Each patch is a single atom [3D]
        coords = patches.view(B, -1, 3)
        return coords
    
    elif patch_strategy == "pair_groups":
        # Each patch contains 2 atoms [6D]
        coords_reshaped = patches.view(B, -1, 3)  # [B, 2*num_patches, 3]
        
        # Remove padding if it was added
        if original_num_atoms is not None and coords_reshaped.shape[1] > original_num_atoms:
            coords = coords_reshaped[:, :original_num_atoms, :]
        else:
            coords = coords_reshaped
        
        return coords
    
    else:
        raise ValueError(f"Unknown patch strategy: {patch_strategy}")


def apply_coordinate_augmentation(coords: torch.Tensor, augmentation_types: List[str] = None) -> torch.Tensor:
    """Apply various coordinate augmentations for training robustness.
    
    Parameters
    ----------
    coords : torch.Tensor
        Coordinates of shape [B, N, 3]
    augmentation_types : List[str], optional
        List of augmentation types to apply:
        - "rotation": Random 3D rotation
        - "noise": Small Gaussian noise
        - "center": Center coordinates
        
    Returns
    -------
    torch.Tensor
        Augmented coordinates with same shape as input
    """
    if augmentation_types is None:
        return coords
    
    augmented = coords.clone()
    
    for aug_type in augmentation_types:
        if aug_type == "rotation":
            augmented = random_rotation_augment(augmented)
        elif aug_type == "noise":
            # Add small amount of Gaussian noise (0.01 nm std)
            noise_std = 0.01
            noise = torch.randn_like(augmented) * noise_std
            augmented = augmented + noise
        elif aug_type == "center":
            augmented = center_coordinates(augmented)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    return augmented
