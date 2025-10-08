"""Differentiable OpenMM energy wrapper for Alanine Dipeptide (23 atoms).

Adapted from accelmd/training/openmm_bridge.py to handle 23-atom systems.
Provides autograd-aware potential energy computation for molecular transport.
"""

from functools import lru_cache
from typing import Tuple

import torch
from torch import Tensor

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app

__all__ = [
    "compute_potential_energy",
]

# Boltzmann constant in kJ/(mol*K)
K_BOLTZ = 8.314462618e-3


@lru_cache(maxsize=1)
def _get_aldp_context_23_atoms(pdb_path: str = "datasets/AA/ref.pdb") -> Tuple[mm.Context, int]:
    """Build OpenMM Context for 23-atom Alanine Dipeptide system.
    
    Args:
        pdb_path: Path to reference PDB file
        
    Returns:
        (context, n_atoms) tuple for energy/force computation
    """
    # Load from PDB instead of test system to get correct 23-atom structure
    pdb = app.PDBFile(pdb_path)
    forcefield = app.ForceField("amber99sbildn.xml", "implicit/obc1.xml")
    
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )
    
    integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("Reference")
    context = mm.Context(system, integrator, platform)
    context.setPositions(pdb.positions)
    
    n_atoms = len(pdb.positions)
    assert n_atoms == 23, f"Expected 23 atoms, got {n_atoms}"
    
    return context, n_atoms


class _OpenMMPotentialEnergyFunction(torch.autograd.Function):
    """Autograd bridge for potential energy computation.
    
    Forward:  U(x) in kJ/mol
    Backward: dU/dx = -F (forces in kJ/mol/nm)
    """
    
    @staticmethod
    def forward(ctx, coords_nm: Tensor) -> Tensor:
        # coords_nm shape: [B, 69] or [B, 23, 3]
        ctx.original_ndim = coords_nm.ndim
        if coords_nm.ndim == 3:  # [B, N, 3]
            B, N, _ = coords_nm.shape
            flat = coords_nm.view(B, -1)
        else:
            B, D = coords_nm.shape
            N = D // 3
            flat = coords_nm
            
        if flat.shape[1] != 69:
            raise ValueError(f"Expected 69-D coordinates (23 atoms × 3), got {flat.shape[1]}")
        
        context, n_atoms = _get_aldp_context_23_atoms()
        
        energies = torch.zeros(B, dtype=torch.double)
        forces_out = torch.zeros_like(flat, dtype=torch.double)
        
        for i in range(B):
            xyz = flat[i].detach().double().cpu().numpy().reshape(n_atoms, 3)
            context.setPositions(xyz * unit.nanometer)
            state = context.getState(getEnergy=True, getForces=True)
            
            # Energy in kJ/mol
            e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            
            # Forces in kJ/mol/nm
            forces_quantity = state.getForces(asNumpy=True)
            f_kj_per_nm = forces_quantity.value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            
            energies[i] = e_kj
            forces_out[i] = torch.tensor(f_kj_per_nm, dtype=torch.double).reshape(-1)
        
        # Save -F = dU/dx for backward pass
        ctx.save_for_backward(-forces_out.to(coords_nm.device))
        return energies.to(coords_nm.dtype).to(coords_nm.device)
    
    @staticmethod
    def backward(ctx, grad_out: Tensor):
        deriv, = ctx.saved_tensors  # [B, 69]
        # grad_out [B] - broadcast over coordinate dimension
        grad_coords = grad_out.unsqueeze(-1) * deriv  # [B, 69]
        
        if ctx.original_ndim == 3:
            B = grad_coords.shape[0]
            grad_coords = grad_coords.view(B, -1, 3)
        return grad_coords


def compute_potential_energy(coords: Tensor) -> Tensor:
    """Compute potential energy U(x) with autograd support.
    
    Args:
        coords: Molecular coordinates in nanometers
                Shape: [B, 69] or [B, 23, 3]
    
    Returns:
        energies: Potential energies in kJ/mol, shape [B]
    """
    return _OpenMMPotentialEnergyFunction.apply(coords)


def compute_reduced_energy(coords: Tensor, beta: float) -> Tensor:
    """Compute dimensionless reduced energy β·U(x).
    
    Args:
        coords: Molecular coordinates in nm, shape [B, 69] or [B, 23, 3]
        beta: Inverse temperature β = 1/(kB·T) in mol/kJ
        
    Returns:
        reduced_energies: β·U(x), shape [B]
    """
    U = compute_potential_energy(coords)
    return beta * U

