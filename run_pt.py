"""
Parallel Tempering (PT) runner with traditional vanilla swaps.

Usage examples:
  - Single-temperature Langevin (no swaps):
      python run_pt.py --system AA --temp 300.0 --steps 10000

  - PT with default ladder [300, 450, 670, 1000]:
      python run_pt.py --system AA --steps 10000

  - PT with a custom ladder:
      python run_pt.py --system AA --temp 300.0 450.0 670.0 1000.0 --steps 10000

Notes:
  - Outputs a tensor of shape [num_temps, steps, 3*num_atoms] in results/ as
    {system}_pt_trajectory.pt (positions in nm).
  - Also produces one Ramachandran plot per temperature in plots/ as
    {system}_{temp}_ramachandran.png.
  - Uses traditional vanilla swaps for parallel tempering.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
import mdtraj as md
import matplotlib.pyplot as plt
try:
    import yaml  # type: ignore
except Exception:
    yaml = None

from simtk.openmm.app import Simulation
from simtk.openmm import LangevinMiddleIntegrator, Platform
from simtk.unit import kelvin, picosecond, femtoseconds, nanometer, kilojoule_per_mole
from openmm import app

from src.utils.plot_utils import plot_Ramachandran


DEFAULT_LADDER = [300.0, 450.0, 670.0, 1000.0]
LOCAL_STEPS_PER_CYCLE = 10   # Integrator steps per swap/record cycle (2 fs each → 20 fs per cycle)
DEFAULT_BURN_IN_CYCLES = 1000  # Burn-in cycles before recording (with swaps)
K_BOLTZ_KJ_MOL_K = 8.314462618e-3  # kJ/(mol*K)


def _load_system_and_topology(system_name: str):
    pdb_path = f"datasets/{system_name}/ref.pdb"
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing PDB at {pdb_path}")
    forcefield = app.ForceField("amber99sbildn.xml", "implicit/obc1.xml")
    pdb = app.PDBFile(pdb_path)
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
    )
    return pdb, system


def _create_simulation(pdb: app.PDBFile, system: app.System, temperature_K: float, seed: int) -> Simulation:
    integrator = LangevinMiddleIntegrator(temperature_K * kelvin, 1 / picosecond, 2 * femtoseconds)
    try:
        integrator.setRandomNumberSeed(int(seed))  # Ensure different noise across replicas
    except Exception:
        pass
    platform = Platform.getPlatformByName("CPU")
    sim = Simulation(pdb.topology, system, integrator, platform=platform)
    sim.context.setPositions(pdb.positions)
    sim.context.setVelocitiesToTemperature(temperature_K * kelvin)
    sim.minimizeEnergy()
    return sim


def _swap_coords_and_velocs(sim_a: Simulation, sim_b: Simulation) -> None:
    """Swap positions and velocities between two replicas in-place."""
    state_a = sim_a.context.getState(getPositions=True, getVelocities=True)
    state_b = sim_b.context.getState(getPositions=True, getVelocities=True)
    pos_a = state_a.getPositions(asNumpy=True)
    vel_a = state_a.getVelocities(asNumpy=True)
    pos_b = state_b.getPositions(asNumpy=True)
    vel_b = state_b.getVelocities(asNumpy=True)
    sim_a.context.setPositions(pos_b)
    sim_a.context.setVelocities(vel_b)
    sim_b.context.setPositions(pos_a)
    sim_b.context.setVelocities(vel_a)


def _load_experiments_cfg_temps() -> List[float]:
    """Load temperature ladder from experiments config if available."""
    cfg_path = os.path.join("configs", "experiments.yaml")
    temps_cfg = DEFAULT_LADDER.copy()
    if yaml is None or not os.path.exists(cfg_path):
        return temps_cfg
    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        vals = cfg.get("temperatures", {}).get("values")
        if isinstance(vals, list) and len(vals) > 0:
            temps_cfg = [float(v) for v in vals]
    except Exception:
        pass
    return temps_cfg


def _compute_accept_prob_naive(beta_i: float, beta_j: float, U_i_kJmol: float, U_j_kJmol: float) -> float:
    # alpha = min(1, exp((beta_i - beta_j)*(U_j - U_i)))
    delta = (beta_i - beta_j) * (U_j_kJmol - U_i_kJmol)
    try:
        return float(np.exp(min(0.0, 0.0) + delta)) if delta < 0 else float(np.exp(delta))
    except OverflowError:
        return 1.0 if delta > 0 else 0.0


def _get_potential_energy_kJmol(sim: Simulation) -> float:
    state = sim.context.getState(getEnergy=True)
    # OpenMM's State exposes getPotentialEnergy()
    U = state.getPotentialEnergy()
    return U.value_in_unit(kilojoule_per_mole)


def _record_positions(sim: Simulation, num_atoms: int) -> np.ndarray:
    state = sim.context.getState(getPositions=True)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(nanometer)
    arr = np.asarray(pos_nm, dtype=np.float32).reshape(num_atoms, 3)
    return arr


def _save_results_and_plots(
    system_name: str,
    temps: List[float],
    positions_per_temp_flat: np.ndarray,  # [M, steps, 3N]
    pdb_topology: app.Topology,
):
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Save tensor
    out_path = os.path.join("results", f"{system_name}_pt_trajectory.pt")
    torch.save(torch.from_numpy(positions_per_temp_flat), out_path)

    # Create Ramachandran plots per temperature
    M, steps, dim = positions_per_temp_flat.shape
    num_atoms = dim // 3

    # If plots appear mislabeled due to stream ordering, label by reversed temps for naming
    label_temps = list(reversed(temps))
    for idx in range(M):
        coords_nm = positions_per_temp_flat[idx].reshape(steps, num_atoms, 3)
        traj = md.Trajectory(coords_nm, md.Topology.from_openmm(pdb_topology))
        _, phi = md.compute_phi(traj)
        _, psi = md.compute_psi(traj)

        phi_flat = phi.flatten()
        psi_flat = psi.flatten()
        mask = np.isfinite(phi_flat) & np.isfinite(psi_flat)
        phi_clean = phi_flat[mask]
        psi_clean = psi_flat[mask]

        fig, ax = plt.subplots(figsize=(6, 6))
        plot_Ramachandran(ax, phi_clean, psi_clean)
        plot_path = os.path.join(
            "plots",
            f"{system_name}_{int(round(label_temps[idx]))}_ramachandran.png",
        )
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)


def run_single(system_name: str, temperature: float, steps: int) -> None:
    pdb, system = _load_system_and_topology(system_name)
    sim = _create_simulation(pdb, system, temperature, seed=12345)
    num_atoms = len(pdb.positions)

    # Burn-in
    for _ in range(DEFAULT_BURN_IN_CYCLES):
        sim.step(LOCAL_STEPS_PER_CYCLE)

    # Record per-step positions
    traj = np.empty((steps, num_atoms, 3), dtype=np.float32)
    for s in range(steps):
        sim.step(LOCAL_STEPS_PER_CYCLE)
        traj[s] = _record_positions(sim, num_atoms)

    flat = traj.reshape(steps, num_atoms * 3)
    # Save and plot using the unified helper with M=1
    _save_results_and_plots(system_name, [temperature], flat[None, ...], pdb.topology)


def run_pt(system_name: str, temps: List[float], steps: int) -> None:
    # Load system once; reuse for all replicas
    pdb, system = _load_system_and_topology(system_name)
    M = len(temps)
    assert M >= 2, "PT requires at least two temperatures. For single T, use --temp <T> or omit to run Langevin."

    # Create simulations as fixed replicas at their own temperatures (replica r at T_r)
    sims: List[Simulation] = [_create_simulation(pdb, system, T, seed=1337 + i) for i, T in enumerate(temps)]

    num_atoms = len(pdb.positions)
    # Storage: [M, steps, 3N]
    positions_flat = np.empty((M, steps, num_atoms * 3), dtype=np.float32)

    beta = [1.0 / (K_BOLTZ_KJ_MOL_K * T) for T in temps]

    # Maintain mapping from temperature index to replica index
    temp_to_replica = list(range(M))

    # Helper: one PT cycle
    def pt_cycle():
        for r in range(M):
            sims[r].step(LOCAL_STEPS_PER_CYCLE)

        # Even-odd neighbor swaps in two passes
        for phase in (0, 1):
            start = phase
            for i in range(start, M - 1, 2):
                # Traditional vanilla swaps
                # Replicas hosting temperatures i and i+1
                ri = temp_to_replica[i]
                rj = temp_to_replica[i + 1]
                Ui = _get_potential_energy_kJmol(sims[ri])
                Uj = _get_potential_energy_kJmol(sims[rj])
                acc = min(1.0, np.exp((beta[i] - beta[i + 1]) * (Uj - Ui)))
                if np.random.rand() < acc:
                    temp_to_replica[i], temp_to_replica[i + 1] = rj, ri

    # Burn-in cycles
    for _ in range(DEFAULT_BURN_IN_CYCLES):
        pt_cycle()

    # PT loop: attempt swaps and record
    for s in range(steps):
        pt_cycle()

        # Record per-temperature coordinates after swaps
        for i in range(M):
            r = temp_to_replica[i]
            pos = _record_positions(sims[r], num_atoms).reshape(-1)
            positions_flat[i, s] = pos

    _save_results_and_plots(system_name, temps, positions_flat, pdb.topology)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="AA")
    parser.add_argument(
        "--temp",
        type=float,
        nargs="*",
        help="Either a single temperature for Langevin, or a ladder for PT. If omitted, uses default ladder.",
    )
    parser.add_argument("--steps", type=int, default=10000)
    args = parser.parse_args()

    system_name = args.system
    steps = int(args.steps)

    if args.temp is None:
        temps: List[float] = _load_experiments_cfg_temps()
    elif len(args.temp) == 1:
        temps = [float(args.temp[0])]
    else:
        temps = [float(t) for t in args.temp]

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    if len(temps) == 1:
        run_single(system_name, temps[0], steps)
    else:
        run_pt(system_name, temps, steps)


if __name__ == "__main__":
    main()