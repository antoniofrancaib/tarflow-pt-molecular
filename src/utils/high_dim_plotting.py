#!/usr/bin/env python3
"""
Visualization tools for high-dimensional distributions and flow analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Tuple, Optional, List
import os


def analyze_high_dim_distribution(distribution, 
                                num_samples: int = 2000,
                                device: str = 'cpu',
                                save_dir: str = 'plots_high_dim',
                                prefix: str = 'distribution'):
    """
    Comprehensive analysis of a high-dimensional distribution
    
    Args:
        distribution: Distribution object with sample() and log_prob() methods
        num_samples: Number of samples for analysis
        device: Device for sampling
        save_dir: Directory to save plots
        prefix: Prefix for saved files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🔍 Analyzing high-dimensional distribution: {type(distribution).__name__}")
    
    # Generate samples
    samples = distribution.sample(num_samples, device).cpu().numpy()
    dim = samples.shape[1]
    
    print(f"   Samples shape: {samples.shape}")
    print(f"   Sample statistics: mean={samples.mean():.3f}, std={samples.std():.3f}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # 1. PCA Analysis
    print("   Computing PCA...")
    pca = PCA(n_components=min(10, dim))
    samples_pca = pca.fit_transform(samples)
    
    # Plot PCA projections
    ax1 = plt.subplot(3, 4, 1)
    plt.scatter(samples_pca[:, 0], samples_pca[:, 1], alpha=0.6, s=2, c='blue')
    plt.title(f'PCA: PC1 vs PC2\n(Var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[1]:.2f})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 4, 2)
    if samples_pca.shape[1] > 2:
        plt.scatter(samples_pca[:, 0], samples_pca[:, 2], alpha=0.6, s=2, c='red')
        plt.title(f'PCA: PC1 vs PC3\n(Var: {pca.explained_variance_ratio_[0]:.2f}, {pca.explained_variance_ratio_[2]:.2f})')
        plt.xlabel('PC1')
        plt.ylabel('PC3')
    plt.grid(True, alpha=0.3)
    
    # 2. Explained variance
    ax3 = plt.subplot(3, 4, 3)
    n_components_plot = min(len(pca.explained_variance_ratio_), 20)
    plt.bar(range(n_components_plot), pca.explained_variance_ratio_[:n_components_plot])
    plt.title('PCA Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.xticks(range(0, n_components_plot, max(1, n_components_plot//5)))
    
    # 3. Cumulative explained variance
    ax4 = plt.subplot(3, 4, 4)
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(len(cumsum_var)), cumsum_var, 'b-o', markersize=3)
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.grid(True, alpha=0.3)
    
    # 4. Pairwise marginal analysis (first few dimensions)
    max_pairwise = min(6, dim)
    if max_pairwise >= 2:
        ax5 = plt.subplot(3, 4, 5)
        plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=2, c='green')
        plt.title('Raw Dimensions: X₁ vs X₂')
        plt.xlabel('X₁')
        plt.ylabel('X₂')
        plt.grid(True, alpha=0.3)
    
    if max_pairwise >= 4:
        ax6 = plt.subplot(3, 4, 6)
        plt.scatter(samples[:, 2], samples[:, 3], alpha=0.6, s=2, c='orange')
        plt.title('Raw Dimensions: X₃ vs X₄')
        plt.xlabel('X₃')
        plt.ylabel('X₄')
        plt.grid(True, alpha=0.3)
    
    # 5. Distance analysis
    ax7 = plt.subplot(3, 4, 7)
    # Compute pairwise distances (subsample for efficiency)
    subsample_idx = np.random.choice(len(samples), min(500, len(samples)), replace=False)
    distances = np.linalg.norm(samples[subsample_idx, None] - samples[None, subsample_idx], axis=2)
    distances = distances[np.triu_indices_from(distances, k=1)]
    
    plt.hist(distances, bins=50, alpha=0.7, density=True)
    plt.title(f'Pairwise Distance Distribution\n(Mean: {distances.mean():.2f})')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    
    # 6. Dimension-wise statistics
    ax8 = plt.subplot(3, 4, 8)
    dim_means = samples.mean(axis=0)
    dim_stds = samples.std(axis=0)
    
    dims_to_plot = min(20, dim)
    plt.errorbar(range(dims_to_plot), dim_means[:dims_to_plot], 
                yerr=dim_stds[:dims_to_plot], fmt='o-', alpha=0.7)
    plt.title('Per-Dimension Statistics')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    
    # 7. t-SNE (if computationally feasible)
    if dim <= 50 and num_samples <= 1000:
        print("   Computing t-SNE...")
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, num_samples//4))
            samples_tsne = tsne.fit_transform(samples)
            
            ax9 = plt.subplot(3, 4, 9)
            plt.scatter(samples_tsne[:, 0], samples_tsne[:, 1], alpha=0.6, s=2, c='purple')
            plt.title('t-SNE Projection')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.grid(True, alpha=0.3)
        except Exception as e:
            print(f"   t-SNE failed: {e}")
    
    # 8. Correlation matrix (for reasonable dimensions)
    if dim <= 20:
        ax10 = plt.subplot(3, 4, 10)
        corr_matrix = np.corrcoef(samples.T)
        im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.colorbar(im, shrink=0.8)
    
    # 9. Log probability distribution
    if hasattr(distribution, 'log_prob'):
        try:
            print("   Computing log probabilities...")
            samples_torch = torch.tensor(samples, dtype=torch.float32).to(device)
            log_probs = distribution.log_prob(samples_torch).cpu().numpy()
            
            ax11 = plt.subplot(3, 4, 11)
            plt.hist(log_probs, bins=50, alpha=0.7, density=True)
            plt.title(f'Log Probability Distribution\n(Mean: {log_probs.mean():.2f})')
            plt.xlabel('Log Probability')
            plt.ylabel('Density')
            plt.grid(True, alpha=0.3)
        except Exception as e:
            print(f"   Log probability computation failed: {e}")
    
    # 10. Effective dimensionality metrics
    ax12 = plt.subplot(3, 4, 12)
    
    # Participation ratio
    eigenvals = pca.explained_variance_
    participation_ratio = (eigenvals.sum() ** 2) / (eigenvals ** 2).sum()
    
    # 95% variance dimensions
    var_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    var_99 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.99) + 1
    
    metrics = [dim, participation_ratio, var_95, var_99]
    labels = ['Total Dim', 'Participation\nRatio', '95% Var\nDims', '99% Var\nDims']
    
    bars = plt.bar(labels, metrics, color=['gray', 'blue', 'green', 'red'], alpha=0.7)
    plt.title('Dimensionality Metrics')
    plt.ylabel('Value')
    
    # Add value labels on bars
    for bar, metric in zip(bars, metrics):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{metric:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the comprehensive analysis
    save_path = os.path.join(save_dir, f'{prefix}_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved analysis to: {save_path}")
    plt.show()
    
    # Return analysis summary
    analysis_summary = {
        'dimension': dim,
        'num_samples': num_samples,
        'participation_ratio': participation_ratio,
        'var_95_dims': var_95,
        'var_99_dims': var_99,
        'mean_pairwise_distance': distances.mean() if 'distances' in locals() else None,
        'explained_variance_ratios': pca.explained_variance_ratio_[:10].tolist(),
        'sample_mean': samples.mean(axis=0)[:10].tolist(),
        'sample_std': samples.std(axis=0)[:10].tolist()
    }
    
    return analysis_summary


def compare_distributions(dist1, dist2, 
                         names: List[str] = ['Distribution 1', 'Distribution 2'],
                         num_samples: int = 1000,
                         device: str = 'cpu',
                         save_dir: str = 'plots_high_dim'):
    """
    Compare two high-dimensional distributions
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🔄 Comparing distributions: {names[0]} vs {names[1]}")
    
    # Sample from both distributions
    samples1 = dist1.sample(num_samples, device).cpu().numpy()
    samples2 = dist2.sample(num_samples, device).cpu().numpy()
    
    dim = samples1.shape[1]
    
    # PCA on combined data
    combined_samples = np.vstack([samples1, samples2])
    pca = PCA(n_components=min(10, dim))
    combined_pca = pca.fit_transform(combined_samples)
    
    samples1_pca = combined_pca[:num_samples]
    samples2_pca = combined_pca[num_samples:]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # PCA comparison
    axes[0, 0].scatter(samples1_pca[:, 0], samples1_pca[:, 1], alpha=0.6, s=2, c='blue', label=names[0])
    axes[0, 0].scatter(samples2_pca[:, 0], samples2_pca[:, 1], alpha=0.6, s=2, c='red', label=names[1])
    axes[0, 0].set_title('PCA: PC1 vs PC2')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Raw dimensions comparison
    if dim >= 2:
        axes[0, 1].scatter(samples1[:, 0], samples1[:, 1], alpha=0.6, s=2, c='blue', label=names[0])
        axes[0, 1].scatter(samples2[:, 0], samples2[:, 1], alpha=0.6, s=2, c='red', label=names[1])
        axes[0, 1].set_title('Raw: X₁ vs X₂')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Distance distributions
    dist1_distances = np.linalg.norm(samples1[::10, None] - samples1[None, ::10], axis=2)
    dist1_distances = dist1_distances[np.triu_indices_from(dist1_distances, k=1)]
    
    dist2_distances = np.linalg.norm(samples2[::10, None] - samples2[None, ::10], axis=2)
    dist2_distances = dist2_distances[np.triu_indices_from(dist2_distances, k=1)]
    
    axes[0, 2].hist(dist1_distances, bins=30, alpha=0.7, density=True, label=names[0])
    axes[0, 2].hist(dist2_distances, bins=30, alpha=0.7, density=True, label=names[1])
    axes[0, 2].set_title('Pairwise Distance Distributions')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Per-dimension statistics
    axes[1, 0].plot(samples1.mean(axis=0)[:20], 'b-o', label=f'{names[0]} mean', markersize=3)
    axes[1, 0].plot(samples2.mean(axis=0)[:20], 'r-o', label=f'{names[1]} mean', markersize=3)
    axes[1, 0].set_title('Per-Dimension Means')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(samples1.std(axis=0)[:20], 'b-o', label=f'{names[0]} std', markersize=3)
    axes[1, 1].plot(samples2.std(axis=0)[:20], 'r-o', label=f'{names[1]} std', markersize=3)
    axes[1, 1].set_title('Per-Dimension Standard Deviations')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Explained variance comparison
    pca1 = PCA(n_components=min(10, dim))
    pca1.fit(samples1)
    pca2 = PCA(n_components=min(10, dim))
    pca2.fit(samples2)
    
    axes[1, 2].plot(pca1.explained_variance_ratio_[:10], 'b-o', label=names[0], markersize=3)
    axes[1, 2].plot(pca2.explained_variance_ratio_[:10], 'r-o', label=names[1], markersize=3)
    axes[1, 2].set_title('PCA Explained Variance')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{names[0]}_vs_{names[1]}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved comparison to: {save_path}")
    plt.show()


if __name__ == "__main__":
    # Test the visualization tools
    from ..distributions.high_dim_gaussian_mixture import HighDimGaussianMixture, HighDimSwissRoll
    
    print("Testing high-dimensional visualization tools...")
    
    # Create test distributions
    mixture = HighDimGaussianMixture(dim=20, num_components=3, separation=2.5)
    swiss_roll = HighDimSwissRoll(dim=20, intrinsic_dim=2, noise_level=0.15)
    
    # Analyze individual distributions
    mixture_analysis = analyze_high_dim_distribution(
        mixture, num_samples=1500, prefix='gaussian_mixture_20d'
    )
    
    swiss_analysis = analyze_high_dim_distribution(
        swiss_roll, num_samples=1500, prefix='swiss_roll_20d'
    )
    
    # Compare distributions
    compare_distributions(
        mixture, swiss_roll, 
        names=['Gaussian_Mixture', 'Swiss_Roll'],
        num_samples=1000
    )
    
    print("High-dimensional visualization test completed!")
