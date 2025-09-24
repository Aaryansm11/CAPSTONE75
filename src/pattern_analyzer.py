# src/pattern_analyzer.py
"""
Advanced pattern analysis and interpretation for discovered cardiac patterns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from sklearn.manifold import TSNE, UMAP
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

from src.config import CLASS_NAMES, REPORT_DIR
from src.data_utils import FeatureExtractor
from src.validation import SignalValidator

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    """Comprehensive analysis of discovered cardiac patterns."""

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or REPORT_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.signal_validator = SignalValidator()

    def analyze_discovered_patterns(self,
                                  embeddings: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  ecg_windows: np.ndarray,
                                  ppg_windows: np.ndarray,
                                  patient_indices: np.ndarray,
                                  quality_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive analysis of discovered patterns.

        Args:
            embeddings: Learned embeddings [n_samples, embedding_dim]
            cluster_labels: Cluster assignments [n_samples]
            ecg_windows: Original ECG windows [n_samples, window_length]
            ppg_windows: Original PPG windows [n_samples, window_length]
            patient_indices: Patient IDs for each window [n_samples]
            quality_scores: Signal quality scores [n_samples]

        Returns:
            Dictionary with comprehensive analysis results
        """
        analysis_results = {
            'cluster_statistics': self._compute_cluster_statistics(cluster_labels, patient_indices, quality_scores),
            'signal_characteristics': self._analyze_signal_characteristics(cluster_labels, ecg_windows, ppg_windows),
            'dimensionality_reduction': self._perform_dimensionality_reduction(embeddings, cluster_labels),
            'pattern_stability': self._assess_pattern_stability(embeddings, cluster_labels, patient_indices),
            'clinical_interpretation': self._interpret_patterns_clinically(cluster_labels, ecg_windows, ppg_windows),
            'visualization_data': self._prepare_visualization_data(embeddings, cluster_labels, patient_indices)
        }

        # Generate comprehensive report
        self._generate_pattern_report(analysis_results)

        return analysis_results

    def _compute_cluster_statistics(self,
                                  cluster_labels: np.ndarray,
                                  patient_indices: np.ndarray,
                                  quality_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Compute basic statistics for each discovered cluster."""
        unique_clusters = np.unique(cluster_labels)
        stats = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Skip noise points from DBSCAN
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_patients = patient_indices[cluster_mask]

            cluster_stats = {
                'size': int(np.sum(cluster_mask)),
                'unique_patients': len(np.unique(cluster_patients)),
                'avg_windows_per_patient': np.sum(cluster_mask) / len(np.unique(cluster_patients)),
                'patient_distribution': dict(zip(*np.unique(cluster_patients, return_counts=True)))
            }

            if quality_scores is not None:
                cluster_quality = quality_scores[cluster_mask]
                cluster_stats.update({
                    'avg_quality': float(np.mean(cluster_quality)),
                    'quality_std': float(np.std(cluster_quality)),
                    'min_quality': float(np.min(cluster_quality)),
                    'max_quality': float(np.max(cluster_quality))
                })

            stats[f'cluster_{cluster_id}'] = cluster_stats

        return stats

    def _analyze_signal_characteristics(self,
                                      cluster_labels: np.ndarray,
                                      ecg_windows: np.ndarray,
                                      ppg_windows: np.ndarray) -> Dict[str, Any]:
        """Analyze signal characteristics for each cluster."""
        unique_clusters = np.unique(cluster_labels)
        characteristics = {}

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_ecg = ecg_windows[cluster_mask]
            cluster_ppg = ppg_windows[cluster_mask]

            # Extract features for cluster
            ecg_features = []
            ppg_features = []

            for i in range(min(100, len(cluster_ecg))):  # Sample for efficiency
                ecg_feat = self.feature_extractor.extract_all_features(cluster_ecg[i], cluster_ppg[i])

                ecg_only = {k: v for k, v in ecg_feat.items() if k.startswith('ecg_')}
                ppg_only = {k: v for k, v in ecg_feat.items() if k.startswith('ppg_')}

                ecg_features.append(list(ecg_only.values()))
                ppg_features.append(list(ppg_only.values()))

            if ecg_features and ppg_features:
                ecg_features_array = np.array(ecg_features)
                ppg_features_array = np.array(ppg_features)

                characteristics[f'cluster_{cluster_id}'] = {
                    'ecg_features': {
                        'mean': ecg_features_array.mean(axis=0).tolist(),
                        'std': ecg_features_array.std(axis=0).tolist(),
                        'median': np.median(ecg_features_array, axis=0).tolist()
                    },
                    'ppg_features': {
                        'mean': ppg_features_array.mean(axis=0).tolist(),
                        'std': ppg_features_array.std(axis=0).tolist(),
                        'median': np.median(ppg_features_array, axis=0).tolist()
                    },
                    'signal_morphology': self._analyze_signal_morphology(cluster_ecg, cluster_ppg)
                }

        return characteristics

    def _analyze_signal_morphology(self, ecg_cluster: np.ndarray, ppg_cluster: np.ndarray) -> Dict[str, Any]:
        """Analyze morphological characteristics of signal clusters."""
        # Sample representative signals
        n_samples = min(50, len(ecg_cluster))
        sample_indices = np.random.choice(len(ecg_cluster), n_samples, replace=False)

        ecg_sample = ecg_cluster[sample_indices]
        ppg_sample = ppg_cluster[sample_indices]

        # Compute average waveforms
        avg_ecg = np.mean(ecg_sample, axis=0)
        avg_ppg = np.mean(ppg_sample, axis=0)
        std_ecg = np.std(ecg_sample, axis=0)
        std_ppg = np.std(ppg_sample, axis=0)

        # Basic morphological metrics
        ecg_peak_indices = self._find_peaks_robust(avg_ecg)
        ppg_peak_indices = self._find_peaks_robust(avg_ppg)

        morphology = {
            'avg_ecg_waveform': avg_ecg.tolist(),
            'avg_ppg_waveform': avg_ppg.tolist(),
            'ecg_variability': std_ecg.tolist(),
            'ppg_variability': std_ppg.tolist(),
            'ecg_peak_count': len(ecg_peak_indices),
            'ppg_peak_count': len(ppg_peak_indices),
            'ecg_amplitude_range': float(np.max(avg_ecg) - np.min(avg_ecg)),
            'ppg_amplitude_range': float(np.max(avg_ppg) - np.min(avg_ppg)),
            'signal_correlation': float(np.corrcoef(avg_ecg, avg_ppg)[0, 1]) if len(avg_ecg) == len(avg_ppg) else 0.0
        }

        return morphology

    def _find_peaks_robust(self, signal: np.ndarray) -> np.ndarray:
        """Robust peak detection for averaged signals."""
        from scipy.signal import find_peaks

        # Use adaptive threshold
        threshold = np.mean(signal) + 0.5 * np.std(signal)
        min_distance = len(signal) // 10  # Minimum distance between peaks

        peaks, _ = find_peaks(signal, height=threshold, distance=min_distance)
        return peaks

    def _perform_dimensionality_reduction(self,
                                        embeddings: np.ndarray,
                                        cluster_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Perform various dimensionality reduction techniques."""
        reduction_results = {}

        # PCA
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(embeddings)
            reduction_results['pca'] = {
                'coordinates': pca_result,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'total_variance_explained': np.sum(pca.explained_variance_ratio_)
            }
        except Exception as e:
            logger.warning(f"PCA failed: {e}")

        # t-SNE
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4))
            tsne_result = tsne.fit_transform(embeddings)
            reduction_results['tsne'] = {
                'coordinates': tsne_result
            }
        except Exception as e:
            logger.warning(f"t-SNE failed: {e}")

        # UMAP
        try:
            import umap
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            umap_result = umap_reducer.fit_transform(embeddings)
            reduction_results['umap'] = {
                'coordinates': umap_result
            }
        except ImportError:
            logger.warning("UMAP not available")
        except Exception as e:
            logger.warning(f"UMAP failed: {e}")

        return reduction_results

    def _assess_pattern_stability(self,
                                embeddings: np.ndarray,
                                cluster_labels: np.ndarray,
                                patient_indices: np.ndarray) -> Dict[str, float]:
        """Assess stability of discovered patterns."""
        stability_metrics = {}

        # Intra-cluster stability (how consistent are patterns within clusters)
        unique_clusters = np.unique(cluster_labels)
        intra_cluster_distances = []

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]

            if len(cluster_embeddings) > 1:
                # Compute pairwise distances within cluster
                from scipy.spatial.distance import pdist
                distances = pdist(cluster_embeddings, metric='cosine')
                intra_cluster_distances.extend(distances)

        stability_metrics['avg_intra_cluster_distance'] = float(np.mean(intra_cluster_distances)) if intra_cluster_distances else 0.0

        # Patient consistency (do different windows from same patient cluster together?)
        patient_consistency_scores = []
        unique_patients = np.unique(patient_indices)

        for patient_id in unique_patients:
            patient_mask = patient_indices == patient_id
            patient_clusters = cluster_labels[patient_mask]

            if len(patient_clusters) > 1:
                # Most common cluster for this patient
                most_common_cluster = np.bincount(patient_clusters[patient_clusters >= 0]).argmax()
                consistency = np.sum(patient_clusters == most_common_cluster) / len(patient_clusters)
                patient_consistency_scores.append(consistency)

        stability_metrics['avg_patient_consistency'] = float(np.mean(patient_consistency_scores)) if patient_consistency_scores else 0.0

        return stability_metrics

    def _interpret_patterns_clinically(self,
                                     cluster_labels: np.ndarray,
                                     ecg_windows: np.ndarray,
                                     ppg_windows: np.ndarray) -> Dict[str, Any]:
        """Provide clinical interpretation of discovered patterns."""
        interpretations = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_ecg = ecg_windows[cluster_mask]
            cluster_ppg = ppg_windows[cluster_mask]

            # Analyze representative signals
            n_representative = min(10, len(cluster_ecg))
            representative_indices = np.random.choice(len(cluster_ecg), n_representative, replace=False)

            # Extract clinical features
            heart_rates = []
            rhythm_regularity_scores = []
            signal_quality_scores = []

            for idx in representative_indices:
                ecg_signal = cluster_ecg[idx]
                ppg_signal = cluster_ppg[idx]

                # Estimate heart rate
                from src.data_utils import SignalProcessor
                processor = SignalProcessor()
                r_peaks = processor.detect_r_peaks(ecg_signal)
                hr = processor.calculate_heart_rate(r_peaks)
                heart_rates.append(hr)

                # Assess rhythm regularity
                if len(r_peaks) > 2:
                    rr_intervals = np.diff(r_peaks)
                    rhythm_regularity = 1.0 / (np.std(rr_intervals) + 1e-6)
                    rhythm_regularity_scores.append(rhythm_regularity)

                # Signal quality
                ecg_quality = self.signal_validator.validate_signal_quality(ecg_signal, 360)
                signal_quality_scores.append(ecg_quality['quality_score'])

            # Clinical interpretation
            interpretation = {
                'avg_heart_rate': float(np.mean(heart_rates)) if heart_rates else 0.0,
                'heart_rate_variability': float(np.std(heart_rates)) if heart_rates else 0.0,
                'rhythm_regularity': float(np.mean(rhythm_regularity_scores)) if rhythm_regularity_scores else 0.0,
                'signal_quality': float(np.mean(signal_quality_scores)) if signal_quality_scores else 0.0,
                'clinical_characteristics': self._categorize_clinical_pattern(
                    np.mean(heart_rates) if heart_rates else 0.0,
                    np.mean(rhythm_regularity_scores) if rhythm_regularity_scores else 0.0
                )
            }

            interpretations[f'cluster_{cluster_id}'] = interpretation

        return interpretations

    def _categorize_clinical_pattern(self, avg_hr: float, rhythm_regularity: float) -> Dict[str, str]:
        """Categorize patterns based on clinical characteristics."""
        characteristics = {}

        # Heart rate categorization
        if avg_hr < 60:
            characteristics['heart_rate_category'] = 'Bradycardia'
        elif avg_hr > 100:
            characteristics['heart_rate_category'] = 'Tachycardia'
        else:
            characteristics['heart_rate_category'] = 'Normal'

        # Rhythm categorization
        if rhythm_regularity > 100:
            characteristics['rhythm_category'] = 'Very Regular'
        elif rhythm_regularity > 50:
            characteristics['rhythm_category'] = 'Regular'
        elif rhythm_regularity > 20:
            characteristics['rhythm_category'] = 'Irregular'
        else:
            characteristics['rhythm_category'] = 'Very Irregular'

        # Combined interpretation
        if characteristics['heart_rate_category'] == 'Normal' and characteristics['rhythm_category'] in ['Regular', 'Very Regular']:
            characteristics['overall_pattern'] = 'Normal Sinus Rhythm'
        elif characteristics['rhythm_category'] in ['Irregular', 'Very Irregular']:
            characteristics['overall_pattern'] = 'Possible Arrhythmia'
        elif characteristics['heart_rate_category'] in ['Bradycardia', 'Tachycardia']:
            characteristics['overall_pattern'] = f'{characteristics["heart_rate_category"]} Pattern'
        else:
            characteristics['overall_pattern'] = 'Unclassified Pattern'

        return characteristics

    def _prepare_visualization_data(self,
                                  embeddings: np.ndarray,
                                  cluster_labels: np.ndarray,
                                  patient_indices: np.ndarray) -> Dict[str, Any]:
        """Prepare data for visualization."""
        viz_data = {
            'embeddings': embeddings,
            'cluster_labels': cluster_labels,
            'patient_indices': patient_indices,
            'cluster_colors': self._generate_cluster_colors(cluster_labels),
            'cluster_info': {}
        }

        unique_clusters = np.unique(cluster_labels)
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                continue

            cluster_mask = cluster_labels == cluster_id
            viz_data['cluster_info'][f'cluster_{cluster_id}'] = {
                'size': int(np.sum(cluster_mask)),
                'centroid': embeddings[cluster_mask].mean(axis=0).tolist()
            }

        return viz_data

    def _generate_cluster_colors(self, cluster_labels: np.ndarray) -> Dict[int, str]:
        """Generate distinct colors for each cluster."""
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

        color_map = {}
        for i, cluster_id in enumerate(unique_clusters):
            if cluster_id == -1:
                color_map[cluster_id] = 'gray'  # Noise points
            else:
                color_map[cluster_id] = colors[i]

        return color_map

    def _generate_pattern_report(self, analysis_results: Dict[str, Any]):
        """Generate comprehensive pattern analysis report."""
        report_path = self.save_dir / "pattern_discovery_report.md"

        with open(report_path, 'w') as f:
            f.write("# Cardiac Pattern Discovery Report\n\n")

            # Cluster overview
            f.write("## Discovered Patterns Overview\n\n")
            cluster_stats = analysis_results['cluster_statistics']

            f.write(f"**Total Patterns Discovered:** {len(cluster_stats)}\n\n")

            for cluster_name, stats in cluster_stats.items():
                f.write(f"### {cluster_name.replace('_', ' ').title()}\n")
                f.write(f"- **Size:** {stats['size']} windows\n")
                f.write(f"- **Unique Patients:** {stats['unique_patients']}\n")
                f.write(f"- **Avg Windows per Patient:** {stats['avg_windows_per_patient']:.1f}\n")
                if 'avg_quality' in stats:
                    f.write(f"- **Average Signal Quality:** {stats['avg_quality']:.3f}\n")
                f.write("\n")

            # Clinical interpretations
            f.write("## Clinical Interpretations\n\n")
            clinical_data = analysis_results['clinical_interpretation']

            for cluster_name, interpretation in clinical_data.items():
                f.write(f"### {cluster_name.replace('_', ' ').title()}\n")
                characteristics = interpretation['clinical_characteristics']
                f.write(f"- **Heart Rate Category:** {characteristics['heart_rate_category']}\n")
                f.write(f"- **Rhythm Category:** {characteristics['rhythm_category']}\n")
                f.write(f"- **Overall Pattern:** {characteristics['overall_pattern']}\n")
                f.write(f"- **Average Heart Rate:** {interpretation['avg_heart_rate']:.1f} BPM\n")
                f.write(f"- **Signal Quality Score:** {interpretation['signal_quality']:.3f}\n")
                f.write("\n")

            # Pattern stability
            f.write("## Pattern Stability Analysis\n\n")
            stability = analysis_results['pattern_stability']
            f.write(f"- **Average Intra-cluster Distance:** {stability['avg_intra_cluster_distance']:.3f}\n")
            f.write(f"- **Patient Consistency Score:** {stability['avg_patient_consistency']:.3f}\n")
            f.write("\n")

            # Dimensionality reduction
            f.write("## Dimensionality Reduction Results\n\n")
            if 'pca' in analysis_results['dimensionality_reduction']:
                pca_data = analysis_results['dimensionality_reduction']['pca']
                f.write(f"- **PCA Explained Variance (PC1):** {pca_data['explained_variance_ratio'][0]:.3f}\n")
                f.write(f"- **PCA Explained Variance (PC2):** {pca_data['explained_variance_ratio'][1]:.3f}\n")
                f.write(f"- **Total Variance Explained:** {pca_data['total_variance_explained']:.3f}\n")

            f.write("\n---\n")
            f.write("*Report generated automatically by Pattern Discovery System*\n")

        logger.info(f"Pattern analysis report saved to {report_path}")

    def create_pattern_visualizations(self, analysis_results: Dict[str, Any]) -> List[Path]:
        """Create comprehensive visualizations of discovered patterns."""
        viz_paths = []

        # 1. Dimensionality reduction plots
        if 'dimensionality_reduction' in analysis_results:
            dim_red_path = self._plot_dimensionality_reduction(analysis_results)
            viz_paths.extend(dim_red_path)

        # 2. Cluster characteristics heatmap
        char_path = self._plot_cluster_characteristics(analysis_results)
        if char_path:
            viz_paths.append(char_path)

        # 3. Pattern distribution plots
        dist_path = self._plot_pattern_distributions(analysis_results)
        if dist_path:
            viz_paths.append(dist_path)

        return viz_paths

    def _plot_dimensionality_reduction(self, analysis_results: Dict[str, Any]) -> List[Path]:
        """Plot dimensionality reduction results."""
        paths = []
        dim_red_data = analysis_results['dimensionality_reduction']
        viz_data = analysis_results['visualization_data']

        cluster_labels = viz_data['cluster_labels']

        fig, axes = plt.subplots(1, len(dim_red_data), figsize=(5 * len(dim_red_data), 5))
        if len(dim_red_data) == 1:
            axes = [axes]

        for i, (method, data) in enumerate(dim_red_data.items()):
            coords = data['coordinates']

            # Plot points colored by cluster
            unique_clusters = np.unique(cluster_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

            for j, cluster_id in enumerate(unique_clusters):
                cluster_mask = cluster_labels == cluster_id
                color = 'gray' if cluster_id == -1 else colors[j]
                label = 'Noise' if cluster_id == -1 else f'Pattern {cluster_id}'

                axes[i].scatter(coords[cluster_mask, 0], coords[cluster_mask, 1],
                              c=[color], label=label, alpha=0.6, s=20)

            axes[i].set_title(f'{method.upper()} Visualization')
            axes[i].set_xlabel('Component 1')
            axes[i].set_ylabel('Component 2')
            axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        path = self.save_dir / "dimensionality_reduction_plots.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        paths.append(path)

        return paths

    def _plot_cluster_characteristics(self, analysis_results: Dict[str, Any]) -> Optional[Path]:
        """Plot cluster characteristics heatmap."""
        try:
            clinical_data = analysis_results['clinical_interpretation']

            # Prepare data for heatmap
            cluster_names = []
            heart_rates = []
            rhythm_scores = []
            quality_scores = []

            for cluster_name, data in clinical_data.items():
                cluster_names.append(cluster_name.replace('cluster_', 'Pattern '))
                heart_rates.append(data['avg_heart_rate'])
                rhythm_scores.append(data['rhythm_regularity'])
                quality_scores.append(data['signal_quality'])

            # Create heatmap data
            heatmap_data = np.array([heart_rates, rhythm_scores, quality_scores]).T

            fig, ax = plt.subplots(figsize=(8, 6))

            # Normalize data for better visualization
            heatmap_data_norm = (heatmap_data - heatmap_data.min(axis=0)) / (heatmap_data.max(axis=0) - heatmap_data.min(axis=0) + 1e-6)

            im = ax.imshow(heatmap_data_norm, cmap='viridis', aspect='auto')

            # Set ticks and labels
            ax.set_xticks(range(3))
            ax.set_xticklabels(['Heart Rate', 'Rhythm Regularity', 'Signal Quality'])
            ax.set_yticks(range(len(cluster_names)))
            ax.set_yticklabels(cluster_names)

            # Add colorbar
            plt.colorbar(im, ax=ax, label='Normalized Value')

            # Add text annotations
            for i in range(len(cluster_names)):
                for j in range(3):
                    text = f'{heatmap_data[i, j]:.1f}'
                    ax.text(j, i, text, ha='center', va='center', color='white', fontweight='bold')

            plt.title('Cluster Characteristics Heatmap')
            plt.tight_layout()

            path = self.save_dir / "cluster_characteristics_heatmap.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            return path

        except Exception as e:
            logger.warning(f"Failed to create cluster characteristics heatmap: {e}")
            return None

    def _plot_pattern_distributions(self, analysis_results: Dict[str, Any]) -> Optional[Path]:
        """Plot pattern size and patient distributions."""
        try:
            cluster_stats = analysis_results['cluster_statistics']

            cluster_names = []
            sizes = []
            unique_patients = []

            for cluster_name, stats in cluster_stats.items():
                cluster_names.append(cluster_name.replace('cluster_', 'Pattern '))
                sizes.append(stats['size'])
                unique_patients.append(stats['unique_patients'])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Pattern sizes
            bars1 = ax1.bar(cluster_names, sizes, color='skyblue', alpha=0.7)
            ax1.set_title('Pattern Sizes (Number of Windows)')
            ax1.set_ylabel('Number of Windows')
            ax1.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height)}', ha='center', va='bottom')

            # Unique patients per pattern
            bars2 = ax2.bar(cluster_names, unique_patients, color='lightcoral', alpha=0.7)
            ax2.set_title('Patient Distribution Across Patterns')
            ax2.set_ylabel('Number of Unique Patients')
            ax2.tick_params(axis='x', rotation=45)

            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height)}', ha='center', va='bottom')

            plt.tight_layout()

            path = self.save_dir / "pattern_distributions.png"
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()

            return path

        except Exception as e:
            logger.warning(f"Failed to create pattern distribution plots: {e}")
            return None