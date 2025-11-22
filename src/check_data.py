#!/usr/bin/env python3
"""
Data validation and visualization for 2D materials
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

try:
    from .config import CONFIG, PROCESSED_DATA_DIR, PLOTS_DIR
except ImportError:
    from config import CONFIG, PROCESSED_DATA_DIR, PLOTS_DIR

logger = logging.getLogger(__name__)

class DataValidator:
    """Validate and visualize materials data"""
    
    def __init__(self, data_path=None):
        if data_path is None:
            data_path = PROCESSED_DATA_DIR / 'materials_processed.csv'
        
        self.data_path = Path(data_path)
        self.df = None
        
        if self.data_path.exists():
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded data from {self.data_path}")
        else:
            logger.warning(f"Data file not found: {self.data_path}")
    
    def check_structure(self):
        """Check data structure and quality"""
        if self.df is None:
            print("❌ No data found!")
            print(f"   Expected location: {self.data_path}")
            return False
        
        print("\n" + "="*60)
        print("DATA STRUCTURE CHECK")
        print("="*60)
        
        print(f"\n📊 Dataset Overview:")
        print(f"   Total samples: {len(self.df)}")
        print(f"   Total features: {len(self.df.columns)}")
        print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\n📋 Columns available:")
        for col in self.df.columns:
            non_null = self.df[col].notna().sum()
            null_pct = (self.df[col].isna().sum() / len(self.df)) * 100
            dtype = self.df[col].dtype
            print(f"   • {col:<25} {str(dtype):<10} ({non_null}/{len(self.df)} non-null, {null_pct:.1f}% missing)")
        
        # Check target variables
        print(f"\n🎯 Target Variables:")
        
        if 'band_gap' in self.df.columns:
            bg_data = self.df['band_gap'].dropna()
            if len(bg_data) > 0:
                print(f"   Band Gap:")
                print(f"      - Count:  {len(bg_data)}")
                print(f"      - Mean:   {bg_data.mean():.3f} eV")
                print(f"      - Std:    {bg_data.std():.3f} eV")
                print(f"      - Min:    {bg_data.min():.3f} eV")
                print(f"      - Max:    {bg_data.max():.3f} eV")
                print(f"      - Median: {bg_data.median():.3f} eV")
        
        if 'formation_energy' in self.df.columns:
            fe_data = self.df['formation_energy'].dropna()
            if len(fe_data) > 0:
                print(f"   Formation Energy:")
                print(f"      - Count:  {len(fe_data)}")
                print(f"      - Mean:   {fe_data.mean():.3f} eV/atom")
                print(f"      - Std:    {fe_data.std():.3f} eV/atom")
                print(f"      - Min:    {fe_data.min():.3f} eV/atom")
                print(f"      - Max:    {fe_data.max():.3f} eV/atom")
                print(f"      - Median: {fe_data.median():.3f} eV/atom")
        
        # Data quality summary
        print(f"\n✅ Data Quality:")
        
        if 'is_complete' in self.df.columns:
            complete = self.df['is_complete'].sum()
            print(f"   Complete samples: {complete}/{len(self.df)} ({100*complete/len(self.df):.1f}%)")
        
        duplicates = self.df.duplicated().sum()
        print(f"   Duplicate rows: {duplicates}")
        
        # Check for NaN/Inf values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(self.df[numeric_cols]).sum().sum()
        print(f"   Infinite values: {inf_count}")
        
        # Data sources
        if 'source' in self.df.columns:
            print(f"\n📚 Data Sources:")
            source_counts = self.df['source'].value_counts()
            for source, count in source_counts.items():
                print(f"   • {source}: {count} materials ({100*count/len(self.df):.1f}%)")
        
        return True
    
    def visualize(self, save_plots=True):
        """Create comprehensive visualizations"""
        if self.df is None:
            print("❌ No data to visualize")
            return
        
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
        sns.set_palette("husl")
        
        # Create main overview plot
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Band gap distribution
        ax1 = plt.subplot(3, 3, 1)
        if 'band_gap' in self.df.columns:
            data = self.df['band_gap'].dropna()
            ax1.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
            ax1.set_xlabel('Band Gap (eV)')
            ax1.set_ylabel('Count')
            ax1.set_title('Band Gap Distribution')
            ax1.legend()
        
        # 2. Formation energy distribution
        ax2 = plt.subplot(3, 3, 2)
        if 'formation_energy' in self.df.columns:
            data = self.df['formation_energy'].dropna()
            ax2.hist(data, bins=30, edgecolor='black', alpha=0.7, color='green')
            ax2.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
            ax2.set_xlabel('Formation Energy (eV/atom)')
            ax2.set_ylabel('Count')
            ax2.set_title('Formation Energy Distribution')
            ax2.legend()
        
        # 3. Band gap vs Formation energy
        ax3 = plt.subplot(3, 3, 3)
        if all(col in self.df.columns for col in ['band_gap', 'formation_energy']):
            valid = self.df[['band_gap', 'formation_energy']].dropna()
            scatter = ax3.scatter(valid['formation_energy'], valid['band_gap'], 
                                alpha=0.5, s=20, c=valid['band_gap'], cmap='viridis')
            ax3.set_xlabel('Formation Energy (eV/atom)')
            ax3.set_ylabel('Band Gap (eV)')
            ax3.set_title('Band Gap vs Formation Energy')
            plt.colorbar(scatter, ax=ax3, label='Band Gap (eV)')
        
        # 4. Stability distribution
        ax4 = plt.subplot(3, 3, 4)
        if 'energy_above_hull' in self.df.columns:
            data = self.df['energy_above_hull'].dropna()
            ax4.hist(data, bins=30, edgecolor='black', alpha=0.7, color='orange')
            ax4.set_xlabel('Energy Above Hull (eV/atom)')
            ax4.set_ylabel('Count')
            ax4.set_title('Stability Distribution')
            ax4.axvline(0.1, color='red', linestyle='--', label='Stability threshold')
            ax4.legend()
        
        # 5. Thickness distribution
        ax5 = plt.subplot(3, 3, 5)
        if 'thickness' in self.df.columns:
            data = self.df['thickness'].dropna()
            ax5.hist(data, bins=30, edgecolor='black', alpha=0.7, color='purple')
            ax5.set_xlabel('Thickness (Å)')
            ax5.set_ylabel('Count')
            ax5.set_title('Material Thickness Distribution')
        
        # 6. Direct vs Indirect band gap
        ax6 = plt.subplot(3, 3, 6)
        if 'is_gap_direct' in self.df.columns and 'band_gap' in self.df.columns:
            direct = self.df[self.df['is_gap_direct'] == True]['band_gap'].dropna()
            indirect = self.df[self.df['is_gap_direct'] == False]['band_gap'].dropna()
            ax6.hist([direct, indirect], bins=20, label=['Direct', 'Indirect'], 
                    alpha=0.7, edgecolor='black')
            ax6.set_xlabel('Band Gap (eV)')
            ax6.set_ylabel('Count')
            ax6.set_title('Direct vs Indirect Band Gap')
            ax6.legend()
        
        # 7. Data sources
        ax7 = plt.subplot(3, 3, 7)
        if 'source' in self.df.columns:
            source_counts = self.df['source'].value_counts()
            ax7.bar(range(len(source_counts)), source_counts.values, 
                   tick_label=source_counts.index)
            ax7.set_ylabel('Count')
            ax7.set_title('Materials by Source')
            plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 8. Missing data heatmap
        ax8 = plt.subplot(3, 3, 8)
        missing_data = (self.df.isnull().sum() / len(self.df)) * 100
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)[:10]
        if not missing_data.empty:
            ax8.barh(range(len(missing_data)), missing_data.values)
            ax8.set_yticks(range(len(missing_data)))
            ax8.set_yticklabels(missing_data.index)
            ax8.set_xlabel('Missing %')
            ax8.set_title('Top Missing Data Fields')
        
        # 9. Correlation matrix
        ax9 = plt.subplot(3, 3, 9)
        numeric_cols = ['band_gap', 'formation_energy', 'energy_above_hull', 
                       'thickness', 'natoms']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        if len(available_cols) > 1:
            corr_data = self.df[available_cols].corr()
            im = ax9.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax9.set_xticks(range(len(available_cols)))
            ax9.set_yticks(range(len(available_cols)))
            ax9.set_xticklabels(available_cols, rotation=45, ha='right')
            ax9.set_yticklabels(available_cols)
            ax9.set_title('Feature Correlation Matrix')
            plt.colorbar(im, ax=ax9)
            
            # Add correlation values
            for i in range(len(available_cols)):
                for j in range(len(available_cols)):
                    text = ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        plt.suptitle('2D Materials Dataset Overview', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_plots:
            output_file = PLOTS_DIR / 'data_overview.png'
            plt.savefig(output_file, dpi=CONFIG.get('figure_dpi', 100), bbox_inches='tight')
            print(f"✅ Plot saved: {output_file}")
        
        plt.show()
        
        # Create additional detailed plots
        self._create_detailed_plots(save_plots)
    
    def _create_detailed_plots(self, save_plots=True):
        """Create additional detailed visualizations"""
        
        # Box plots for outlier detection
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'band_gap' in self.df.columns:
            self.df.boxplot(column='band_gap', ax=axes[0])
            axes[0].set_ylabel('Band Gap (eV)')
            axes[0].set_title('Band Gap Outliers')
        
        if 'formation_energy' in self.df.columns:
            self.df.boxplot(column='formation_energy', ax=axes[1])
            axes[1].set_ylabel('Formation Energy (eV/atom)')
            axes[1].set_title('Formation Energy Outliers')
        
        plt.suptitle('Outlier Detection', fontsize=14)
        plt.tight_layout()
        
        if save_plots:
            output_file = PLOTS_DIR / 'outlier_detection.png'
            plt.savefig(output_file, dpi=CONFIG.get('figure_dpi', 100))
            print(f"✅ Plot saved: {output_file}")
        
        plt.show()
    
    def prepare_ml_data(self):
        """Prepare ML-ready dataset"""
        if self.df is None:
            print("❌ No data to prepare")
            return None
        
        print("\n" + "="*60)
        print("PREPARING ML-READY DATA")
        print("="*60)
        
        # Select complete samples
        if 'is_complete' in self.df.columns:
            ml_df = self.df[self.df['is_complete']].copy()
        else:
            ml_df = self.df.dropna(subset=['band_gap', 'formation_energy']).copy()
        
        print(f"✓ Selected {len(ml_df)} complete samples from {len(self.df)} total")
        
        # Identify feature columns
        feature_cols = ['thickness', 'energy_above_hull', 'exfoliation_energy', 'natoms']
        available_features = [col for col in feature_cols if col in ml_df.columns]
        
        # Prepare data dictionary
        ml_data = {
            'band_gap': ml_df['band_gap'].values,
            'formation_energy': ml_df['formation_energy'].values,
            'formulas': ml_df['formula'].values if 'formula' in ml_df.columns else None,
            'material_ids': ml_df['material_id'].values if 'material_id' in ml_df.columns else None,
            'feature_names': available_features,
            'n_samples': len(ml_df),
            'n_features': len(available_features)
        }
        
        # Add features if available
        if available_features:
            ml_data['features'] = ml_df[available_features].values
            print(f"✓ Included {len(available_features)} features: {', '.join(available_features)}")
        else:
            print("⚠️  No additional features available")
            ml_data['features'] = None
        
        # Save ML-ready data
        output_file = PROCESSED_DATA_DIR / 'ml_ready_data.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(ml_data, f)
        
        print(f"\n💾 ML-ready data saved: {output_file}")
        print(f"   Samples: {ml_data['n_samples']}")
        print(f"   Features: {ml_data['n_features']}")
        
        # Also save as numpy arrays
        np_dir = PROCESSED_DATA_DIR / 'numpy'
        np_dir.mkdir(exist_ok=True)
        
        np.save(np_dir / 'y_band_gap.npy', ml_data['band_gap'])
        np.save(np_dir / 'y_formation_energy.npy', ml_data['formation_energy'])
        
        if ml_data['features'] is not None:
            np.save(np_dir / 'X_features.npy', ml_data['features'])
        
        print(f"✓ Numpy arrays saved in: {np_dir}/")
        
        return ml_data
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        if self.df is None:
            return {}
        
        stats = {
            'total_materials': len(self.df),
            'complete_samples': self.df['is_complete'].sum() if 'is_complete' in self.df.columns else 0,
            'sources': self.df['source'].value_counts().to_dict() if 'source' in self.df.columns else {},
        }
        
        # Add property statistics
        for prop in ['band_gap', 'formation_energy', 'energy_above_hull', 'thickness']:
            if prop in self.df.columns:
                data = self.df[prop].dropna()
                if len(data) > 0:
                    stats[f'{prop}_stats'] = {
                        'count': len(data),
                        'mean': data.mean(),
                        'std': data.std(),
                        'min': data.min(),
                        'max': data.max(),
                        'median': data.median()
                    }
        
        return stats

def main():
    """Main function for standalone execution"""
    validator = DataValidator()
    
    if validator.check_structure():
        validator.visualize()
        ml_data = validator.prepare_ml_data()
        
        print("\n✅ Data validation complete!")
        return ml_data
    else:
        print("\n❌ Data validation failed")
        return None

if __name__ == "__main__":
    main()
