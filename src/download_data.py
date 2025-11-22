#!/usr/bin/env python3
"""
Data downloader for 2D materials databases
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from .config import CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR
except ImportError:
    from config import CONFIG, RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MaterialsDataDownloader:
    """Main class for downloading 2D materials data"""
    
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.data = pd.DataFrame()
        self.download_summary = {}
    
    def download_jarvis(self):
        """Download JARVIS-2D database"""
        logger.info("Starting JARVIS-2D download...")
        print("\n📥 Downloading JARVIS-2D database...")
        
        try:
            from jarvis.db.figshare import data
            
            # Get 2D materials data
            print("⏳ This may take a few minutes...")
            dft_2d = data('dft_2d')
            
            materials = []
            for item in tqdm(dft_2d, desc="Processing JARVIS materials"):
                material = {
                    'material_id': item.get('jid', ''),
                    'formula': item.get('formula', ''),
                    
                    # Band gap properties
                    'band_gap': item.get('optb88vdw_bandgap', np.nan),
                    'band_gap_mbj': item.get('mbj_bandgap', np.nan),
                    'is_gap_direct': item.get('gap_type', '') == 'direct',
                    
                    # Energy properties
                    'formation_energy': item.get('formation_energy_peratom', np.nan),
                    'energy_above_hull': item.get('ehull', np.nan),
                    'exfoliation_energy': item.get('exfoliation_energy', np.nan),
                    
                    # Structural properties
                    'thickness': item.get('thickness', np.nan),
                    'natoms': item.get('natoms', np.nan),
                    'crystal_system': item.get('crys', ''),
                    
                    # Metadata
                    'source': 'JARVIS-2D',
                    'download_date': datetime.now().isoformat()
                }
                materials.append(material)
            
            df = pd.DataFrame(materials)
            
            # Save raw JARVIS data
            jarvis_file = self.raw_data_dir / 'jarvis_2d_raw.csv'
            df.to_csv(jarvis_file, index=False)
            logger.info(f"Saved {len(df)} JARVIS materials to {jarvis_file}")
            
            print(f"✅ Downloaded {len(df)} materials from JARVIS-2D")
            self.download_summary['JARVIS-2D'] = len(df)
            return df
            
        except Exception as e:
            logger.error(f"JARVIS download failed: {str(e)}")
            print(f"❌ JARVIS download failed: {e}")
            return pd.DataFrame()
    
    def download_c2db_info(self):
        """Create instructions for C2DB manual download"""
        print("\n📥 C2DB Database Information...")
        
        c2db_dir = self.raw_data_dir / 'c2db'
        c2db_dir.mkdir(exist_ok=True)
        
        instructions = """C2DB Database Download Instructions
=====================================

The C2DB database contains ~4000 2D materials with comprehensive properties.

To download:
1. Visit: https://c2db.fysik.dtu.dk/
2. Click on "Download" section
3. Download the full database or selected properties
4. Extract files to: {}

Required properties for ML:
- Band gap (gap, gap_dir)
- Formation energy (hform)
- Energy above hull (ehull)
- Thickness
- Magnetic properties (optional)

After downloading, run:
  python -m src.process_c2db

For more information, see:
  https://c2db.fysik.dtu.dk/about.html
""".format(c2db_dir)
        
        # Save instructions
        with open(c2db_dir / 'README.txt', 'w') as f:
            f.write(instructions)
        
        print(f"ℹ️  C2DB requires manual download")
        print(f"   Instructions saved to: {c2db_dir}/README.txt")
        print(f"   Visit: https://c2db.fysik.dtu.dk/")
        
        self.download_summary['C2DB'] = 'Manual download required'
        return pd.DataFrame()
    
    def download_sample_data(self):
        """Generate sample dataset for testing"""
        print("\n📥 Generating sample dataset...")
        
        # Common 2D materials with approximate properties
        materials_data = [
            # TMDCs
            ('MoS2', 1.8, -0.24, 6.5), ('WS2', 2.0, -0.22, 6.5),
            ('MoSe2', 1.5, -0.20, 7.0), ('WSe2', 1.6, -0.19, 7.0),
            ('MoTe2', 1.0, -0.15, 7.5), ('WTe2', 0.7, -0.12, 7.5),
            
            # Other 2D materials
            ('h-BN', 5.9, -0.45, 3.3), ('Graphene', 0.0, 0.0, 3.4),
            ('Phosphorene', 2.0, -0.35, 5.3), ('Silicene', 0.5, -0.10, 2.3),
            ('GaSe', 2.0, -0.30, 8.0), ('InSe', 1.3, -0.28, 8.5),
            ('SnS2', 2.2, -0.25, 5.9), ('SnSe2', 1.0, -0.22, 6.1),
            ('GeS', 1.6, -0.32, 4.5), ('GeSe', 1.1, -0.30, 4.8),
            ('PtS2', 1.6, -0.18, 5.0), ('PtSe2', 1.2, -0.16, 5.3),
            ('HfS2', 2.0, -0.35, 5.6), ('HfSe2', 1.1, -0.32, 6.0),
            ('ZrS2', 1.8, -0.38, 5.5), ('ZrSe2', 0.9, -0.35, 5.9),
            
            # MXenes
            ('Ti2C', 0.1, -0.40, 2.5), ('Ti3C2', 0.2, -0.42, 3.0),
            ('V2C', 0.15, -0.38, 2.6), ('Nb2C', 0.3, -0.36, 2.7),
        ]
        
        materials = []
        for i, (formula, gap, fe, thick) in enumerate(materials_data):
            # Add some realistic variation
            materials.append({
                'material_id': f'sample_{i+1:04d}',
                'formula': formula,
                'band_gap': gap + np.random.normal(0, 0.05),
                'formation_energy': fe + np.random.normal(0, 0.02),
                'energy_above_hull': abs(np.random.normal(0, 0.03)),
                'thickness': thick + np.random.normal(0, 0.1),
                'is_gap_direct': np.random.choice([True, False]),
                'natoms': np.random.randint(2, 10),
                'source': 'Sample',
                'download_date': datetime.now().isoformat()
            })
        
        df = pd.DataFrame(materials)
        
        # Save sample data
        sample_file = self.raw_data_dir / 'sample_data.csv'
        df.to_csv(sample_file, index=False)
        
        print(f"✅ Generated {len(df)} sample materials")
        self.download_summary['Sample'] = len(df)
        return df
    
    def download_all(self, sources=None):
        """Download from all specified sources"""
        if sources is None:
            sources = CONFIG['data_sources']
        
        print("\n" + "="*60)
        print("STARTING DATA DOWNLOAD")
        print("="*60)
        print(f"Sources: {', '.join(sources)}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        all_data = []
        
        # JARVIS-2D
        if 'jarvis' in sources:
            df = self.download_jarvis()
            if not df.empty:
                all_data.append(df)
        
        # C2DB (manual download)
        if 'c2db' in sources:
            self.download_c2db_info()
        
        # Sample data
        if 'sample' in sources or len(all_data) == 0:
            df = self.download_sample_data()
            if not df.empty:
                all_data.append(df)
        
        # Combine all data
        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            self.save_raw_data()
            self.save_download_summary()
        else:
            logger.warning("No data was downloaded")
            print("\n⚠️  No data was downloaded")
        
        return self.data
    
    def save_raw_data(self):
        """Save combined raw data"""
        if not self.data.empty:
            output_file = self.raw_data_dir / 'all_materials_raw.csv'
            self.data.to_csv(output_file, index=False)
            
            # Also save as pickle
            pickle_file = output_file.with_suffix('.pkl')
            self.data.to_pickle(pickle_file)
            
            logger.info(f"Saved combined data: {output_file}")
            print(f"\n💾 Combined data saved:")
            print(f"   CSV: {output_file}")
            print(f"   Pickle: {pickle_file}")
    
    def save_download_summary(self):
        """Save download summary"""
        summary_file = self.raw_data_dir / 'download_summary.json'
        summary = {
            'timestamp': datetime.now().isoformat(),
            'sources': self.download_summary,
            'total_materials': len(self.data),
            'columns': list(self.data.columns)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Download Summary:")
        for source, count in self.download_summary.items():
            print(f"   {source}: {count}")
        print(f"   Total: {len(self.data)} materials")
    
    def process_data(self):
        """Process and clean data"""
        if self.data.empty:
            logger.warning("No data to process")
            return self.data
        
        print("\n" + "="*60)
        print("PROCESSING DATA")
        print("="*60)
        
        df = self.data.copy()
        initial_count = len(df)
        
        # 1. Standardize column names
        column_mapping = {
            'band_gap_opt': 'band_gap',
            'band_gap_mbj': 'band_gap_mbj',
            'formation_energy_peratom': 'formation_energy',
            'ehull': 'energy_above_hull'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # 2. Create unified band_gap column
        if 'band_gap' not in df.columns and 'band_gap_mbj' in df.columns:
            df['band_gap'] = df['band_gap_mbj']
        
        # 3. Remove duplicates
        df = df.drop_duplicates(subset=['formula'], keep='first')
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            print(f"✓ Removed {duplicates_removed} duplicate formulas")
        
        # 4. Filter by band gap range
        if 'band_gap' in df.columns:
            df = df[(df['band_gap'] >= CONFIG['band_gap_min']) & 
                   (df['band_gap'] <= CONFIG['band_gap_max'])]
            print(f"✓ Filtered band gap range: {CONFIG['band_gap_min']}-{CONFIG['band_gap_max']} eV")
        
        # 5. Filter by stability
        if 'energy_above_hull' in df.columns:
            stable_count = len(df)
            df = df[df['energy_above_hull'] <= CONFIG['stability_threshold']]
            print(f"✓ Filtered by stability: {stable_count - len(df)} unstable materials removed")
        
        # 6. Add data quality flags
        df['has_band_gap'] = df['band_gap'].notna()
        df['has_formation_energy'] = df['formation_energy'].notna()
        df['is_complete'] = df['has_band_gap'] & df['has_formation_energy']
        
        # 7. Save processed data
        output_file = self.processed_data_dir / 'materials_processed.csv'
        df.to_csv(output_file, index=False)
        df.to_pickle(output_file.with_suffix('.pkl'))
        
        print(f"\n📊 Processing Summary:")
        print(f"   Initial materials: {initial_count}")
        print(f"   After processing: {len(df)}")
        print(f"   Complete samples: {df['is_complete'].sum()}")
        print(f"\n💾 Processed data saved: {output_file}")
        
        logger.info(f"Processing complete: {len(df)} materials")
        
        return df

def main():
    """Main function for standalone execution"""
    downloader = MaterialsDataDownloader()
    
    # Download data
    data = downloader.download_all()
    
    if not data.empty:
        # Process data
        processed = downloader.process_data()
        print(f"\n✅ Data download and processing complete!")
        return processed
    else:
        print("\n❌ No data downloaded")
        return None

if __name__ == "__main__":
    main()
