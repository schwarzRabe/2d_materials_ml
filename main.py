#!/usr/bin/env python3
"""
Main pipeline for 2D Materials ML project
Band Gap and Formation Energy Prediction
FIXED VERSION - with correct imports
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path

# Fix imports - add both current directory and src to path
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(src_dir))

# Now import modules directly from src folder
try:
    # Try importing from src package
    from src.setup_env import setup
    from src.download_data import MaterialsDataDownloader
    from src.check_data import DataValidator
    print("✓ Imported from src package")
except ImportError as e:
    print(f"Package import failed: {e}")
    try:
        # Try direct import from src directory
        import setup_env
        import download_data
        import check_data
        
        setup = setup_env.setup
        MaterialsDataDownloader = download_data.MaterialsDataDownloader
        DataValidator = check_data.DataValidator
        print("✓ Imported directly from src directory")
    except ImportError as e2:
        print(f"❌ Failed to import modules: {e2}")
        print("\nPlease check:")
        print("1. Files exist in src/ directory")
        print("2. File names are correct (without src_ prefix)")
        print("3. Python files don't have .txt extension")
        sys.exit(1)

def print_header():
    """Print project header"""
    print("\n" + "="*70)
    print(" "*20 + "2D MATERIALS ML PROJECT")
    print(" "*15 + "Band Gap & Formation Energy Prediction")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

def print_section(title):
    """Print section header"""
    print("\n" + "-"*60)
    print(f"  {title}")
    print("-"*60)

def check_environment():
    """Check if environment is properly set up"""
    print_section("ENVIRONMENT CHECK")
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version < (3, 7):
        print("⚠️  Python 3.7+ recommended")
    else:
        print("✓ Python version OK")
    
    # Check required directories
    required_dirs = ['src', 'data', 'data/raw', 'data/processed', 'output', 'output/plots']
    for dir_name in required_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"⚠️  Creating directory: {dir_name}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check for required files
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/setup_env.py',
        'src/download_data.py',
        'src/check_data.py'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = current_dir / file_name
        if file_path.exists():
            print(f"✓ File exists: {file_name}")
        else:
            print(f"❌ Missing file: {file_name}")
            missing_files.append(file_name)
    
    if missing_files:
        print("\n⚠️  Some files are missing. Please check:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nMake sure to rename files:")
        print("  src__init__.py → src/__init__.py")
        print("  src_config.py → src/config.py")
        print("  etc.")
        return False
    
    return True

def run_setup():
    """Setup environment"""
    print_section("STEP 1: ENVIRONMENT SETUP")
    
    try:
        if setup():
            print("✅ Environment setup complete")
            time.sleep(1)
            return True
        else:
            print("⚠️  Setup incomplete, but continuing...")
            return True
    except Exception as e:
        print(f"⚠️  Setup warning: {e}")
        print("Continuing anyway...")
        return True

def install_requirements():
    """Try to install requirements if missing"""
    print_section("CHECKING REQUIREMENTS")
    
    try:
        import pandas
        import numpy
        print("✓ Core packages available")
    except ImportError:
        print("Installing core packages...")
        os.system(f"{sys.executable} -m pip install pandas numpy tqdm requests matplotlib seaborn scikit-learn")
    
    try:
        import jarvis
        print("✓ JARVIS-tools available")
    except ImportError:
        print("⚠️  JARVIS-tools not installed")
        print("Installing JARVIS-tools...")
        os.system(f"{sys.executable} -m pip install jarvis-tools")

def run_download(sources=None):
    """Download data from specified sources"""
    print_section("STEP 2: DATA DOWNLOAD")
    
    if sources is None:
        sources = ['sample']
    
    print(f"Data sources: {', '.join(sources)}")
    
    try:
        downloader = MaterialsDataDownloader()
        
        # Download data
        data = downloader.download_all(sources)
        
        if data.empty:
            print("⚠️  No data downloaded")
            print("\nTrying with sample data...")
            data = downloader.download_sample_data()
            
            if data.empty:
                print("❌ Could not download any data")
                return None
        
        print(f"✅ Downloaded {len(data)} materials")
        time.sleep(1)
        
        # Process data
        print_section("STEP 3: DATA PROCESSING")
        processed = downloader.process_data()
        
        if processed.empty:
            print("❌ Data processing failed")
            return None
        
        print(f"✅ Processed {len(processed)} materials")
        return processed
        
    except Exception as e:
        print(f"❌ Download error: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to continue with sample data
        print("\n⚠️  Trying to generate sample data...")
        try:
            downloader = MaterialsDataDownloader()
            data = downloader.download_sample_data()
            if not data.empty:
                processed = downloader.process_data()
                return processed
        except:
            pass
        
        return None

def run_validation(visualize=True):
    """Validate and visualize data"""
    print_section("STEP 4: DATA VALIDATION")
    
    try:
        validator = DataValidator()
        
        # Check structure
        if not validator.check_structure():
            print("❌ No data found for validation")
            print("  Run download first: python main.py --download")
            return None
        
        # Create visualizations
        if visualize:
            print_section("STEP 5: VISUALIZATION")
            try:
                validator.visualize(save_plots=True)
                print("✅ Visualizations created")
            except Exception as e:
                print(f"⚠️  Visualization warning: {e}")
                print("Continuing without plots...")
        
        # Prepare ML data
        print_section("STEP 6: ML DATA PREPARATION")
        ml_data = validator.prepare_ml_data()
        
        if ml_data:
            print("✅ ML-ready data prepared")
            return ml_data
        else:
            print("⚠️  ML data preparation incomplete")
            return None
            
    except Exception as e:
        print(f"❌ Validation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_summary():
    """Print pipeline summary"""
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)
    
    # Check what was created
    data_dir = current_dir / 'data'
    processed_dir = data_dir / 'processed'
    plots_dir = current_dir / 'output' / 'plots'
    
    # Check processed data
    processed_file = processed_dir / 'materials_processed.csv'
    if processed_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(processed_file)
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total materials: {len(df)}")
            
            if 'is_complete' in df.columns:
                complete = df['is_complete'].sum()
                print(f"   Complete samples: {complete} ({100*complete/len(df):.1f}%)")
            
            if 'source' in df.columns:
                print(f"\n   Sources:")
                for source, count in df['source'].value_counts().items():
                    print(f"     • {source}: {count}")
        except Exception as e:
            print(f"Could not load statistics: {e}")
    else:
        print("No processed data found")
    
    # Check ML data
    ml_file = processed_dir / 'ml_ready_data.pkl'
    if ml_file.exists():
        try:
            import pickle
            with open(ml_file, 'rb') as f:
                ml_data = pickle.load(f)
            print(f"\n🤖 ML-Ready Data:")
            print(f"   Samples: {ml_data.get('n_samples', 'N/A')}")
            print(f"   Features: {ml_data.get('n_features', 'N/A')}")
        except:
            print("ML data file exists but could not load details")
    
    # Check plots
    if plots_dir.exists():
        plots = list(plots_dir.glob('*.png'))
        if plots:
            print(f"\n📈 Visualizations:")
            for plot in plots:
                print(f"   • {plot.name}")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED!")
    print("="*70)
    
    print("\n🚀 Next Steps:")
    print("   1. Review visualizations in: output/plots/")
    print("   2. Check processed data in: data/processed/")
    print("   3. Start ML training with: data/processed/ml_ready_data.pkl")
    
    print("\n📝 To load ML data in Python:")
    print("   >>> import pickle")
    print("   >>> with open('data/processed/ml_ready_data.pkl', 'rb') as f:")
    print("   >>>     ml_data = pickle.load(f)")

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(
        description='2D Materials ML Data Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline with sample data
  python main.py --sources jarvis   # Download from JARVIS
  python main.py --sources sample   # Use sample data only
  python main.py --skip-setup       # Skip environment setup
  python main.py --validate-only    # Only validate existing data
        """
    )
    
    parser.add_argument(
        '--sources', 
        nargs='+',
        choices=['jarvis', 'c2db', 'sample'],
        default=['sample'],
        help='Data sources to use (default: sample)'
    )
    
    parser.add_argument(
        '--skip-setup',
        action='store_true',
        help='Skip environment setup'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing data'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check environment'
    )
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    try:
        # Check environment first
        if not check_environment():
            print("\n⚠️  Environment check failed")
            print("Please fix the issues above and try again")
            if args.check_only:
                return 1
            print("\nTrying to continue anyway...")
        
        if args.check_only:
            print("\n✅ Environment check complete")
            return 0
        
        # Try to install missing packages
        if not args.skip_setup:
            install_requirements()
            run_setup()
        
        # Main pipeline
        if args.validate_only:
            # Only validate existing data
            ml_data = run_validation(visualize=not args.no_plots)
            if ml_data:
                print_summary()
                return 0
            else:
                print("\n⚠️  No data to validate")
                print("Run without --validate-only to download data first")
                return 1
        else:
            # Download and process data
            data = run_download(sources=args.sources)
            
            if data is not None:
                # Validate and prepare ML data
                ml_data = run_validation(visualize=not args.no_plots)
                print_summary()
                return 0
            else:
                print("\n❌ Pipeline failed - no data processed")
                return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # First, print current directory info
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {__file__}")
    print(f"Python executable: {sys.executable}")
    
    # Run main
    sys.exit(main())
