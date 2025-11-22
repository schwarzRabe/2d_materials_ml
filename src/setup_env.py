#!/usr/bin/env python3
"""
Environment setup for 2D Materials ML project
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required (current: {sys.version})")
        return False
    print(f"✅ Python {sys.version}")
    return True

def create_directories():
    """Create project directory structure"""
    # Import from config to ensure consistency
    try:
        from config import (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                          OUTPUT_DIR, PLOTS_DIR, LOGS_DIR, MODELS_DIR)
    except ImportError:
        # If config not available, create manually
        base_dir = Path.cwd()
        directories = [
            base_dir / 'data' / 'raw',
            base_dir / 'data' / 'processed',
            base_dir / 'output' / 'plots',
            base_dir / 'logs',
            base_dir / 'models',
            base_dir / 'notebooks'
        ]
    else:
        directories = [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                      OUTPUT_DIR, PLOTS_DIR, LOGS_DIR, MODELS_DIR]
    
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files
        gitkeep = dir_path / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.write_text('# Keep this directory in git\n')
        print(f"✅ Directory ready: {dir_path}")

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing packages...")
    requirements_file = Path.cwd() / 'requirements.txt'
    
    if not requirements_file.exists():
        print("⚠️  requirements.txt not found")
        return False
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False

def check_installations():
    """Check if key packages are installed"""
    packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'tqdm': 'tqdm'
    }
    
    print("\n🔍 Checking installations:")
    all_installed = True
    
    for import_name, package_name in packages.items():
        try:
            __import__(import_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} not installed")
            all_installed = False
    
    return all_installed

def setup():
    """Run complete setup"""
    print("="*60)
    print("2D MATERIALS ML - ENVIRONMENT SETUP")
    print("="*60)
    
    if not check_python_version():
        return False
    
    create_directories()
    
    if not install_requirements():
        print("\n⚠️  Try manual installation:")
        print("  pip install -r requirements.txt")
        return False
    
    if check_installations():
        print("\n✅ Setup complete!")
        print("\n🚀 Next steps:")
        print("  1. Run 'python main.py' to start data download")
        print("  2. Check 'notebooks/' for analysis examples")
        return True
    
    return False

if __name__ == "__main__":
    success = setup()
    sys.exit(0 if success else 1)
