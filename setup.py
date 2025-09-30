#!/usr/bin/env python3
"""
Dependency installer for Doc Management System
Run this script before using the application
"""

import subprocess
import sys
import os

def main():
    print("🔧 Doc Management System - Dependency Installer")
    print("=" * 50)
    
    requirements_file = "requirements.txt"
    
    if not os.path.exists(requirements_file):
        print(f"❌ Error: {requirements_file} not found!")
        print("Please make sure you're in the correct directory")
        sys.exit(1)
    
    print("📦 The following dependencies will be installed:")
    with open(requirements_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                print(f"   - {line.strip()}")
    
    print("\n⚠️  This may take a few minutes...")
    
    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        
        print("✅ All dependencies installed successfully!")
        print("🚀 You can now run the application with: python main.py")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed with error: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()