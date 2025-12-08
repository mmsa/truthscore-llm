#!/usr/bin/env python3
"""Fix metadata in distribution files to remove problematic license-file fields."""
import zipfile
import tarfile
import tempfile
import os
import shutil
from pathlib import Path

def fix_wheel(wheel_path):
    """Remove license-file fields from wheel METADATA."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract wheel
        with zipfile.ZipFile(wheel_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Fix METADATA
        metadata_path = Path(tmpdir) / f"{Path(wheel_path).stem.replace('-py3-none-any', '')}.dist-info" / "METADATA"
        if metadata_path.exists():
            lines = []
            with open(metadata_path, 'r') as f:
                for line in f:
                    if not line.startswith('License-File:') and not line.startswith('Dynamic: license-file'):
                        lines.append(line)
            
            with open(metadata_path, 'w') as f:
                f.writelines(lines)
        
        # Recreate wheel
        os.remove(wheel_path)
        with zipfile.ZipFile(wheel_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, tmpdir)
                    zip_ref.write(file_path, arc_name)
    
    print(f"Fixed {wheel_path}")

def fix_sdist(sdist_path):
    """Remove license-file fields from sdist PKG-INFO."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract sdist
        with tarfile.open(sdist_path, 'r:gz') as tar_ref:
            tar_ref.extractall(tmpdir)
        
        # Find and fix PKG-INFO
        pkg_info_path = None
        for root, dirs, files in os.walk(tmpdir):
            if 'PKG-INFO' in files:
                pkg_info_path = os.path.join(root, 'PKG-INFO')
                break
        
        if pkg_info_path:
            lines = []
            with open(pkg_info_path, 'r') as f:
                for line in f:
                    if not line.startswith('License-File:') and not line.startswith('Dynamic: license-file'):
                        lines.append(line)
            
            with open(pkg_info_path, 'w') as f:
                f.writelines(lines)
        
        # Recreate sdist
        os.remove(sdist_path)
        with tarfile.open(sdist_path, 'w:gz') as tar_ref:
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, tmpdir)
                    tar_ref.add(file_path, arc_name)
    
    print(f"Fixed {sdist_path}")

if __name__ == '__main__':
    dist_dir = Path('dist')
    for dist_file in dist_dir.glob('*'):
        if dist_file.suffix == '.whl':
            fix_wheel(dist_file)
        elif dist_file.suffix == '.gz' and 'tar' in dist_file.name:
            fix_sdist(dist_file)

