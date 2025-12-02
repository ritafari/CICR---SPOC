import os
import glob
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRAINING_IMAGES_DIR, TRAINING_LABELS_DIR

def fix_and_rename_labels():
    """Your existing fix_labels.py logic goes here"""
    print(f"🔧 Fixing labels in: {TRAINING_LABELS_DIR}")
    
    # Get all label files
    label_files = glob.glob(os.path.join(TRAINING_LABELS_DIR, '*.txt'))
    
    fixed_count = 0
    for label_file in label_files:
        try:
            filename = os.path.basename(label_file)
            
            # Extract image number from filename patterns like "0e2f75c6-img_154.txt"
            if '-img_' in filename:
                # Get the number part
                number_part = filename.split('img_')[1].split('.')[0]
                image_number = int(number_part)
                
                # New label name matching the image
                new_label_name = f"img_{image_number:03d}.txt"
                new_label_path = os.path.join(TRAINING_LABELS_DIR, new_label_name)
                
                # Read and fix class IDs
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                fixed_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Convert class 15 to 0 (ID) - adjust this logic as needed
                        # If you need some to be class 1 (Picture), add conditional logic here
                        old_class = int(parts[0])
                        if old_class == 15:
                            parts[0] = '0'  # Change to ID class
                        fixed_lines.append(' '.join(parts) + '\n')
                
                # Write to new file
                with open(new_label_path, 'w') as f:
                    f.writelines(fixed_lines)
                
                # Remove old file
                os.remove(label_file)
                
                fixed_count += 1
                print(f"✅ Fixed: {filename} -> {new_label_name}")
                
        except Exception as e:
            print(f"❌ Error processing {label_file}: {e}")
    
    print(f"\n🎉 Fixed and renamed {fixed_count} label files")

if __name__ == "__main__":
    fix_and_rename_labels()