"""
Script to create a classification dataset structure from COCO format annotations.
Organizes images into: Dataset -> Class labels -> images
"""

import json
import os
import shutil
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

def load_coco_annotations(annotations_path: str) -> Dict:
    """Load COCO format annotations file."""
    with open(annotations_path, 'r') as f:
        return json.load(f)

def get_image_classes(annotations_data: Dict) -> Dict[str, List[str]]:
    """
    Extract the dominant class for each image based on annotations.
    Returns a dictionary mapping image filenames to their classes.
    """
    # Create mappings
    image_id_to_filename = {img['id']: img['file_name'] for img in annotations_data['images']}
    category_id_to_name = {cat['id']: cat['name'] for cat in annotations_data['categories']}
    
    # Count annotations per image per category
    image_annotations = defaultdict(list)
    for ann in annotations_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        if image_id in image_id_to_filename:
            image_annotations[image_id].append(category_id)
    
    # Determine dominant class for each image
    image_classes = {}
    for image_id, category_ids in image_annotations.items():
        filename = image_id_to_filename[image_id]
        
        # Count occurrences of each category in this image
        category_counts = Counter(category_ids)
        
        # Get the most common category (dominant class)
        if category_counts:
            dominant_category_id = category_counts.most_common(1)[0][0]
            dominant_class = category_id_to_name[dominant_category_id]
            image_classes[filename] = dominant_class
    
    return image_classes

def create_classification_structure(
    source_images_dir: str,
    target_dataset_dir: str,
    image_classes: Dict[str, str],
    copy_files: bool = True
) -> None:
    """
    Create classification dataset structure and organize images.
    
    Args:
        source_images_dir: Path to source images directory
        target_dataset_dir: Path to target classification dataset directory
        image_classes: Dictionary mapping image filenames to class names
        copy_files: If True, copy files; if False, create symbolic links
    """
    source_path = Path(source_images_dir)
    target_path = Path(target_dataset_dir)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all available classes
    classes = set(image_classes.values())
    print(f"Found {len(classes)} classes: {sorted(classes)}")
    
    # Create class directories
    class_dirs = {}
    for class_name in classes:
        # Clean class name for directory (replace spaces and special chars)
        clean_class_name = class_name.replace(' ', '_').replace('-', '_')
        class_dir = target_path / clean_class_name
        class_dir.mkdir(exist_ok=True)
        class_dirs[class_name] = class_dir
        print(f"Created directory: {class_dir}")
    
    # Organize images by class
    stats = defaultdict(int)
    missing_files = []
    
    for filename, class_name in image_classes.items():
        source_file = source_path / filename
        target_dir = class_dirs[class_name]
        target_file = target_dir / filename
        
        if source_file.exists():
            try:
                if copy_files:
                    shutil.copy2(source_file, target_file)
                else:
                    # Create symbolic link
                    if target_file.exists():
                        target_file.unlink()
                    target_file.symlink_to(source_file.absolute())
                
                stats[class_name] += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            missing_files.append(filename)
    
    # Print statistics
    print("\n=== Classification Dataset Statistics ===")
    total_images = 0
    for class_name, count in sorted(stats.items()):
        print(f"{class_name}: {count} images")
        total_images += count
    
    print(f"\nTotal images organized: {total_images}")
    
    if missing_files:
        print(f"\nWarning: {len(missing_files)} files were not found:")
        for filename in missing_files[:10]:  # Show first 10
            print(f"  - {filename}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

def main():
    """Main function to create classification dataset."""
    # Paths - using relative paths from project root
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up from code/data_preparation to project root
    annotations_file = project_root / "data" / "object_detection_dataset" / "_annotations.coco.json"
    source_images_dir = project_root / "data" / "object_detection_dataset" / "images"
    target_dataset_dir = project_root / "data" / "classification_dataset"
    
    print("Creating classification dataset structure...")
    print(f"Source annotations: {annotations_file}")
    print(f"Source images: {source_images_dir}")
    print(f"Target dataset: {target_dataset_dir}")
    
    # Load annotations
    print("\nLoading COCO annotations...")
    annotations_data = load_coco_annotations(str(annotations_file))
    
    print(f"Found {len(annotations_data['images'])} images")
    print(f"Found {len(annotations_data['categories'])} categories")
    print(f"Found {len(annotations_data['annotations'])} annotations")
    
    # Extract image classes
    print("\nExtracting dominant classes for each image...")
    image_classes = get_image_classes(annotations_data)
    print(f"Classified {len(image_classes)} images")
    
    # Create classification structure
    print("\nCreating classification dataset structure...")
    create_classification_structure(
        str(source_images_dir),
        str(target_dataset_dir),
        image_classes,
        copy_files=True  # Set to False to create symbolic links instead
    )
    
    print("\n✅ Classification dataset created successfully!")
    print(f"Dataset location: {target_dataset_dir}")
    print("\nDataset structure:")
    print("data/classification_dataset/")
    for class_dir in sorted(target_dataset_dir.glob("*")):
        if class_dir.is_dir():
            image_count = len(list(class_dir.glob("*.jpg")))
            print(f"├── {class_dir.name}/ ({image_count} images)")

if __name__ == "__main__":
    main()
