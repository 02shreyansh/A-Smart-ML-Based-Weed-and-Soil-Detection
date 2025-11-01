import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from scipy import ndimage
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.filters import sobel

class Config:
    DATASET_PATH = '/content/dataset'
    OUTPUT_PATH = '/content/processed_data'
    TARGET_SIZE = (224, 224)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42

class BotanicalFeatureExtractor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def extract_features(self, image_path):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
          
            height, width = img.shape[:2]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            mask = self._create_plant_mask(img, gray, hsv)
            features = {}
  
            morph_features = self._extract_morphological_features(mask, gray)
            features.update(morph_features)
            leaf_features = self._extract_leaf_features(mask, gray, img)
            features.update(leaf_features)
            color_features = self._extract_color_features(img, hsv, lab, mask)
            features.update(color_features)
          
            texture_features = self._extract_texture_features(gray, mask)
            features.update(texture_features)
            veg_indices = self._extract_vegetation_indices(img, mask)
            features.update(veg_indices)
            struct_features = self._extract_structural_features(mask, gray)
            features.update(struct_features)
            spatial_features = self._extract_spatial_features(mask, img)
            features.update(spatial_features)
            env_features = self._extract_environmental_features(img, mask)
            features.update(env_features)
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def _create_plant_mask(self, img, gray, hsv):
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_and(binary, green_mask)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _extract_morphological_features(self, mask, gray):
        features = {}
        labeled = label(mask)
        regions = regionprops(labeled)
        features['leaf_count'] = len(regions)
        
        if len(regions) > 0:
            largest_region = max(regions, key=lambda r: r.area)
            features['leaf_area'] = float(largest_region.area)
            minr, minc, maxr, maxc = largest_region.bbox
            features['plant_height'] = float(maxr - minr)
            total_plant_area = sum([r.area for r in regions])
            features['canopy_coverage'] = round(total_plant_area / (mask.shape[0] * mask.shape[1]), 4)
            features['compactness'] = round(largest_region.solidity, 4)
            features['orientation_angle'] = round(np.degrees(largest_region.orientation), 2)
            features['eccentricity'] = round(largest_region.eccentricity, 4)
            width_region = maxc - minc
            height_region = maxr - minr
            features['aspect_ratio'] = round(width_region / height_region if height_region > 0 else 0, 4)
            
        else:
            features['leaf_area'] = 0
            features['plant_height'] = 0
            features['canopy_coverage'] = 0
            features['compactness'] = 0
            features['orientation_angle'] = 0
            features['eccentricity'] = 0
            features['aspect_ratio'] = 0
        
        return features
    
    def _extract_leaf_features(self, mask, gray, img):
        features = {}
      
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            if len(largest_contour) > 5:
                ellipse = cv2.fitEllipse(largest_contour)
                ellipse_area = np.pi * (ellipse[1][0]/2) * (ellipse[1][1]/2)
                contour_area = cv2.contourArea(largest_contour)
                shape_factor = contour_area / ellipse_area if ellipse_area > 0 else 0
                
                if shape_factor > 0.9:
                    leaf_shape = 'elliptical'
                elif shape_factor > 0.7:
                    leaf_shape = 'ovate'
                else:
                    leaf_shape = 'irregular'
            else:
                leaf_shape = 'simple'
            
            features['leaf_shape'] = leaf_shape
            features['leaf_shape_factor'] = round(float(shape_factor) if 'shape_factor' in locals() else 0, 4)
            perimeter = cv2.arcLength(largest_contour, True)
            area = cv2.contourArea(largest_contour)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity > 0.95:
                leaf_margin = 'smooth'
            elif solidity > 0.85:
                leaf_margin = 'slightly_serrated'
            else:
                leaf_margin = 'serrated'
            
            features['leaf_margin'] = leaf_margin
            features['leaf_margin_score'] = round(float(solidity), 4)
            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            num_vertices = len(approx)
            
            if num_vertices <= 4:
                leaf_tip_geometry = 'rounded'
            elif num_vertices <= 7:
                leaf_tip_geometry = 'pointed'
            else:
                leaf_tip_geometry = 'highly_pointed'
            
            features['leaf_tip_geometry'] = leaf_tip_geometry
            features['leaf_tip_count'] = num_vertices
            
        else:
            features['leaf_shape'] = 'unknown'
            features['leaf_shape_factor'] = 0
            features['leaf_margin'] = 'unknown'
            features['leaf_margin_score'] = 0
            features['leaf_tip_geometry'] = 'unknown'
            features['leaf_tip_count'] = 0
        labeled = label(mask)
        regions = regionprops(labeled)
        
        if len(regions) >= 2:
            centroids = [r.centroid for r in regions]
            centroids = np.array(centroids)
            if len(centroids) > 2:
                y_coords = centroids[:, 0]
                y_variance = np.var(y_coords)
                
                if y_variance < 100:
                    leaf_arrangement = 'opposite'
                elif len(regions) > 4:
                    leaf_arrangement = 'whorled'
                else:
                    leaf_arrangement = 'alternate'
            else:
                leaf_arrangement = 'basal'
        else:
            leaf_arrangement = 'single'
        
        features['leaf_arrangement'] = leaf_arrangement
        skeleton = skeletonize(mask // 255)
        branch_points = self._detect_branch_points(skeleton)
        features['branching_points'] = len(branch_points)
        
        if len(branch_points) == 0:
            branching_pattern = 'unbranched'
        elif len(branch_points) <= 2:
            branching_pattern = 'simple_branched'
        else:
            branching_pattern = 'highly_branched'
        
        features['branching_pattern'] = branching_pattern
        stem_visible = self._detect_stem(mask, gray)
        features['stem_visible'] = int(stem_visible)
        if len(regions) > 0:
            image_center = np.array([mask.shape[0]/2, mask.shape[1]/2])
            distances = [np.linalg.norm(image_center - r.centroid) for r in regions]
            features['petiole_length'] = round(float(np.mean(distances)), 2)
        else:
            features['petiole_length'] = 0

        vein_pattern = self._detect_vein_pattern(img, mask)
        features['vein_pattern'] = vein_pattern
        features['vein_density'] = round(float(vein_pattern.split('_')[0]) if '_' in vein_pattern else 0, 4)
        
        return features
    
    def _extract_color_features(self, img, hsv, lab, mask):
        features = {}
        plant_pixels_bgr = img[mask > 0]
        plant_pixels_hsv = hsv[mask > 0]
        plant_pixels_lab = lab[mask > 0]
        
        if len(plant_pixels_bgr) > 0:
            green_channel = plant_pixels_bgr[:, 1]
            features['green_mean'] = round(float(np.mean(green_channel)), 2)
            features['green_std'] = round(float(np.std(green_channel)), 2)
            mean_r = np.mean(plant_pixels_bgr[:, 2])
            mean_g = np.mean(plant_pixels_bgr[:, 1])
            mean_b = np.mean(plant_pixels_bgr[:, 0])
            features['green_ratio'] = round(float(mean_g / (mean_r + mean_b + 1e-6)), 4)
            features['color_variance'] = round(float(np.var(plant_pixels_bgr)), 2)
            features['hue_mean'] = round(float(np.mean(plant_pixels_hsv[:, 0])), 2)
            features['saturation_mean'] = round(float(np.mean(plant_pixels_hsv[:, 1])), 2)
            features['value_mean'] = round(float(np.mean(plant_pixels_hsv[:, 2])), 2)
            hist_hue = np.histogram(plant_pixels_hsv[:, 0], bins=180, range=(0, 180))[0]
            features['color_histogram_peak'] = int(np.argmax(hist_hue))
            num_patches = self._count_color_patches(img, mask)
            features['dominant_patches'] = num_patches
            
        else:
            features['green_mean'] = 0
            features['green_std'] = 0
            features['green_ratio'] = 0
            features['color_variance'] = 0
            features['hue_mean'] = 0
            features['saturation_mean'] = 0
            features['value_mean'] = 0
            features['color_histogram_peak'] = 0
            features['dominant_patches'] = 0
        
        return features
    
    def _extract_texture_features(self, gray, mask):
        features = {}
        masked_gray = gray.copy()
        masked_gray[mask == 0] = 0
        if np.sum(mask) > 0:
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(masked_gray, distances, angles, levels=256, symmetric=True, normed=True)
            features['glcm_contrast'] = round(float(graycoprops(glcm, 'contrast').mean()), 4)
            features['glcm_homogeneity'] = round(float(graycoprops(glcm, 'homogeneity').mean()), 4)
            features['glcm_energy'] = round(float(graycoprops(glcm, 'energy').mean()), 4)
            features['glcm_correlation'] = round(float(graycoprops(glcm, 'correlation').mean()), 4)
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(masked_gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp[mask > 0], bins=n_points + 2, range=(0, n_points + 2), density=True)
            features['LBP_texture'] = round(float(np.mean(lbp_hist)), 4)
            features['LBP_uniformity'] = round(float(np.max(lbp_hist)), 4)
            edges = cv2.Canny(masked_gray, 50, 150)
            edge_density = np.sum(edges > 0) / np.sum(mask > 0) if np.sum(mask) > 0 else 0
            features['edge_density'] = round(float(edge_density), 4)
            
        else:
            features['glcm_contrast'] = 0
            features['glcm_homogeneity'] = 0
            features['glcm_energy'] = 0
            features['glcm_correlation'] = 0
            features['LBP_texture'] = 0
            features['LBP_uniformity'] = 0
            features['edge_density'] = 0
        
        return features
    
    def _extract_vegetation_indices(self, img, mask):
        features = {}

        if np.sum(mask) > 0:
            plant_pixels = img[mask > 0]
            
            R = plant_pixels[:, 2].astype(float)
            G = plant_pixels[:, 1].astype(float)
            B = plant_pixels[:, 0].astype(float)

            ndvi = (G - R) / (G + R + 1e-6)
            features['NDVI'] = round(float(np.mean(ndvi)), 4)
            grvi = (G - R) / (G + R + 1e-6)
            features['GRVI'] = round(float(np.mean(grvi)), 4)
            exg = 2*G - R - B
            features['ExG'] = round(float(np.mean(exg)), 2)
            vari = (G - R) / (G + R - B + 1e-6)
            features['VARI'] = round(float(np.mean(vari)), 4)
            
        else:
            features['NDVI'] = 0
            features['GRVI'] = 0
            features['ExG'] = 0
            features['VARI'] = 0
        
        return features
    
    def _extract_structural_features(self, mask, gray):
        features = {}
        
        total_area = np.sum(mask > 0)
        image_area = mask.shape[0] * mask.shape[1]
        coverage_ratio = total_area / image_area
        labeled = label(mask)
        num_regions = len(np.unique(labeled)) - 1 
        
        if coverage_ratio < 0.1:
            growth_stage = 'seedling'
        elif coverage_ratio < 0.3:
            growth_stage = 'juvenile'
        elif coverage_ratio < 0.6:
            growth_stage = 'mature'
        else:
            growth_stage = 'overgrown'
        
        features['growth_stage'] = growth_stage
        features['growth_stage_score'] = round(float(coverage_ratio), 4)
        features['occlusion'] = num_regions if num_regions > 1 else 0
        
        return features
    
    def _extract_spatial_features(self, mask, img):
        features = {}
        labeled = label(mask)
        regions = regionprops(labeled)
        
        if len(regions) > 0:
            centroids = np.array([r.centroid for r in regions])
            if len(centroids) > 1:
                spatial_spread = np.std(centroids, axis=0).mean()
                features['spatial_spread'] = round(float(spatial_spread), 2)
            else:
                features['spatial_spread'] = 0

            center_of_mass = centroids.mean(axis=0)
            image_center = np.array([mask.shape[0]/2, mask.shape[1]/2])
            features['center_offset'] = round(float(np.linalg.norm(center_of_mass - image_center)), 2)
            
        else:
            features['spatial_spread'] = 0
            features['center_offset'] = 0
        
        return features
    
    def _extract_environmental_features(self, img, mask):
        features = {}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80:
            lighting_condition = 'low_light'
        elif mean_brightness < 160:
            lighting_condition = 'normal'
        else:
            lighting_condition = 'bright'
        
        features['lighting_condition'] = lighting_condition
        features['brightness_level'] = round(float(mean_brightness), 2)
        dark_threshold = 50
        shadow_mask = gray < dark_threshold
        shadow_percentage = np.sum(shadow_mask) / (img.shape[0] * img.shape[1])
        features['shadow_percentage'] = round(float(shadow_percentage), 4)
        background_ratio = 1 - (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1]))
        features['neighboring_context'] = round(float(background_ratio), 4)
        
        return features
    
    def _detect_branch_points(self, skeleton):
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])
        
        filtered = ndimage.convolve(skeleton.astype(float), kernel, mode='constant')
        branch_points = np.argwhere(filtered > 12)
        return branch_points
    
    def _detect_stem(self, mask, gray):
        height, width = mask.shape
        center_col = width // 2
        center_region = mask[:, center_col-10:center_col+10]
        vertical_sum = np.sum(center_region, axis=1)
        continuous_pixels = np.sum(vertical_sum > 0)
        
        return continuous_pixels > height * 0.3
    
    def _detect_vein_pattern(self, img, mask):
        green = img[:, :, 1]

        green_masked = green.copy()
        green_masked[mask == 0] = 0
        kernel = np.ones((3, 3), np.uint8)
        tophat = cv2.morphologyEx(green_masked, cv2.MORPH_TOPHAT, kernel)
        vein_density = np.sum(tophat > 20) / np.sum(mask > 0) if np.sum(mask) > 0 else 0
        
        if vein_density < 0.05:
            return '0.0_low_density'
        elif vein_density < 0.15:
            return f'{vein_density:.3f}_parallel'
        else:
            return f'{vein_density:.3f}_reticulate'
    
    def _count_color_patches(self, img, mask):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hue = hsv[:, :, 0]
        hue[mask == 0] = 0
        hue_quantized = (hue // 20) * 20 
        unique_colors = len(np.unique(hue_quantized[mask > 0]))
        return unique_colors
    
    def preprocess_image(self, image_path, output_path):
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False

            img_resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            img_denoised = cv2.fastNlMeansDenoisingColored(img_resized, None, 10, 10, 7, 21)
            cv2.imwrite(str(output_path), img_denoised)
            return True
            
        except Exception as e:
            print(f"Error preprocessing {image_path}: {str(e)}")
            return False


class DatasetProcessor:
    def __init__(self, config):
        self.config = config
        self.extractor = BotanicalFeatureExtractor(config.TARGET_SIZE)
        self.create_directories()
    
    def create_directories(self):
        paths = [
            self.config.OUTPUT_PATH,
            f"{self.config.OUTPUT_PATH}/images/train",
            f"{self.config.OUTPUT_PATH}/images/val",
            f"{self.config.OUTPUT_PATH}/images/test",
            f"{self.config.OUTPUT_PATH}/annotations"
        ]
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    def find_images(self, root_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        root = Path(root_path)
        
        for ext in image_extensions:
            image_files.extend(list(root.rglob(f'*{ext}')))
            image_files.extend(list(root.rglob(f'*{ext.upper()}')))
        
        return image_files
    
    def get_class_from_path(self, image_path):
        return Path(image_path).parts[-2] if len(Path(image_path).parts) >= 2 else "unknown"
    
    def process_dataset(self):
        print("=" * 70)
        print("ADVANCED WEED DETECTION - BOTANICAL FEATURE EXTRACTION")
        print("=" * 70)
        print("\n[1/5] Finding images...")
        image_paths = self.find_images(self.config.DATASET_PATH)
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            print("ERROR: No images found!")
            return
        print("\n[2/5] Extracting 40+ botanical features...")
        print("Features: leaf morphology, color indices, texture, vegetation indices, etc.")
        data_records = []
        
        for img_path in tqdm(image_paths, desc="Processing"):
            features = self.extractor.extract_features(img_path)
            
            if features:
                features['image_filename'] = img_path.name
                features['original_path'] = str(img_path)
                features['class_name'] = self.get_class_from_path(img_path)
                data_records.append(features)
        
        print(f"Successfully extracted features from {len(data_records)} images")
        df = pd.DataFrame(data_records)
        cols = ['image_filename', 'class_name', 'original_path'] + \
               [col for col in df.columns if col not in ['image_filename', 'class_name', 'original_path']]
        df = df[cols]
        annotations_path = f"{self.config.OUTPUT_PATH}/annotations/full_annotations.csv"
        df.to_csv(annotations_path, index=False)
        print(f"\n[3/5] Saved annotations: {annotations_path}")
        self.display_feature_stats(df)
        print("\n[4/5] Splitting dataset...")
        train_df, val_df, test_df = self.split_dataset(df)
        
        print(f"  Train: {len(train_df)} images")
        print(f"  Val:   {len(val_df)} images")
        print(f"  Test:  {len(test_df)} images")
        train_df.to_csv(f"{self.config.OUTPUT_PATH}/annotations/train_annotations.csv", index=False)
        val_df.to_csv(f"{self.config.OUTPUT_PATH}/annotations/val_annotations.csv", index=False)
        test_df.to_csv(f"{self.config.OUTPUT_PATH}/annotations/test_annotations.csv", index=False)
        print("\n[5/5] Preprocessing images...")
        self.copy_preprocessed_images(train_df, 'train')
        self.copy_preprocessed_images(val_df, 'val')
        self.copy_preprocessed_images(test_df, 'test')
        self.generate_summary_report(df, train_df, val_df, test_df)
        
        print("\n" + "=" * 70)
        print("✓ PROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nExtracted {len(df.columns) - 3} features per image")
        print(f"Output: {self.config.OUTPUT_PATH}")
    
    def split_dataset(self, df):
        train_df, temp_df = train_test_split(
            df, test_size=(self.config.VAL_RATIO + self.config.TEST_RATIO),
            random_state=self.config.RANDOM_SEED, stratify=df['class_name']
        )
        
        val_ratio_adjusted = self.config.VAL_RATIO / (self.config.VAL_RATIO + self.config.TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df, test_size=(1 - val_ratio_adjusted),
            random_state=self.config.RANDOM_SEED, stratify=temp_df['class_name']
        )
        
        return train_df, val_df, test_df
    
    def copy_preprocessed_images(self, df, split_name):
        output_dir = f"{self.config.OUTPUT_PATH}/images/{split_name}"
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            original_path = row['original_path']
            output_path = os.path.join(output_dir, row['image_filename'])
            self.extractor.preprocess_image(original_path, output_path)
    
    def display_feature_stats(self, df):
        print("\n" + "=" * 70)
        print("EXTRACTED FEATURES SUMMARY")
        print("=" * 70)
        key_features = [
            'leaf_count', 'leaf_area', 'plant_height', 'canopy_coverage',
            'green_ratio', 'NDVI', 'GRVI', 'edge_density',
            'glcm_contrast', 'LBP_texture', 'compactness'
        ]
        
        print("\nNumerical Features:")
        for feat in key_features:
            if feat in df.columns:
                print(f"  {feat:25s}: mean={df[feat].mean():8.3f}, "
                      f"std={df[feat].std():8.3f}, "
                      f"min={df[feat].min():8.3f}, "
                      f"max={df[feat].max():8.3f}")
        
        categorical_features = ['leaf_shape', 'leaf_margin', 'leaf_tip_geometry',
                               'leaf_arrangement', 'branching_pattern', 
                               'growth_stage', 'lighting_condition']
        
        print("\nCategorical Features:")
        for feat in categorical_features:
            if feat in df.columns:
                values = df[feat].value_counts().head(3)
                print(f"  {feat:25s}: {dict(values)}")
        
        print(f"\nClass Distribution:")
        for class_name, count in df['class_name'].value_counts().items():
            print(f"  {class_name:30s}: {count:5d} images")
    
    def generate_summary_report(self, df, train_df, val_df, test_df):
        report = {
            'dataset_info': {
                'total_images': len(df),
                'num_classes': df['class_name'].nunique(),
                'classes': df['class_name'].unique().tolist()
            },
            'split_info': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            },
            'feature_categories': {
                'morphological': ['leaf_count', 'leaf_area', 'plant_height', 'canopy_coverage', 'compactness'],
                'color': ['green_mean', 'green_std', 'green_ratio', 'color_variance'],
                'texture': ['edge_density', 'glcm_contrast', 'glcm_homogeneity', 'LBP_texture'],
                'vegetation_indices': ['NDVI', 'GRVI', 'ExG', 'VARI'],
                'structural': ['leaf_shape', 'leaf_margin', 'leaf_tip_geometry', 'branching_pattern'],
                'environmental': ['lighting_condition', 'shadow_percentage', 'neighboring_context']
            },
            'total_features': len(df.columns) - 3
        }
        
        report_path = f"{self.config.OUTPUT_PATH}/annotations/summary_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"\nSummary report: {report_path}")

def visualize_samples(config, n_samples=6):
    df = pd.read_csv(f"{config.OUTPUT_PATH}/annotations/train_annotations.csv")
    samples = df.sample(n=min(n_samples, len(df)))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= n_samples:
            break
        
        img_path = f"{config.OUTPUT_PATH}/images/train/{row['image_filename']}"
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        
        title = f"{row['class_name']}\n"
        title += f"Leaves: {row['leaf_count']}, Shape: {row.get('leaf_shape', 'N/A')}\n"
        title += f"Growth: {row.get('growth_stage', 'N/A')}, NDVI: {row.get('NDVI', 0):.3f}\n"
        title += f"Green Ratio: {row.get('green_ratio', 0):.3f}, Canopy: {row.get('canopy_coverage', 0):.3f}"
        
        axes[idx].set_title(title, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_PATH}/sample_visualization.png", dpi=150, bbox_inches='tight')
    print(f"\nVisualization: {config.OUTPUT_PATH}/sample_visualization.png")
    plt.show()

if __name__ == "__main__":
    config = Config()
    
    print("\n" + "="*70)
    print("ADVANCED BOTANICAL FEATURE EXTRACTION FOR WEED DETECTION")
    print("="*70)
    print("\nFeatures to be extracted:")
    print("  • Morphological: leaf_count, leaf_area, plant_height, canopy_coverage")
    print("  • Leaf: leaf_shape, leaf_margin, leaf_tip_geometry, leaf_arrangement")
    print("  • Color: green_mean, green_std, green_ratio, color_variance")
    print("  • Texture: edge_density, glcm_contrast, glcm_homogeneity, LBP_texture")
    print("  • Vegetation: NDVI, GRVI, ExG, VARI")
    print("  • Structural: branching_pattern, stem_visible, vein_pattern")
    print("  • Environmental: lighting_condition, shadow_percentage, occlusion")
    print("  • Spatial: orientation_angle, spatial_spread, center_offset")
    print("\nTotal: 50+ features\n")
    
    processor = DatasetProcessor(config)
    processor.process_dataset()
    
    print("\nGenerating visualization...")
    visualize_samples(config)
    
    print("\n" + "="*70)
    print("✓ ALL DONE! Advanced features extracted successfully")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review CSV files in annotations/")
    print("  2. Analyze feature importance")
    print("  3. Train YOLOv + CNN model")
    print("  4. Fine-tune based on feature analysis")