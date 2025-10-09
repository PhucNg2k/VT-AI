import pandas as pd
import numpy as np
import os
import sys
from typing import Tuple


def load_csv(csv_path: str) -> pd.DataFrame:
    """Load CSV file and validate required columns."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required_cols = ['image_filename', 'x', 'y', 'z', 'Rx', 'Ry', 'Rz']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")
    
    return df


def calculate_position_error(pred_pos: np.ndarray, gt_pos: np.ndarray) -> float:
    """Calculate Euclidean distance between predicted and ground truth positions."""
    return np.linalg.norm(pred_pos - gt_pos)


def calculate_orientation_error(pred_orient: np.ndarray, gt_orient: np.ndarray) -> float:
    """Calculate angle between predicted and ground truth orientation vectors (in degrees)."""
    # Normalize vectors
    pred_norm = pred_orient / np.linalg.norm(pred_orient)
    gt_norm = gt_orient / np.linalg.norm(gt_orient)
    
    # Calculate dot product (clamped to avoid numerical errors)
    dot_product = np.clip(np.dot(pred_norm, gt_norm), -1.0, 1.0)
    
    # Calculate angle in radians, then convert to degrees
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)

    print(f"Angle between predicted and ground truth orientation vectors: {angle_deg} degrees")
    return angle_deg


def evaluate_csvs(pred_csv_path: str, gt_csv_path: str) -> Tuple[float, float, float]:
    """
    Evaluate prediction CSV against ground truth CSV.
    
    Returns:
        Tuple of (MCE, OE, AC) scores
    """
    # Load CSV files
    pred_df = load_csv(pred_csv_path)
    gt_df = load_csv(gt_csv_path)
    
    # Merge on image_filename
    merged = pd.merge(pred_df, gt_df, on='image_filename', suffixes=('_pred', '_gt'))
    
    if len(merged) == 0:
        raise ValueError("No matching image filenames found between prediction and ground truth")
    
    print(f"Evaluating {len(merged)} matching images...")
    
    # Calculate errors for each image
    position_errors = []
    orientation_errors = []
    
    for _, row in merged.iterrows():
        # Position error
        pred_pos = np.array([row['x_pred'], row['y_pred'], row['z_pred']])
        gt_pos = np.array([row['x_gt'], row['y_gt'], row['z_gt']])
        pos_error = calculate_position_error(pred_pos, gt_pos)
        position_errors.append(pos_error)
        
        # Orientation error
        pred_orient = np.array([row['Rx_pred'], row['Ry_pred'], row['Rz_pred']])
        gt_orient = np.array([row['Rx_gt'], row['Ry_gt'], row['Rz_gt']])
        orient_error = calculate_orientation_error(pred_orient, gt_orient)
        orientation_errors.append(orient_error)
    
    # Calculate MCE and OE
    N = len(merged)
    MCE = (1.0 / N) * sum(e_i / 50.0 for e_i in position_errors)
    OE = (1.0 / N) * sum(theta_i / 20.0 for theta_i in orientation_errors)
    
    # Calculate final accuracy score
    AC = (1 - MCE) * 0.7 + (1 - OE) * 0.3
    
    return MCE, OE, AC


def main():
    # if len(sys.argv) != 3:
    #     print("Usage: python evaluate.py <prediction_csv> <ground_truth_csv>")
    #     print("Example: python evaluate.py submission.csv ground_truth.csv")
    #     sys.exit(1)
    
    #pred_csv = sys.argv[1] || "/Users/cybercs/Documents/Competition/Code/finetune_yolo/submit/Submit_train.csv"
    #gt_csv = sys.argv[2] || "/Users/cybercs/Documents/Competition/Code/finetune_yolo/submit/Submit_train_gt.csv"

    pred_csv = "/Users/cybercs/Documents/Competition/Code/finetune_yolo/submit/train_test.csv"

    gt_csv = "/Users/cybercs/Documents/Competition/Code/finetune_yolo/submit/Public_train_gt.csv"
    try:
        MCE, OE, AC = evaluate_csvs(pred_csv, gt_csv)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Mean Center Error (MCE): {MCE:.6f}")
        print(f"Orientation Error (OE):  {OE:.6f}")
        print(f"Final Accuracy (AC):    {AC:.6f}")
        print("="*50)
        
        # Additional statistics
        print(f"\nScore breakdown:")
        print(f"  Position component: {(1-MCE)*0.7:.6f}")
        print(f"  Orientation component: {(1-OE)*0.3:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
