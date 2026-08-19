#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from polygraphy.json import load_json

# Configurable list of scales and their row counts
# Final tensor shape: [1, 544000, 6]
SCALES = [
    {"name": "320x320 (model.36/m.0)", "rows": 409600},
    {"name": "160x160 (model.36/m.1)", "rows": 102400},
    {"name": "80x80 (model.36/m.2)", "rows": 25600},
    {"name": "40x40 (model.36/m.3)", "rows": 6400}
]

COL_NAMES = ['x', 'y', 'w', 'h', 'conf', 'cls']
COL_TOLS = [2.0, 2.0, 2.0, 2.0, 1e-2, 1e-2]

def find_tensor(data, target_shape=(544000, 6)):
    """
    Recursively search Polygraphy JSON deserialized structures to find the output ndarray.
    Handles standard dicts, Polygraphy custom objects, lists, and tuples.
    """
    if isinstance(data, np.ndarray):
        # Handle both [1, 544000, 6] and [544000, 6]
        if data.shape == target_shape or (len(data.shape) == 3 and data.shape[1:] == target_shape):
            return data.squeeze() if len(data.shape) == 3 else data
    elif isinstance(data, dict):
        for _, v in data.items():
            res = find_tensor(v, target_shape)
            if res is not None:
                return res
    elif hasattr(data, 'items'):
        for _, v in data.items():
            res = find_tensor(v, target_shape)
            if res is not None:
                return res
    elif isinstance(data, (list, tuple)):
        for item in data:
            res = find_tensor(item, target_shape)
            if res is not None:
                return res
    return None

def analyze_columns(onnx_slice, trt_slice, prefix=""):
    """
    Computes and prints per-column statistics for a given tensor slice.
    """
    diffs = np.abs(onnx_slice - trt_slice)
    
    print(f"\n--- {prefix} Column Analysis ---")
    print(f"{'Column':<6} | {'Max Diff':<12} | {'Mean Diff':<12} | {'Tolerance':<10} | {'Pass Rate'}")
    print("-" * 65)
    
    for i, (col, tol) in enumerate(zip(COL_NAMES, COL_TOLS)):
        col_diff = diffs[:, i]
        max_diff = np.nanmax(col_diff)
        mean_diff = np.nanmean(col_diff)
        pass_rate = np.mean(col_diff <= tol) * 100
        print(f"{col:<6} | {max_diff:<12.5f} | {mean_diff:<12.5f} | {tol:<10} | {pass_rate:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Compare ONNX vs TRT Polygraphy outputs.")
    parser.add_argument("onnx_json", help="Path to ONNX Polygraphy output JSON.")
    parser.add_argument("trt_json", help="Path to TRT Polygraphy output JSON.")
    parser.add_argument("-o", "--output", help="Optional output file. Overrides stdout if provided.")
    
    args = parser.parse_args()

    # Redirect stdout if an output file is explicitly requested via -o/--output
    # (Standard bash > redirection works natively without this)
    if args.output:
        sys.stdout = open(args.output, 'w')

    try:
        print("Loading Polygraphy JSON files...")
        onnx_data = load_json(args.onnx_json)
        trt_data = load_json(args.trt_json)

        onnx_tensor = find_tensor(onnx_data)
        trt_tensor = find_tensor(trt_data)

        if onnx_tensor is None or trt_tensor is None:
            print("Error: Could not find target tensor of shape (1, 544000, 6) in one or both files.")
            sys.exit(1)

        # 1. Overall Summary
        total_diff = np.abs(onnx_tensor - trt_tensor)
        overall_tol = 1e-2
        pass_rate = np.mean(total_diff <= overall_tol) * 100

        print("\n=== OVERALL TENSOR SUMMARY ===")
        print(f"Shape matched: {onnx_tensor.shape}")
        print(f"Max Difference:    {np.nanmax(total_diff):.5f}")
        print(f"Mean Difference:   {np.nanmean(total_diff):.5f}")
        print(f"Median Difference: {np.nanmedian(total_diff):.5f}")
        print(f"Values <= {overall_tol}:   {pass_rate:.2f}%")

        # 2. Per-Column Summary (Entire Tensor)
        analyze_columns(onnx_tensor, trt_tensor, prefix="FULL TENSOR")

        # 3. Sliced Summary
        start_idx = 0
        for scale in SCALES:
            end_idx = start_idx + scale['rows']
            onnx_slice = onnx_tensor[start_idx:end_idx, :]
            trt_slice = trt_tensor[start_idx:end_idx, :]
            
            analyze_columns(onnx_slice, trt_slice, prefix=f"SCALE {scale['name']}")
            start_idx = end_idx

        # 4. Top 10 Conf Column Outliers
        conf_idx = 4
        conf_diff = total_diff[:, conf_idx]
        
        # Handle NaNs in sorting by substituting them with -inf so they go to the bottom, 
        # or handle them specifically if quantisation causes NaNs.
        top10_indices = np.argsort(np.nan_to_num(conf_diff, nan=-1.0))[-10:][::-1]

        print("\n=== TOP 10 OUTLIERS (CONFIDENCE COLUMN) ===")
        print(f"{'Row Index':<12} | {'ONNX Score':<12} | {'TRT Score':<12} | {'Abs Diff'}")
        print("-" * 55)
        for idx in top10_indices:
            o_val = onnx_tensor[idx, conf_idx]
            t_val = trt_tensor[idx, conf_idx]
            d_val = conf_diff[idx]
            print(f"{idx:<12} | {o_val:<12.5f} | {t_val:<12.5f} | {d_val:.5f}")

    finally:
        if args.output:
            sys.stdout.close()

if __name__ == "__main__":
    main()