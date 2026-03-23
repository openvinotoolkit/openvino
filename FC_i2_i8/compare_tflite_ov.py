"""
Compare TFLite vs OpenVINO model outputs for the i2/i8 FC model.

Usage:
    python3 compare_tflite_ov.py [--seed SEED] [--device DEVICE]
    python3 compare_tflite_ov.py --tflite-only
    python3 compare_tflite_ov.py --ov-only [--device CPU]

Both models take input shape (1, 1, 1536) int8 and produce (1, 1, 12288) int8.
"""

import argparse
import numpy as np

TFLITE_MODEL = "sample_i2_i8_fc.tflite"
OV_XML = "model1804289383.xml"

# Quantization parameters (from TFLite model metadata)
INPUT_SCALE = 0.05792360380291939
INPUT_ZERO_POINT = 0
OUTPUT_SCALE = 0.039862215518951416
OUTPUT_ZERO_POINT = 0


def run_tflite(input_data: np.ndarray) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=TFLITE_MODEL)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], input_data)
    interp.invoke()
    return interp.get_tensor(out["index"])


def run_openvino(input_data: np.ndarray, device: str) -> np.ndarray:
    import openvino as ov
    core = ov.Core()
    model = core.read_model(OV_XML)
    compiled = core.compile_model(model, device)
    infer_req = compiled.create_infer_request()
    infer_req.infer({0: input_data})
    return infer_req.get_output_tensor(0).data


def compare(tflite_out: np.ndarray, ov_out: np.ndarray) -> None:
    tfl = tflite_out.flatten().astype(np.int32)
    ov_ = ov_out.flatten().astype(np.int32)

    diff = np.abs(tfl - ov_)
    max_diff = diff.max()
    mean_diff = diff.mean()
    exact_match = (tfl == ov_).sum()
    total = tfl.size

    print(f"\n{'='*50}")
    print(f"  Output shape  : {tflite_out.shape}")
    print(f"  Max |diff|    : {max_diff}")
    print(f"  Mean |diff|   : {mean_diff:.4f}")
    print(f"  Exact matches : {exact_match} / {total}  ({100*exact_match/total:.1f}%)")

    # Dequantized comparison (float)
    tfl_f = (tfl - OUTPUT_ZERO_POINT) * OUTPUT_SCALE
    ov_f  = (ov_ - OUTPUT_ZERO_POINT) * OUTPUT_SCALE
    cos_sim = float(
        np.dot(tfl_f, ov_f) / (np.linalg.norm(tfl_f) * np.linalg.norm(ov_f) + 1e-12)
    )
    print(f"  Cosine sim    : {cos_sim:.6f}")
    print(f"  Max float diff: {np.abs(tfl_f - ov_f).max():.6f}")
    print(f"{'='*50}\n")

    # Show first 16 values side-by-side
    n = 16
    print(f"First {n} output values (int8):")
    print(f"  TFLite : {tfl[:n].tolist()}")
    print(f"  OV     : {ov_[:n].tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed for input")
    parser.add_argument("--device", default="CPU", help="OpenVINO device (CPU, NPU, …)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tflite-only", action="store_true", help="Run TFLite inference only")
    mode.add_argument("--ov-only", action="store_true", help="Run OpenVINO inference only")
    args = parser.parse_args()

    run_tfl = not args.ov_only
    run_ov  = not args.tflite_only

    rng = np.random.default_rng(args.seed)
    # Uniform random int8 input in full range
    input_data = rng.integers(-128, 128, size=(1, 1, 1536), dtype=np.int8)

    print(f"Input shape : {input_data.shape}, dtype: {input_data.dtype}")
    print(f"Input range : [{input_data.min()}, {input_data.max()}]")

    tflite_out = None
    ov_out = None

    if run_tfl:
        print("\n[TFLite] Running inference...")
        tflite_out = run_tflite(input_data)
        print(f"  Output shape: {tflite_out.shape}, dtype: {tflite_out.dtype}")
        print(f"  Output range: [{tflite_out.min()}, {tflite_out.max()}]")

    if run_ov:
        print(f"\n[OpenVINO ({args.device})] Running inference...")
        ov_out = run_openvino(input_data, args.device)
        print(f"  Output shape: {ov_out.shape}, dtype: {ov_out.dtype}")
        print(f"  Output range: [{ov_out.min()}, {ov_out.max()}]")

    if run_tfl and run_ov:
        compare(tflite_out, ov_out)
    elif run_tfl:
        out = tflite_out.flatten()
        print(f"\nFirst 16 output values (TFLite): {out[:16].tolist()}")
    else:
        out = ov_out.flatten()
        print(f"\nFirst 16 output values (OV): {out[:16].tolist()}")


if __name__ == "__main__":
    main()
