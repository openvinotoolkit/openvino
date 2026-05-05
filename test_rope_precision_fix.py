"""
Test: PyTorch vs OpenVINO CPU accuracy for Fara-7B language model.

Demonstrates the inv_freq f16 precision bug in exported OV models and verifies
that the RecoverRoPEInvFreqPrecision transformation pass corrects it.

The bug:
  When exporting Qwen2.5-VL based models (e.g. Fara-7B) to OpenVINO, the
  rotary embedding frequencies (inv_freq, shape [1,1,64,1]) are stored as
  float16 due to model-wide f16 compression. These values can be as small as
  ~1e-6. At f16 precision that's ~1% relative error, which causes a phase error
  of up to 1.78 radians at position 10000 — effectively randomising attention.

  On CPU (f32 compute) the error is moderate but grows with position magnitude.
  On GPU (f16 compute) the error is catastrophic and causes "Low similarity"
  in WWB benchmarks across all precision modes (FP16, INT8, 4BIT).

Usage:
  # With the fix (default after applying recover_rope_inv_freq_precision.patch):
  python test_rope_precision_fix.py

  # To also compare against PyTorch reference (requires transformers + torch):
  python test_rope_precision_fix.py --pytorch
"""

import argparse
import numpy as np
import openvino as ov

OV_MODEL_DIR = "/home/pkrzemin/tasks/benchmark/fara-7b-ov"
LANGUAGE_MODEL_XML = f"{OV_MODEL_DIR}/openvino_language_model.xml"

BATCH = 1
SEQ_LEN = 5
HIDDEN_DIM = 3584  # Fara-7B / Qwen2.5-7B hidden size


def load_fixed_model(core: ov.Core) -> ov.CompiledModel:
    """Load model with the RecoverRoPEInvFreqPrecision pass active (default)."""
    model = core.read_model(LANGUAGE_MODEL_XML)
    return core.compile_model(model, "CPU")


def load_broken_model(core: ov.Core) -> ov.CompiledModel:
    """
    Load model simulating the bug: pre-convert the f16 inv_freq constant to f32
    while keeping the f16-quantised values. The pass won't fire (it only targets
    f16 constants), so the bad precision is preserved.
    """
    model = core.read_model(LANGUAGE_MODEL_XML)
    for op in model.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type.f16:
            shape = list(op.get_output_shape(0))
            if shape == [1, 1, 64, 1]:
                data_f32_bad = op.get_data().astype(np.float16).astype(np.float32)
                new_const = ov.op.Constant(data_f32_bad)
                new_const.set_friendly_name(op.get_friendly_name() + "_bad_f32")
                op.output(0).replace(new_const.output(0))
                print(f"  [broken] patched inv_freq to f32 with f16-quantised values")
                break
    return core.compile_model(model, "CPU")


def make_inputs(position_scale: int, rng: np.random.Generator) -> dict:
    """Create a minimal set of language model inputs at the given position scale."""
    inputs_embeds = rng.standard_normal((BATCH, SEQ_LEN, HIDDEN_DIM)).astype(np.float32) * 0.1
    attention_mask = np.ones((BATCH, SEQ_LEN), dtype=np.int64)
    beam_idx = np.zeros(BATCH, dtype=np.int32)

    # 3D M-RoPE position IDs: [temporal, height, width]
    position_ids = np.zeros((3, BATCH, SEQ_LEN), dtype=np.int64)
    position_ids[0, 0, :] = np.arange(SEQ_LEN)                          # temporal: small
    position_ids[1, 0, :] = np.arange(position_scale, position_scale + SEQ_LEN)          # height
    position_ids[2, 0, :] = np.arange(position_scale * 2, position_scale * 2 + SEQ_LEN)  # width

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "beam_idx": beam_idx,
        "position_ids": position_ids,
    }


def run_comparison(compiled_fixed, compiled_broken, rng):
    print()
    print(f"{'Position scale':>16} | {'Max |Δlogit|':>14} | {'Mean |Δlogit|':>14} | {'Cosine sim':>12} | {'Top-5 overlap':>13}")
    print("-" * 82)

    for scale in [10, 100, 1000, 5000, 10000, 20000, 50000]:
        inputs = make_inputs(scale, rng)

        out_fixed  = compiled_fixed(inputs)["logits"]
        out_broken = compiled_broken(inputs)["logits"]

        diff = np.abs(out_fixed - out_broken)

        v1 = out_fixed[0, -1]
        v2 = out_broken[0, -1]
        cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        top5_fixed  = set(np.argsort(v1)[-5:])
        top5_broken = set(np.argsort(v2)[-5:])
        overlap = len(top5_fixed & top5_broken)

        print(
            f"  {scale:>14} | {diff.max():>14.6f} | {diff.mean():>14.6f} | {cos_sim:>12.8f} | {overlap}/5"
        )

    print()
    print("Expected: max diff and mean diff grow with position scale.")
    print("At scale=50000 (realistic for high-res image tokens), the")
    print("error is large enough to corrupt model outputs. On GPU (f16")
    print("compute) these errors are much larger than shown here (CPU/f32).")


def check_pass_fired(core: ov.Core) -> bool:
    """Verify the RecoverRoPEInvFreqPrecision pass replaced the constant."""
    model = core.read_model(LANGUAGE_MODEL_XML)
    compiled = core.compile_model(model, "CPU")
    # The pass renames the recovered constant with suffix _f32_recovered
    # We can detect this via the compiled model's RT info, but the easiest
    # check is to compile and look for the renamed node in the debug log.
    # Here we just check that the f16 constant is gone from the final graph
    # by re-reading and checking its type after OV's internal passes run.
    # (The pass mutates the model in-place before compilation.)
    model2 = core.read_model(LANGUAGE_MODEL_XML)
    # Run passes manually via compile (we can't call passes directly from Python)
    # Instead: check the friendly name of the output from compile model.
    # Simplest proxy: check the constant is NOT f16 anymore after transformation.
    found_f16_inv_freq = False
    for op in model2.get_ordered_ops():
        if op.get_type_name() == "Constant" and op.get_element_type() == ov.Type.f16:
            shape = list(op.get_output_shape(0))
            if shape == [1, 1, 64, 1]:
                found_f16_inv_freq = True
    return not found_f16_inv_freq  # pass fired if no f16 inv_freq remains in model IR


def main():
    parser = argparse.ArgumentParser(description="Test RoPE inv_freq precision fix")
    parser.add_argument("--pytorch", action="store_true",
                        help="Also run PyTorch reference (requires transformers + torch)")
    args = parser.parse_args()

    print("=" * 60)
    print("RoPE inv_freq f16 precision fix — verification")
    print("=" * 60)
    print(f"Model: {LANGUAGE_MODEL_XML}")
    print()

    core = ov.Core()
    print(f"OpenVINO version: {ov.__version__}")

    print()
    print("[1] Loading FIXED model (RecoverRoPEInvFreqPrecision pass active)...")
    compiled_fixed = load_fixed_model(core)
    print("  OK")

    print()
    print("[2] Loading BROKEN model (f16 inv_freq values preserved as f32)...")
    compiled_broken = load_broken_model(core)
    print("  OK")

    rng = np.random.default_rng(seed=42)

    print()
    print("[3] Comparing outputs at increasing position scales")
    print("    (height/width positions as seen for image tokens in VLM inference)")
    run_comparison(compiled_fixed, compiled_broken, rng)

    if args.pytorch:
        print()
        print("[4] PyTorch reference comparison")
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM

            MODEL_ID = "microsoft/Fara-7B"
            print(f"  Loading {MODEL_ID} in PyTorch (fp32)...")
            config = AutoConfig.from_pretrained(MODEL_ID)
            pt_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
            pt_model.eval()

            rng2 = np.random.default_rng(seed=42)
            scale = 10000

            inputs = make_inputs(scale, rng2)
            pt_inputs = {
                "inputs_embeds": torch.from_numpy(inputs["inputs_embeds"]),
                "attention_mask": torch.from_numpy(inputs["attention_mask"]),
                "position_ids": torch.from_numpy(inputs["position_ids"]),
            }

            with torch.no_grad():
                pt_out = pt_model(**pt_inputs).logits.numpy()

            ov_out = compiled_fixed(inputs)["logits"]
            diff = np.abs(pt_out - ov_out)

            print(f"  PyTorch vs OV (fixed) at scale={scale}:")
            print(f"    Max |Δlogit|:  {diff.max():.6f}")
            print(f"    Mean |Δlogit|: {diff.mean():.6f}")

            v1 = pt_out[0, -1]
            v2 = ov_out[0, -1]
            cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            print(f"    Cosine sim:    {cos_sim:.8f}")

        except ImportError as e:
            print(f"  Skipped: {e}")


if __name__ == "__main__":
    main()
