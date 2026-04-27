# ViT INT8 iGPU Benchmark — OpenVINO + NNCF

This sample demonstrates INT8 post-training quantization (PTQ) of a
vision-transformer (ViT-B/16) backbone using NNCF, and benchmarks FP32
vs INT8 latency and top-1 accuracy on Intel iGPU.

This addresses a gap in existing OpenVINO benchmarks, which focus
primarily on CNN architectures (ResNet, MobileNet, YOLO) and CPU targets.
As VLA models for embodied intelligence (e.g. GR00T, Pi0) rely on
vision-transformer encoders, this sample provides a validated reference
for deploying quantized ViT models on Intel Architecture.

## Requirements
```
pip install openvino nncf transformers torchvision torch
```

## Usage
```bash
python vit_int8_igpu_benchmark.py <path_to_imagenet_val> [device]
```

- `path_to_imagenet_val` — path to ImageNet validation set
- `device` — `CPU` (default) or `GPU` for Intel iGPU

## Example Output
```
[ INFO ] Available devices: ['CPU', 'GPU']
[ INFO ] [FP32/GPU] Average latency over 100 runs: 18.43 ms
[ INFO ] [INT8/GPU] Average latency over 100 runs: 9.17 ms
[ INFO ] Speedup (FP32 -> INT8): 2.01x
[ INFO ] ==================================================
[ INFO ]   Results on GPU
[ INFO ] ==================================================
[ INFO ]   Precision     Latency (ms)     Top-1 Acc
[ INFO ]   ----------------------------------------
[ INFO ]   FP32          18.43            0.8110
[ INFO ]   INT8          9.17             0.8043
[ INFO ] ==================================================
```

## References

- [OpenVINO NNCF](https://github.com/openvinotoolkit/nncf)
- [ViT-B/16 on Hugging Face](https://huggingface.co/google/vit-base-patch16-224)
- Resolves [#YOUR_ISSUE_NUMBER](https://github.com/openvinotoolkit/openvino/issues/YOUR_ISSUE_NUMBER)
```

---

**Summary of what you're pushing:**
```
samples/python/vit_int8_igpu_benchmark/
├── vit_int8_igpu_benchmark.py   ← benchmark script
└── README.md                    ← documentation