# bench_kernel

**bench_kernel** is a correctness verification and performance benchmarking
tool for the GPU kernel primitives provided by the
[OpenVINO](https://github.com/openvinotoolkit/openvino) Intel GPU plugin.
Each GPU kernel is executed in isolation on real hardware, and the output is
compared against a CPU reference implementation.
**bench_kernel** itself is a harness for different kernel-specific drivers.

## Harness Usage

```sh
./ov_gpu_bench_kernel --KERNEL [COMMON-OPTIONS] [KERNEL-OPTIONS] SHAPES
```

where `KERNEL` is one of the [supported kernels](#supported-kernels) listed below.

`SHAPES` is a positional argument describing input dimensions in `AxBxC:DxExF`
format. The exact meaning depends on the kernel type (see
[Kernel-Specific Options](#kernel-specific-options)).

### Quick Examples

```sh
# FC performance benchmark (f16, 1x4096 * 4096x4096)
./ov_gpu_bench_kernel --fc --dt=f16 --mode=p --device=0 1x4096:4096x4096

# Eltwise correctness check
./ov_gpu_bench_kernel --eltwise --mode=c --device=1 1x32x128:1x32x128

# SDPA fast performance
./ov_gpu_bench_kernel --sdpa --mode=f --dt=f16 1x32x1x128:1x32x256x128:1x32x256x128

# Batch file execution
./ov_gpu_bench_kernel --batch=batch_fc_llm.txt --mode=c --device=1

# Run mode (single iteration, minimal overhead)
./ov_gpu_bench_kernel --softmax --mode=r 1x32x4096
```

---

## Common Options

Options supported across all kernel drivers.

### --mode

`--mode=MODE` specifies the benchmark execution mode.

| Mode | Alias | Description | Warmup | Iters |
|------|-------|-------------|--------|-------|
| `p`  | `perf` | Performance measurement (default) | 5 | 20 |
| `c`  | `acc`, `accuracy` | Correctness check against CPU reference | — | — |
| `f`  | `fast` | Fast performance (quick sanity check) | 1 | 3 |
| `cp` | `both`, `all` | Correctness + performance | 5 | 20 |
| `r`  | `run` | Run-only (single execution, no warmup) | 0 | 1 |

### --device

`--device=N` specifies the GPU device index. Default is `0`.
Use `--list-devices` to enumerate available GPUs.

### --dt

`--dt=TYPE[:TYPE[:TYPE]]` specifies data type(s), colon-separated.
Common values: `f16`, `f32`, `i8`, `u8`, `i4`, `u4`.

Examples:
- `--dt=f16` — single type for all tensors
- `--dt=f16:i4:f16` — src:weight:dst (for FC with compressed weights)

### --impl

`--impl=IMPL` forces a specific implementation backend.
Values: `ocl` (OpenCL), `onednn` (oneDNN). Empty for auto-selection (default).

### --force_impl

`--force_impl=KERNEL_NAME` forces a specific OCL kernel implementation by name.
If `--impl` is not specified (auto mode), `bench_kernel` automatically switches
the forced primitive to OCL so the requested kernel can be selected.
If `--impl=onednn` is explicitly set, `--force_impl` has no effect.

The kernel name must exactly match one of the registered GPU kernel names
(e.g., `gemm_tiled_opt`, `fully_connected_gpu_bf_tiled`,
`convolution_gpu_bfyx_os_iyx_osv16`).

Examples:
```sh
# Force a specific Gemm OCL kernel
./ov_gpu_bench_kernel --gemm --dt=f16 --shapes=8x16x64:8x64x16 \
    --force_impl=gemm_tiled_opt --mode=c --device=1

# Force a specific FC OCL kernel
./ov_gpu_bench_kernel --fc --dt=f16 --shapes=1x2048:4096x2048 \
    --force_impl=fully_connected_gpu_bf_tiled --mode=p --device=1
```

### --shapes

`--shapes=SHAPES` specifies input shapes. Can also be passed as a positional
argument. Format: `AxBxC:DxExF` where `:` separates multiple inputs.

### --batch

`--batch=FILE` reads problem definitions from a batch file, one per line.
Lines starting with `#` are comments. Each line follows the same format as
CLI arguments.

### --warmup

`--warmup=N` specifies the number of warmup iterations. Default: `5`.
Overridden by `--mode=f` (1) and `--mode=r` (0).

### --iters

`--iters=N` specifies the number of measurement iterations. Default: `20`.
Overridden by `--mode=f` (3) and `--mode=r` (1).

### --max-ms

`--max_ms=N` specifies the maximum time in milliseconds per problem.
Default: `3000`.

### --verbose

`--verbose=N` sets verbosity level.
- `0` — summary only
- `1` — per-test result (default)
- `2` — detailed accuracy/performance breakdown

### --list-devices

`--list_devices` lists available GPU devices and exits.

### --list-kernels

`--list_kernels` lists all registered kernel benchmarks and exits.

### Attribute Options

`--attr-scales=EXPR` — Scale attributes (e.g., `wei:per_ocic:f16:1x128`).

`--attr-zero-points=EXPR` — Zero point attributes (e.g., `wei:per_ocic:i4:1x128`).

`--attr-post-ops=OPS` — Fused post-operations, `+` separated.
Supported: `relu`, `gelu_erf`, `gelu_tanh`, `sigmoid`, `tanh`, `swish`,
`hardswish`, `mish`, `abs`, `sqrt`, `square`, `elu`, `clamp`, `exp`,
`sum[:dt]`, `prod`, `sub`, `div`, `max`, `min`, `quantize`.

Example: `--attr-post-ops=sum:f16+relu`

---

## Supported Kernels

**bench_kernel** supports **89 kernel types** organized into the following
categories. The **Ref** column indicates CPU reference accuracy support
(`mode=c`).

### Compute Kernels

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--fc` | fully_connected | Yes* | `input:weight` | FC with compressed weight & post-op support |
| `--gemm` | gemm | Yes* | `A:B[:C]` | General matrix multiply |
| `--conv` | convolution | Yes* | `input:filter` | N-D convolution |
| `--deconv` | deconvolution | — | `input:filter` | Transposed convolution |
| `--sdpa` | scaled_dot_product_attention | Yes | `Q:K:V` | Multi-head attention |

\* FC/Gemm/Conv support post-ops; some post-op configs report `UNIMPLEMENTED`.

### Normalization & Activation

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--softmax` | softmax | Yes | `input` | Softmax along axis |
| `--activation` | activation | Yes | `input` | Element-wise activation |
| `--eltwise` | eltwise | Yes | `input0:input1` | Binary element-wise ops |
| `--mvn` | mvn | Yes | `input` | Mean-variance normalization |
| `--rms` | rms | Yes | `input` | RMS normalization |
| `--group_normalization` | group_normalization | Yes | `input` | Group normalization |
| `--normalize` | normalize | Yes | `input` | L2 normalize |
| `--grn` | grn | Yes | `input` | Global response normalization |
| `--lrn` | lrn | Yes | `input` | Local response normalization |
| `--swiglu_standalone` | swiglu | Yes | `input` | SwiGLU activation |

### Pooling & Reduction

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--pooling` | pooling | Yes | `input` | Max/avg pooling |
| `--adaptive_pooling` | adaptive_pooling | — | `input` | Adaptive avg/max pooling |
| `--reduce` | reduce | Yes | `input` | Reduction (mean, sum, etc.) |
| `--arg_max_min` | arg_max_min | Yes | `input` | TopK / ArgMax / ArgMin |
| `--cum_sum` | cum_sum | Yes | `input` | Cumulative sum |

### Data Movement

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--reorder` | reorder | Yes | `input` | Layout/type conversion |
| `--permute` | permute | Yes | `input` | Dimension permutation |
| `--concatenation` | concatenation | Yes | `input0:input1:...` | Tensor concatenation |
| `--crop` | crop | Yes | `input:output` | Region crop (SliceOp) |
| `--strided_slice` | strided_slice | Yes | `input:output` | Strided slice |
| `--slice` | slice | Yes | `input` | Slice operation |
| `--broadcast` | broadcast | Yes | `input` | Shape broadcast |
| `--tile` | tile | Yes | `input` | Tile/repeat operation |
| `--select` | select | Yes | `output` | Conditional select |
| `--reverse` | reverse | Yes | `input` | Element reversal |
| `--reverse_sequence` | reverse_sequence | Yes | `input` | Sequence reversal |
| `--roll` | roll | Yes | `input` | Circular shift |
| `--depth_to_space` | depth_to_space | Yes | `input` | Depth-to-space rearrange |
| `--space_to_depth` | space_to_depth | Yes | `input` | Space-to-depth rearrange |
| `--batch_to_space` | batch_to_space | Yes | `input` | Batch-to-space rearrange |
| `--space_to_batch` | space_to_batch | Yes | `input` | Space-to-batch rearrange |
| `--border` | border | Yes | `input` | Padding (constant/edge/reflect) |
| `--shuffle_channels` | shuffle_channels | Yes | `input` | Channel shuffle |
| `--reorg_yolo` | reorg_yolo | Yes | `input` | YOLO reorg |

### Gather & Scatter

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--gather` | gather | Yes | `data:indices` | Gather along axis |
| `--gather_elements` | gather_elements | Yes | `data:indices` | Gather elements |
| `--gather_nd` | gather_nd | — | `data:indices` | N-D gather |
| `--gather_tree` | gather_tree | — | `input` | Beam search gather tree |
| `--scatter_update` | scatter_update | Yes | `data:indices:updates` | Scatter update |
| `--scatter_nd_update` | scatter_nd_update | Yes | `data:indices:updates` | N-D scatter update |
| `--scatter_elements_update` | scatter_elements_update | Yes | `data:indices:updates` | Scatter elements update |

### Quantization & Type

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--quantize` | quantize | Yes | `input:il:ih:ol:oh` | FakeQuantize |
| `--dynamic_quantize` | dynamic_quantize | — | `input` | Dynamic quantization |
| `--fake_convert` | fake_convert | Yes | `input` | Fake type conversion |
| `--resample` | resample | Yes** | `input` | Interpolation/resize |

\*\* Resample: nearest/bilinear modes have CPU ref; cubic mode reports `UNIMPLEMENTED`.

### Positional Encoding

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--rope` | rope | Yes | `input:cos:sin` | Rotary position embedding |
| `--embedding_bag` | embedding_bag | Yes | `input` | Embedding bag |

### Detection & NMS

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--detection_output` | detection_output | Yes | `input` | SSD detection output |
| `--non_max_suppression` | non_max_suppression | — | `input` | NMS |
| `--matrix_nms` | matrix_nms | — | `input` | Matrix NMS |
| `--multiclass_nms` | multiclass_nms | — | `input` | Multi-class NMS |
| `--proposal` | proposal | — | `input` | Proposal generation |
| `--generate_proposals` | generate_proposals | — | `input` | Generate proposals |
| `--prior_box` | prior_box | — | `input` | Prior box generation |
| `--region_yolo` | region_yolo | Yes | `input` | YOLO region layer |

### Experimental Detectron

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--exp_detectron_detection_output` | exp_detectron_detection_output | — | `input` | Detectron detection |
| `--exp_detectron_generate_proposals` | exp_detectron_gen_proposals | — | `input` | Detectron proposals |
| `--exp_detectron_prior_grid_gen` | exp_detectron_prior_grid | — | `input` | Detectron prior grid |
| `--exp_detectron_roi_feature` | exp_detectron_roi_feature | — | `input` | Detectron ROI feature |
| `--exp_detectron_topk_rois` | exp_detectron_topk_rois | — | `input` | Detectron TopK ROIs |

### Signal Processing

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--dft` | dft | — | `input` | Discrete Fourier transform |
| `--stft` | STFT | — | `input` | Short-time Fourier transform |
| `--istft` | ISTFT | — | `input` | Inverse STFT |

### Sequence / RNN

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--lstm_cell` | lstm_cell | Yes | `input` | LSTM cell |
| `--ctc_greedy_decoder` | ctc_greedy_decoder | — | `input` | CTC greedy decoder |
| `--ctc_loss` | ctc_loss | — | `input` | CTC loss |
| `--segment_max` | segment_max | Yes | `input` | Segment max reduction |

### Generators

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--eye` | eye | Yes | `input` | Identity matrix |
| `--range` | range | Yes | `input` | Range tensor |
| `--random_uniform` | random_uniform | Yes | `input` | Random uniform tensor |
| `--one_hot` | one_hot | Yes | `input` | One-hot encoding |

### Miscellaneous

| Flag | Kernel | Ref | Shapes | Description |
|------|--------|-----|--------|-------------|
| `--non_zero` | non_zero | Yes | `input` | Non-zero indices |
| `--bucketize` | bucketize | Yes | `input` | Bucketize |
| `--search_sorted` | search_sorted | Yes | `input` | Sorted search |
| `--unique_count` | unique_count | — | `input` | Unique element count |
| `--unique_gather` | unique_gather | — | `input` | Unique element gather |
| `--col2im` | col2im | Yes | `input` | Column-to-image |
| `--grid_sample` | grid_sample | Yes | `input` | Grid sample |
| `--roi_align` | roi_align | Yes | `input` | ROI align |
| `--roi_pooling` | roi_pooling | — | `input` | ROI pooling |
| `--extract_image_patches` | extract_image_patches | Yes | `input` | Image patch extraction |
| `--convert_color` | convert_color | Yes | `input` | Color space conversion |
| `--multinomial` | multinomial | — | `input` | Multinomial sampling |
| `--sparse_fill_empty_rows` | sparse_fill_empty_rows | — | `input` | Sparse fill empty rows |

---

## Kernel-Specific Options

### fully_connected (`--fc`)

Shapes: `BxM:NxK` (input `[B, K]` × weight `[N, K]`)

| Option | Default | Description |
|--------|---------|-------------|
| `--attr-scales` | — | Weight scale (e.g., `wei:per_ocic:f16:1x128`) |
| `--attr-zero-points` | — | Weight zero point |
| `--attr-post-ops` | — | Fused post-ops (e.g., `sum:f16+relu`) |

Supports compressed weights (`f16:i4:f16`), dynamic quantization, and
fused eltwise/activation post-ops.

### gemm (`--gemm`)

Shapes: `A:B[:C]` (A × B, optional addend C)

| Option | Default | Description |
|--------|---------|-------------|
| `--transpose_a` | `0` | Transpose first input |
| `--transpose_b` | `0` | Transpose second input |
| `--gemm_order0` | — | Input0 transpose order (e.g., `0:1:3:2`) |
| `--gemm_order1` | — | Input1 transpose order |
| `--gemm_order_out` | — | Output transpose order |
| `--attr-post-ops` | — | Fused post-ops |

### convolution (`--conv`)

Shapes: `input:filter` (e.g., `1x3x224x224:64x3x7x7`)

| Option | Default | Description |
|--------|---------|-------------|
| `--groups` | `1` | Number of groups |
| `--strides` | — | Stride (e.g., `2x2`) |
| `--dilations` | — | Dilation (e.g., `1x1`) |
| `--padding_begin` | — | Padding begin (e.g., `3x3`) |
| `--padding_end` | — | Padding end (e.g., `3x3`) |
| `--grouped_weights_shape` | `0` | Explicit group dim in weights |
| `--attr-post-ops` | — | Fused post-ops |

### scaled_dot_product_attention (`--sdpa`)

Shapes: `Q:K:V` (Query, Key, Value tensors)

| Option | Default | Description |
|--------|---------|-------------|
| `--is_causal` | `-1` | Causal mask (-1=auto, 0=false, 1=true) |
| `--order_q` | — | Q transpose order (e.g., `0:2:1:3`) |
| `--order_k` | — | K transpose order |
| `--order_v` | — | V transpose order |
| `--order_out` | — | Output transpose order |
| `--scale_val` | — | Explicit scale value |

### eltwise (`--eltwise`)

Shapes: `input0:input1`

| Option | Default | Description |
|--------|---------|-------------|
| `--eltwise_mode` | `-1` | Operation mode (-1=from post-ops, 0=add, 1=mul, ...) |
| `--pythondiv` | `0` | Python-style floor division |

### softmax (`--softmax`)

Shapes: `input`

| Option | Default | Description |
|--------|---------|-------------|
| `--axis` | `-1` | Axis for softmax (-1=last dim) |

### reduce (`--reduce`)

Shapes: `input`

| Option | Default | Description |
|--------|---------|-------------|
| `--reduce_mode` | `1` | Mode (1=mean, 0=sum, 2=max, 3=min, ...) |
| `--keep_dims` | `1` | Keep reduced dims |
| `--reduce_axes` | — | Reduction axes (e.g., `1:2:3`) |

### pooling (`--pooling`)

Shapes: `input`

| Option | Default | Description |
|--------|---------|-------------|
| `--pool_mode` | `0` | Mode (0=max, 1=avg, 2=avg_no_pad) |
| `--kernel` | — | Kernel size (e.g., `2x2`) |
| `--pool_strides` | — | Stride (e.g., `2x2`) |
| `--pads_begin` | — | Padding begin |
| `--pads_end` | — | Padding end |
| `--rounding_type` | `0` | Rounding (0=floor, 1=ceil) |

### mvn (`--mvn`)

Shapes: `input`

| Option | Default | Description |
|--------|---------|-------------|
| `--normalize_variance` | `1` | Normalize variance |
| `--epsilon` | — | Epsilon value |
| `--eps_inside_sqrt` | `0` | Epsilon inside sqrt |

### gather (`--gather`)

Shapes: `data:indices`

| Option | Default | Description |
|--------|---------|-------------|
| `--gather_axis` / `--axis` | `0` | Gather axis |
| `--batch_dim` | `0` | Batch dimension |
| `--support_neg_ind` | `0` | Support negative indices |

### rope (`--rope`)

Shapes: `input:cos:sin`

| Option | Default | Description |
|--------|---------|-------------|
| `--head_cnt` | `0` | Head count |
| `--head_size` | `0` | Head size |
| `--rotary_ndims` | `0` | Rotary dimensions |
| `--is_interleaved` | `0` | Interleaved mode |
| `--is_chatglm` | `0` | ChatGLM mode |
| `--is_qwen` | `0` | Qwen mode |
| `--input_trans0213` | `1` | Input transpose 0213 |
| `--slice_start` | `0` | Slice start |
| `--slice_stop` | `0` | Slice stop |
| `--gather_rank` | `0` | Gather rank |

### swiglu (`--swiglu_standalone`)

Shapes: `input`

| Option | Default | Description |
|--------|---------|-------------|
| `--glu_type` | `0` | GLU type (0=Swish) |
| `--split_axis` | `-1` | Split axis |
| `--split_length` | `-1` | Split length |
| `--gate_idx` | `0` | Gate index |

### resample (`--resample`)

Shapes: `input`

| Option | Default | Description |
|--------|---------|-------------|
| `--resample_sizes` | — | Output sizes (e.g., `1:64:64:64`) |
| `--resample_mode` | `0` | Mode (0=nearest, 1=linear, 2=cubic) |

### strided_slice (`--strided_slice`)

Shapes: `input:output`

| Option | Default | Description |
|--------|---------|-------------|
| `--ss_begin` | — | Begin indices |
| `--ss_end` | — | End indices |
| `--ss_strides` | — | Strides |
| `--begin_mask` | — | Begin mask |
| `--end_mask` | — | End mask |
| `--shrink_axis_mask` | — | Shrink axis mask |
| `--new_axis_mask` | — | New axis mask |

### Other Kernel Options

Additional kernel-specific options (shown with defaults):

| Kernel | Option | Default | Description |
|--------|--------|---------|-------------|
| `--concatenation` | `--concat_axis` | `1` | Concatenation axis |
| `--tile` | `--tile_repeats` | — | Repeats (e.g., `2:2:1:1`) |
| `--normalize` | `--across_spatial` | `0` | Across spatial |
| `--group_normalization` | `--num_groups` | `1` | Number of groups |
| `--quantize` | `--levels` | `0` | FakeQuantize levels |
| `--scatter_nd_update` | `--indices_rank` | `0` | Indices rank |
| `--permute` | `--permute_order` | — | Permute order (e.g., `0:2:1:3`) |
| `--broadcast` | `--broadcast_axes` | — | Broadcast axes |
| `--broadcast` | `--broadcast_target` | — | Target shape |
| `--adaptive_pooling` | `--adaptive_pool_mode` | `0` | Mode (0=avg, 1=max) |
| `--arg_max_min` | `--topk_mode` | `0` | Mode (0=max, 1=min) |
| `--arg_max_min` | `--top_k` | `1` | K value |
| `--col2im` | `--col2im_output_shape` | — | Output shape (e.g., `4x4`) |
| `--col2im` | `--col2im_kernel_shape` | — | Kernel shape (e.g., `3x3`) |
| `--depth_to_space` | `--block_size` | `0` | Block size |
| `--depth_to_space` | `--d2s_mode` | `0` | Mode (0=blocks_first) |
| `--cum_sum` | `--cum_exclusive` | `0` | Exclusive mode |
| `--cum_sum` | `--cum_reverse` | `0` | Reverse mode |
| `--lrn` | `--lrn_size` | `5` | Normalization size |
| `--lrn` | `--lrn_alpha` | `0.0001` | Alpha |
| `--lrn` | `--lrn_beta` | `0.75` | Beta |
| `--grn` | `--grn_bias` | `1e-6` | Bias |
| `--shuffle_channels` | `--shuffle_group` | `2` | Group |
| `--reorder` | `--truncate` | `false` | Truncation mode |
| `--border` | `--border_mode` | `0` | Pad mode (0=const, 1=edge, 2=reflect) |
| `--one_hot` | `--one_hot_depth` | `0` | Depth |
| `--grid_sample` | `--grid_mode` | `0` | Mode (0=bilinear, 1=bicubic) |
| `--grid_sample` | `--grid_align_corners` | `0` | Align corners |
| `--roi_align` | `--roi_pooled_h` | `7` | Pooled height |
| `--roi_align` | `--roi_pooled_w` | `7` | Pooled width |
| `--roi_align` | `--roi_spatial_scale` | `0.0625` | Spatial scale |
| `--detection_output` | `--det_num_classes` | `21` | Number of classes |
| `--detection_output` | `--det_keep_top_k` | `200` | Keep top K |
| `--detection_output` | `--det_nms_threshold` | `0.45` | NMS threshold |
| `--range` | `--range_start` | `0.0` | Start |
| `--range` | `--range_stop` | `100.0` | Stop |
| `--range` | `--range_step` | `1.0` | Step |
| `--eye` | `--eye_diagonal` | `0` | Diagonal offset |
| `--dft` | `--dft_inverse` | `0` | Inverse mode |

---

## Output Format

Each test produces a result line:

```
impl_info: fc_bf_tiled__f16
0:PASSED (1.23 ms) __REPRO: --fc --dt=f16 --shapes=1x4096:4096x4096 --mode=c --device=0
```

### Test Status

| Status | Meaning |
|--------|---------|
| `PASSED` | GPU output matched CPU reference (or perf-only completed successfully) |
| `FAILED` | Accuracy mismatch or runtime error |
| `UNIMPLEMENTED` | Requested primitive itself is not implemented in `bench_kernel`, or correctness checking or a specific bench/runtime path is not implemented for this kernel/config |
| `SKIPPED` | Test intentionally skipped (e.g., reference too slow for large problem) |

### Auto-Skip for Large Convolutions

In accuracy mode (`--mode=c`), convolutions with estimated reference FLOPs
exceeding a threshold are automatically skipped to prevent long-running CPU
reference computations. The threshold is set as a constant in
`bench_conv.cpp` (`CONV_REF_OPS_SKIP_THRESHOLD = 100,000,000`).

The estimated reference ops are calculated as:
`input_volume × kernel_spatial_volume`.

Skipped tests report:
```
impl_info: skipped
0:SKIPPED (0.00 ms) Conv reference too slow (6936330240 ref ops) __REPRO: ...
```

### Summary Line

```
tests:100 passed:95 unimplemented:3 skipped:1 failed:1
total perf: min(ms):0.00625 avg(ms):0.00812
total acc: elements=4096 mismatches=0 max_abs=0.00e+00 max_rel=0.00e+00
```

The `__REPRO:` string in each result is a copy-pasteable command to
reproduce the exact test case.

---

## Batch File Format

Batch files list one problem per line with full CLI flags:

```
# LLM FC shapes (Llama-7B)
--fc --dt=f16 --shapes=1x4096:4096x4096 --mode=c
--fc --dt=f16 --shapes=1x4096:11008x4096 --mode=c --attr-post-ops=swish:1
--fc --dt=f16:i4:f16 --shapes=1x4096:4096x4096 --attr-scales=wei:per_ocic:f16:1x128 --mode=c
```

## Verbose Log Converter

The `bench_kernel_converter.py` script converts `OV_GPU_BenchVerbose=1`
output from `benchmark_app` into batch files:

```sh
# Capture verbose log
OV_GPU_BenchVerbose=1 benchmark_app -m model.xml -d GPU -niter 1 2>&1 | \
    grep ov_gpu_bench > verbose.log

# Convert to batch file
python3 scripts/bench_kernel_converter.py -i verbose.log -o batch.txt --mode=c

# Filter specific kernel
python3 scripts/bench_kernel_converter.py -i verbose.log -o fc_batch.txt -k fc --uniq

# Show summary, including entries marked unsupported=1 by debug_helper
python3 scripts/bench_kernel_converter.py -i verbose.log -o batch.txt --summary
```

When `debug_helper` emits a line with `unsupported=1`, the converter treats it as a non-reproducible bench entry and
skips command generation for that line.

---

## Directory Structure

```
bench_kernel/
├── README.md                  # This file
├── CMakeLists.txt             # Build configuration
├── bench_kernel_main.cpp      # CLI entry point and harness
├── batch_fc_llm.txt           # Example batch file
├── common/
│   ├── bench_config.hpp       # Configuration and argument parsing
│   ├── bench_timer.hpp        # Performance timer, test status, statistics
│   ├── bench_reference.hpp    # CPU reference implementations (~50 functions)
│   └── bench_utils.hpp        # Utilities (memory, comparison, tolerance)
├── primitives/
│   ├── kernel_base.hpp        # Base class for all kernel drivers
│   ├── bench_fc.cpp           # FC driver
│   ├── bench_gemm.cpp         # Gemm driver
│   ├── bench_conv.cpp         # Conv driver
│   ├── bench_sdpa.cpp         # SDPA driver
│   ├── bench_eltwise.cpp      # Eltwise driver
│   ├── bench_softmax.cpp      # Softmax driver
│   ├── ...                    # (87 kernel files total)
│   └── bench_misc.cpp         # Misc kernels (reorder, concatenation, activation)
└── scripts/
    └── bench_kernel_converter.py  # Verbose log → batch file converter
```
