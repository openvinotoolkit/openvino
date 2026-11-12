# NPU Single Image Test Tool

This page demostrates how to use NPU Single Image Test Tool for end-to-end accuracy validation on a single image or input file with OpenVINO™ Intermediate Representation (IR) of an AI model or a model in ONNX format.


## Description

Single Image Test Tool is a C++ application that enables you to pass OpenVINO IR or ONNX model or pre-compiled blob and a single image or any other compatible file with the model inputs and get 2 sets of files with CPU outputs and NPU outputs that can be compared later or straight after the inference if `-run_test` option is passed.

The tool can be configured to perform various preprocessing methods and output comparison algorithms depending on the model topology and its output semantics. See the tool help message below for the details.

Using Single Image Test is not a basic approach to end-to-end validation or collecting release measures but is created for development CI checks and debugging. However, some methodologies might be useful if you're looking for examples of how to use NPU and preprocess data for the inference on NPU. If you're looking for the standard way of measuring accuracy please refer to [Deep Learning accuracy validation framework](https://github.com/openvinotoolkit/open_model_zoo/tree/master/tools/accuracy_checker).


## How to build

### Within NPU Plugin build

See [How to build](https://github.com/openvinotoolkit/openvino/wiki#how-to-build). If `ENABLE_INTEL_NPU=ON` is provided and `OpenCV` project is linked to the current cmake project, no additional steps are required for Single Image Test. It will be built unconditionally with every NPU Plugin build. It can be found in `bin` folder.

If you need to configure a release package layout and have Single Image Test in it, use `cmake --install <dir> --component npu_internal` from your `build` folder. After installation single-image-test executable can be found in `<install_dir>/tools/single-image-test` folder.

### Standalone build

#### Prerequisites
* [OpenVINO™ Runtime release package](https://docs.openvino.ai/2026/get-started/install-openvino.html)
* [OpenCV: Open Source Computer Vision Library release package](https://opencv.org/get-started/)

#### Build instructions
1. Download and install OpenVINO™ Runtime package
1. Download and install OpenCV package
1. Build Single Image Test Tool
    ```sh
    mkdir sit_build && cd sit_build
    source <openvino_install_dir>/setupvars.sh
    cmake -DOpenVINO_DIR=<openvino_install_dir>/runtime/cmake -DOpenCV_DIR=<opencv_install_dir> <sit_source_dir>
    cmake --build . --config Release
    cmake --install . --prefix <sit_install_dir>
    ```
    > Note 1: command line instruction might differ on different platforms (e.g. Windows cmd)
    > Note 2: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, specifying OpenVINO_DIR and calling setupvars might not be needed. Refer [documentation](https://docs.openvino.ai/2026/get-started/install-openvino.html) for details.
    > Note 3: depending on OpenCV installation method, there might not be a need to specify OpenCV_DIR.
    > Note 4: depending on OpenCV version, cmake configs might be located somewhere else. You need to specify a directory that contains `OpenCVConfig.cmake` file
    > Note 5: `<sit_install_dir>` can be any directory on your filesystem that you want to use for installation including `<openvino_install_dir>` if you wish to extend OpenVINO package
1. Verify the installation
    ```sh
    source <openvino_install_dir>/setupvars.sh
    source <opencv_install_dir>setup_vars_opencv4.sh
    <sit_install_dir>/tools/single-image-test/single-image-test -help
    ```
    > Note 1: command line might differ depending on your platform
    > Note 2: depending on OpenCV installation method, there might not be a need to call setupvars.
    > Note 3: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, calling setupvars might not be needed. Refer [documentation](https://docs.openvino.ai/2026/get-started/install-openvino.html) for details.

    Successful build will show the information about Single Image Test Tool CLI options


## PixelLink / PSO0 post-processing in mAP mode

`--mode map` recognizes PSO0-style named, dual-direction PixelLink heads.
Each spatial scale must contain all six NCHW tensors:

| Name tokens (case-insensitive) | Shape | Meaning |
| --- | --- | --- |
| `scores_hori`, `scores_vert` | `[1,1,H,W]` | Horizontal/vertical text probabilities |
| `bbox_deltas_hori`, `bbox_deltas_vert` | `[1,8,H,W]` | Four `(dx,dy)` corner offsets in grid-cell units |
| `link_scores_hori`, `link_scores_vert` | `[1,8,H,W]` | Horizontal/vertical link probabilities |

Tensors are grouped by spatial dimensions; suffixes such as `fpn2` are not
required. Head role and orientation names are required: shapes alone cannot
distinguish eight-channel links from corner offsets. Missing, duplicate,
incorrectly shaped, or non-finite heads produce an error.

The decoder preserves the Python proof-of-concept algorithm: select text
pixels using the maximum horizontal/vertical score rounded to three decimal
places (ties to even), require the selected orientation's maximum link
probability to pass its threshold, form eight-connected components, fit a
minimum-area rectangle to each component's corner votes, and apply polygon
NMS across scales.

| Option | Default | Scope |
| --- | --- | --- |
| `--pixellink_text_threshold` | `0.55` | Global text-pixel cutoff in `[0,1]` |
| `--pixellink_link_threshold` | `0.60` | Global link-probability cutoff in `[0,1]` |
| `--pixellink_nms_iou` | `0.50` | Global polygon NMS IoU threshold in `[0,1]` |
| `--pixellink_image_size` | `768,1152` | Positive integer image dimensions `H,W` |
| `--pixellink_morph_close` | `false` | Optional 3x3 morphological closing before component labeling; may merge adjacent text |

Thresholds must be finite scalar values, not per-layer maps. Image dimensions
determine the scale-to-image ratios; set them explicitly for other input sizes.
These options are ignored by non-PixelLink detection formats.

Generate CPU reference blobs in a fresh working directory, then compare the
device outputs using the same model, input, and layouts:

```sh
single-image-test --device CPU --network /path/model.onnx \
    --input /path/input.bin --img_as_bin --ip FP32 --op FP32 \
    --il HWC --iml HWC --ol NCHW --oml NCHW

single-image-test --device NPU --network /path/model.onnx \
    --input /path/input.bin --img_as_bin --ip FP32 --op FP32 \
    --il HWC --iml HWC --ol NCHW --oml NCHW \
    --run_test --mode map --map_threshold 0.5 --overlap_threshold 0.5 \
    --pixellink_text_threshold 0.55 --pixellink_link_threshold 0.60 \
    --pixellink_nms_iou 0.5 --pixellink_image_size 768,1152
```

Use the device/configuration appropriate for the installed plugin. Both commands
must complete successfully; do not reuse blobs from a failed or unrelated run.
This example's HWC input is PSO0-specific; the output heads must be NCHW.

### Metric limitations

- Oriented polygons are reduced to **axis-aligned bounding boxes**, all assigned
  **class 0**. Matching uses the existing SIT mAP implementation and
  `--overlap_threshold`, not oriented-polygon IoU or COCO multi-IoU mAP.
- The decoder uses links as a per-pixel mask, not a directional neighbor graph.
  Adjacent text instances can merge. This is PoC-compatible behavior, not a
  generic implementation of every PixelLink model's postprocessor.
- A comparison against CPU-decoded tensors measures **reference agreement**,
  not accuracy against annotated ground truth. Matching errors on both sides
  can still yield high mAP.
- An empty prediction or reference set fails the existing mAP gate, with an
  explicit message; it is not misclassified as an unsupported output format.
- Decoded mAP does not replace raw-tensor NRMSE or change existing validation
  configurations. Run each metric separately when both are needed.

### CPU regression tests

The synthetic tests need a built SIT plus the OpenVINO Python package, NumPy,
and a CPU plugin. No private model or NPU is required:

```sh
SIT_BINARY=/path/to/single-image-test python3 test_pixellink.py -v
```

Set `SIT_TEST_OUTPUT_DIR` to a new directory to retain models, references,
commands, and logs; otherwise temporary test artifacts are cleaned up.

## How to run

Running the application with the `-help` option yields the following usage message:
```
single-image-test.exe: Usage: Release\single-image-test.exe[<options>]

  Flags from C:\Users\mdoronin\work\applications.ai.vpu-accelerators.vpux-plugin\tools\single-image-test\main.cpp:
    -apply_soft_max (Apply SoftMax for 'nrmse' mode) type: bool default: false
    -box_tolerance (Box tolerance for 'detection' mode. Can be a single value
      or per-layer: 'layer1:0.01;layer2:0.02') type: string default: "0.000100"
    -classes (Number of classes for Yolo V3) type: int32 default: 80
    -clamp_inf_outputs (Optional. ';'-separated list of output tensor names for
      which +/-inf values will be clamped to the representable
      [dtype::lowest, dtype::max] range before metric evaluation. Supported
      element types: f32, f16, bf16. Unsupported types are skipped with a
      warning.) type: string default: ""
    -clamp_u8_outputs (Apply clamping when converting FP to U8) type: bool
      default: false
    -color_format (Color format for input: RGB or BGR) type: string
      default: "BGR"
    -compiled_blob (Output compiled network file (compiled result blob))
      type: string default: ""
    -blob_name_prefix (Optional prefix overriding model-derived blob file
      naming for auto-generated input/reference/output blobs. Explicit
      --ref_results paths are not modified.) type: string default: ""
    -confidence_threshold (Confidence threshold for Detection mode. Can be a
      single value or per-layer: 'layer1:0.5;layer2:0.3') type: string
      default: "0.000100"
    -config (Path to the configuration file (optional)) type: string
      default: ""
    -coords (Number of coordinates for Yolo V3) type: int32 default: 4
    -cosim_threshold (Threshold for 'cosim' mode. Can be a single value or
      per-layer: 'layer1:0.95;layer2:0.90') type: string default: "0.900000"
    -data_shape (Required for models with dynamic shapes. Set shape for input
      blobs. Only one shape can be set per input. In case of one input size:
      "[1,3,224,224]", applied to every dynamic input. For several inputs:
      "input1[1,3,224,224],input2[1,4]") type: string default: ""
    -dataset (The dataset used to train the model. Useful for instances such as
      semantic segmentation to visualize the accuracy per-class) type: string
      default: "NONE"
    -device (Device to use) type: string default: ""
    -il (Input layout for all inputs, or ';' separated list of pairs
      <input>:<layout>. Regex in <input> is supported) type: string default: ""
    -img_as_bin (Force binary input even if network expects an image)
      type: bool default: false
    -img_bin_precision (Specify the precision of the binary input files. Eg:
      'FP32,FP16,I32,I64,U8') type: string default: ""
    -iml (Model input layout for all model inputs, or ';' separated list of
      pairs <input>:<layout>. Regex in <input> is supported)
      type: string default: ""
    -input (Input file(s)) type: string default: ""
    -ip (Input precision (default: U8, available: FP32, FP16, I32, I64, U8,
      U16, I16, U4, I4, U2, BF8(F8E5M2), HF8(F8E4M3))) type: string
      default: ""
    -is_tiny_yolo (Is it Tiny Yolo or not (true or false)?) type: bool
      default: false
    -l2norm_dequant_scale (Optional. Dequantization scale factor applied to
      the inference output tensor(s) before the 'l2norm' comparison is
      performed: dequantized = (quantized - zero_point) * scale. Can be a
      single global value applied to all outputs, or per-output:
      'output1:0.00392;output2:0.01'. If not set, dequantization is skipped
      and outputs are compared as-is. The reference tensor is never
      modified.) type: string default: ""
    -l2norm_dequant_zp (Optional. Dequantization zero point used together
      with --l2norm_dequant_scale for 'l2norm' mode. Can be a single global
      value or per-output: 'output1:128;output2:0'. Defaults to 0 for any
      output without an explicit value. Has no effect unless
      --l2norm_dequant_scale is set.) type: string default: ""
    -l2norm_threshold (Threshold for 'l2norm' mode. Can be a single value or
      per-layer: 'layer1:1.0;layer2:2.0') type: string default: "1.000000"
    -log_level (IE logger level (optional)) type: string default: ""
    -map_threshold (mAP score threshold for 'map' mode validation. Can be a
      single value or per-layer: 'layer1:0.5;layer2:0.6') type: string
      default: "0.500000"
    -mean_values (Optional. Mean values to be used for the input image per
      channel. Values to be provided in the [channel1,channel2,channel3]
      format. Can be defined for desired input of the model, for example:
      "--mean_values data[255,255,255],info[255,255,255]". The exact meaning
      and order of channels depend on how the original model was trained.
      Applying the values affects performance and may cause type conversion)
      type: string default: ""
    -mode (Comparison mode to use) type: string default: ""
    -network (Network file (either XML or pre-compiled blob)) type: string
      default: ""
    -normalized_image (Images in [0, 1] range or not) type: bool default: false
    -nrmse_loss_threshold (Threshold for 'nrmse' mode. Can be a single value
      or per-layer: 'logits:0.03;pred_boxes:0.05') type: string
      default: "0.020000"
    -num (Number of scales for Yolo V3) type: int32 default: 3
    -ol (Output layout for all outputs, or ';' separated list of pairs
      <output>:<layout>. Regex in <output> is supported) type: string
      default: ""
    -oml (Model output layout for all outputs, or ';' separated list of pairs
      <output>:<layout>. Regex in <output> is supported) type: string
      default: ""
    -op (Output precision (default: FP32, available: FP32, FP16, I32, I64, U8,
      U16, I16, U4, I4, U2, BF8(F8E5M2), HF8(F8E4M3))) type: string
      default: ""
    -overlap_threshold (IoU threshold for 'map' mode (detection matching). Can
      be a single value or per-layer: 'layer1:0.5;layer2:0.6') type: string
      default: "0.500000"
    -override_model_batch_size (Enforce a model to be compiled for batch size)
      type: uint32 default: 1
    -pc (Report performance counters) type: bool default: false
    -prob_tolerance (Probability tolerance for 'classification/ssd/yolo' mode.
      Can be a single value or per-layer: 'layer1:0.01;layer2:0.02')
      type: string default: "0.000100"
    -psnr_reference (PSNR reference value in dB. Can be a single value or
      per-layer: 'layer1:30.0;layer2:35.0') type: string default: "30.000000"
    -psnr_tolerance (Tolerance for 'psnr' mode. Can be a single value or
      per-layer: 'layer1:0.01;layer2:0.02') type: string default: "0.000100"
    -raw_tolerance (Tolerance for 'raw' mode (absolute diff). Can be a single
      value or per-layer: 'layer1:0.01;layer2:0.02') type: string
      default: "0.000100"
    -ref_dir (A directory with reference blobs to compare with in run_test
      mode. Leave it empty to use the current folder.) type: string default: ""
    -ref_results (String of reference result file(s) to be used during
      run_test mode. For the same test case, files are separated by comma (,);
      for different test cases by semicolon (;). If ref_dir is provided, paths
      should be relative to ref_dir; otherwise absolute paths are expected.)
      type: string default: ""
    -rrmse_loss_threshold (Threshold for 'rrmse' mode. Can be a single value
      or per-layer: 'layer1:0.1;layer2:0.2') type: string
      default: "1.7976931348623157e+308"
    -run_test (Run the test (compare current results with previously dumped))
      type: bool default: false
    -scale_border (Scale border) type: uint32 default: 4
    -scale_values (Optional. Scale values to be used for the input image per
      channel. Values are provided in the [channel1,channel2,channel3] format.
      Can be defined for desired input of the model, for example:
      "--scale_values data[255,255,255],info[255,255,255]". The exact meaning
      and order of channels depend on how the original model was trained. If
      both --mean_values and --scale_values are specified, the mean is
      subtracted first and then scale is applied regardless of the order of
      options in command line. Applying the values affects performance and may
      cause type conversion) type: string default: ""
    -sem_seg_classes (Number of classes for semantic segmentation) type: uint32
      default: 12
    -sem_seg_ignore_label (The number of the label to be ignored) type: uint32
      default: 4294967295
    -sem_seg_threshold (Threshold for 'semantic segmentation' mode. Can be a
      single value or per-layer: 'layer1:0.98;layer2:0.95') type: string
      default: "0.98"
    -shape (Optional. Set shape for model input. For example,
      "input1[1,3,224,224],input2[1,4]" or "[1,3,224,224]" in case of one
      input size. This parameter affects model input shape and can be dynamic.
      For dynamic dimensions use symbol '?' or '-1'. Ex. [?,3,?,?].
      For bounded dimensions specify range 'min..max'. Ex. [1..10,3,?,?].)
      type: string default: ""
    -skip_arg_max (Skip ArgMax post processing step) type: bool default: false
    -skip_output_layers (Skip output layers from the network. Accept ';'
      separated list of output layers) type: string default: ""
    -top_k (Top K parameter for 'classification' mode) type: uint32 default: 1
```

For example, to run inference with mobilenet-v2 model on Intel® Core™ Ultra NPU on Windows 11 OS, run the commands below:

1. Running inference on CPU to collect reference result
    ```
    single-image-test.exe \
        --network \
        mobilenet-v2.xml \
        --input \
        validation-set/224x224/watch.bmp \
        --ip \
        FP16 \
        --op \
        FP16 \
        --device \
        CPU \
        --color_format \
        RGB \
        --il \
        NCHW \
        --ol \
        NC \
        --iml \
        NCHW \
        --oml \
        NC
    ```
    expected output:
    ```
    Parameters:
        Network file:                             mobilenet-v2.xml
        Input file(s):                            validation-set/224x224/watch.bmp
        Output compiled network file:
        Blob name prefix override:
        Color format:                             RGB
        Input precision:                          FP16
        Output precision:                         FP16
        Input layout:                             NCHW
        Output layout:                            NC
        Model input layout:                       NCHW
        Model output layout:                      NC
        Img as binary:                            0
        Bin input file precision:
        Device:                                   CPU
        Config file:
        Run test:                                 0
        Performance counters:                     0
        Mean_values [channel1,channel2,channel3]
        Scale_values [channel1,channel2,channel3]
        Skip checking output layers:
        Clamp U8 outputs:                         0
        Clamp inf outputs:
        Log level:

    Run single image test
    Load network mobilenet-v2.xml
    Load input #0 from validation-set/224x224/watch.bmp as f16 [N,C,H,W] [1,3,224,224]
    Dump input #0_case_0 to _mobilenet_v2_input_0_case_0.blob
    Run inference on CPU
    Latency: 100 ms
    Dump reference output #0 to _mobilenet_v2_ref_out_0_case_0.blob
    ```

1. Running inference on NPU and comparing results. In this example, it's considered that the model has been compiled before and exported as a blob file. You can pass OpenVINO IR with the same success but depending on a model and your setup, some additional configs might be needed in a config file or in CLI.

    ```
    single-image-test.exe \
        --network \
        mobilenet-v2.blob \
        --input \
        validation-set/224x224/watch.bmp \
        --ip \
        FP16 \
        --op \
        FP16 \
        --device \
        NPU \
        --config \
        mobilenet-v2.conf \
        --run_test \
        -log_level \
        LOG_ERROR \
        --mode \
        classification \
        --top_k \
        1 \
        --prob_tolerance \
        0.6 \
        --color_format \
        RGB \
        --il \
        NCHW \
        --ol \
        NC \
        --iml \
        NCHW \
        --oml \
        NC
    ```

    the content of mobilenet-v2.conf:
    ```
    NPU_COMPILER_TYPE DRIVER
    NPU_PLATFORM VPU3720
    ```

    expected output:
    ```
    Parameters:
        Network file:                             mobilenet-v2.blob
        Input file(s):                            validation-set/224x224/watch.bmp
        Output compiled network file:
        Blob name prefix override:
        Color format:                             RGB
        Input precision:                          FP16
        Output precision:                         FP16
        Input layout:                             NCHW
        Output layout:                            NC
        Model input layout:                       NCHW
        Model output layout:                      NC
        Img as binary:                            0
        Bin input file precision:
        Device:                                   NPU
        Config file:                              mobilenet-v2.conf
        Run test:                                 1
        Performance counters:                     0
        Mean_values [channel1,channel2,channel3]
        Scale_values [channel1,channel2,channel3]
        Skip checking output layers:
        Clamp U8 outputs:                         0
        Clamp inf outputs:
        Reference files directory:                Current directory
        Reference file(s):
        Mode:             classification
        Top K:            1
        Tolerance:        0.6
        Log level:                        LOG_ERROR

    Run single image test
    Import network mobilenet-v2.blob
    Load input #0 from validation-set/224x224/watch.bmp as f16 [N,C,H,W] [1,3,224,224]
    Dump input #0_case_0 to _mobilenet_v2_input_0_case_0.blob
    Run inference on NPU
    Latency: 3 ms
    Load reference output #0 from _mobilenet_v2_ref_out_0_case_0.blob as f16
    Dump device output #0_case_0 to _mobilenet_v2_kmb_out_0_case_0.blob
    Actual top:
        0 : 531 : 21.95
    Ref Top:
        0 : 531 : 21.95
    PASSED
    ```
