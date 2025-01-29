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
* [OpenVINO™ Runtime release package](https://docs.openvino.ai/2025/get-started/install-openvino.html)
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
    > Note 2: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, specifying OpenVINO_DIR and calling setupvars might not be needed. Refer [documentation](https://docs.openvino.ai/2025/get-started/install-openvino.html) for details.
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
    > Note 3: this example is based on OpenVINO Archive distribution. If you have chosen another installation method, calling setupvars might not be needed. Refer [documentation](https://docs.openvino.ai/2025/get-started/install-openvino.html) for details.

    Successful build will show the information about Single Image Test Tool CLI options


## How to run

Running the application with the `-help` option yields the following usage message:
```
single-image-test.exe: Usage: Release\single-image-test.exe[<options>]

  Flags from C:\Users\mdoronin\work\applications.ai.vpu-accelerators.vpux-plugin\tools\single-image-test\main.cpp:
    -box_tolerance (Box tolerance for 'detection' mode) type: double
      default: 0.0001
    -classes (Number of classes for Yolo V3) type: int32 default: 80
    -color_format (Color format for input: RGB or BGR) type: string
      default: "BGR"
    -compiled_blob (Output compiled network file (compiled result blob))
      type: string default: ""
    -confidence_threshold (Confidence threshold for Detection mode)
      type: double default: 0.0001
    -config (Path to the configuration file (optional)) type: string
      default: ""
    -coords (Number of coordinates for Yolo V3) type: int32 default: 4
    -cosim_threshold (Threshold for 'cosim' mode) type: double
      default: 0.90000000000000002
    -dataset (The dataset used to train the model. Useful for instances such as
      semantic segmentation to visualize the accuracy per-class) type: string
      default: "NONE"
    -device (Device to use) type: string default: ""
    -il (Input layout) type: string default: ""
    -img_as_bin (Force binary input even if network expects an image)
      type: bool default: false
    -img_bin_precision (Specify the precision of the binary input files. Eg:
      'FP32,FP16,I32,I64,U8') type: string default: ""
    -iml (Model input layout) type: string default: ""
    -input (Input file(s)) type: string default: ""
    -ip (Input precision (default: U8, available: FP32, FP16, I32, I64, U8))
      type: string default: ""
    -is_tiny_yolo (Is it Tiny Yolo or not (true or false)?) type: bool
      default: false
    -log_level (IE logger level (optional)) type: string default: ""
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
    -nrmse_loss_threshold (Threshold for 'nrmse' mode) type: double default: 1
    -num (Number of scales for Yolo V3) type: int32 default: 3
    -ol (Output layout) type: string default: ""
    -oml (Model output layout) type: string default: ""
    -op (Output precision (default: FP32, available: FP32, FP16, I32, I64, U8))
      type: string default: ""
    -override_model_batch_size (Enforce a model to be compiled for batch size)
      type: uint32 default: 1
    -pc (Report performance counters) type: bool default: false
    -prob_tolerance (Probability tolerance for 'classification/ssd/yolo' mode)
      type: double default: 0.0001
    -psnr_reference (PSNR reference value in dB) type: double default: 30
    -psnr_tolerance (Tolerance for 'psnr' mode) type: double default: 0.0001
    -raw_tolerance (Tolerance for 'raw' mode (absolute diff)) type: double
      default: 0.0001
    -rrmse_loss_threshold (Threshold for 'rrmse' mode) type: double
      default: 1.7976931348623157e+308
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
    -sem_seg_threshold (Threshold for 'semantic segmentation' mode)
      type: double default: 0.97999999999999998
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
