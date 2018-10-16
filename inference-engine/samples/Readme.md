Inference Engine Samples {#SamplesOverview}
================

The Inference Engine sample applications are simple console applications that demonstrate how you can use the Intel's Deep Learning Inference Engine in your applications.

The Deep Learning Inference Engine release package provides the following sample applications available in the samples
directory in the Inference Engine installation directory:

 - [CPU Extensions](@ref CPUExtensions) library with topology-specific layers (like DetectionOutput used in the SSD*, below)
 - [Hello Autoresize Classification Sample](@ref InferenceEngineHelloAutoresizeClassificationSample) - Input of any size and layout can be set to an infer request which will be pre-processed automatically during inference (the sample supports only images as inputs)
 - [Hello Infer Request Classification Sample](@ref InferenceEngineHelloRequestClassificationSample) - Inference of image classification networks via Infer Request API (the sample supports only images as inputs)
 - [Image Classification Sample](@ref InferenceEngineClassificationSampleApplication) - Inference of image classification networks like AlexNet and GoogLeNet (the sample supports only images as inputs)
 - [Image Classification Sample, pipelined](@ref InferenceEngineClassificationPipelinedSampleApplication)- Maximize performance via pipelined execution, the sample supports only images as inputs
 - [Neural Style Transfer Sample](@ref InferenceEngineNeuralStyleTransferSampleApplication) - Style Transfer sample (the sample supports only images as inputs)
 - [Object Detection for SSD Sample](@ref InferenceEngineObjectDetectionSSDSampleApplication) - Inference of object detection networks based on the SSD, this sample is simplified version that supports only images as inputs
 - [Validation App](@ref InferenceEngineValidationApp) - Infers pack of images resulting in total accuracy (only images as inputs)

## <a name="build_samples_linux"></a> Building the Sample Applications on Linux*
The officially supported Linux build environment is the following:

* Ubuntu* 16.04 LTS 64-bit or CentOS* 7.4 64-bit
* GCC* 5.4.0 (for Ubuntu* 16.04) or GCC* 4.8.5 (for CentOS* 7.4)
* CMake* version 2.8 or higher.
* OpenCV 3.3 or later (required for some samples)

<br>You can build the sample applications using the <i>CMake</i> file in the `samples` directory.

Create a new directory and change your current directory to the new one:
```sh
mkdir build
cd build
```
Run <i>CMake</i> to generate Make files:
```sh
cmake -DCMAKE_BUILD_TYPE=Release <path_to_inference_engine_samples_directory>
```

To build samples with debug information, use the following command:
```sh
cmake -DCMAKE_BUILD_TYPE=Debug <path_to_inference_engine_samples_directory>
```

Run <i>Make</i> to build the application:
```sh
make
```

For ease of reference, the Inference Engine installation folder is referred to as <code><INSTALL_DIR></code>.

After that you can find binaries for all samples applications in the <code>intel64/Release</code> subfolder.

## <a name="build_samples_windows"></a> Building the Sample Applications on Microsoft Windows* OS

The recommended Windows build environment is the following:
* Microsoft Windows* 10
* Microsoft* Visual Studio* 2015 including Microsoft Visual Studio 2015 Community or Microsoft Visual Studio 2017
* CMake* version 2.8 or later
* OpenCV* 3.3 or later


Generate Microsoft Visual Studio solution file using <code>create_msvc_solution.bat</code> file in the <code>samples</code> directory and then build the solution <code>samples\build\Samples.sln</code> in the Microsoft Visual Studio 2015.

## Running the Sample Applications

Before running compiled binary files, make sure your application can find the Inference Engine libraries.
Use the `setvars.sh` script, which will set all necessary environment variables.

For that, run (assuming that you are in a <code><INSTALL_DIR>/deployment_tools/inference_engine/bin/intel64/Release</code> folder):
<pre>
source ../../setvars.sh
</pre>

What is left is running the required sample with appropriate commands, providing IR information (typically with "-m" command-line option).
Please note that Inference Engine assumes that weights are in the same folder as _.xml_ file.

## See Also
* [Introduction to Intel's Deep Learning Inference Engine](@ref Intro)

---
\* Other names and brands may be claimed as the property of others.
