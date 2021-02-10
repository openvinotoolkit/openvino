# Custom Operations Guide {#openvino_docs_HOWTO_Custom_Layers_Guide}

The Intel® Distribution of OpenVINO™ toolkit supports neural network models trained with multiple frameworks including
TensorFlow*, Caffe*, MXNet*, Kaldi* and ONNX* file format. The list of supported operations (layers) is different for
each of the supported frameworks. To see the operations supported by your framework, refer to
[Supported Framework Layers](../MO_DG/prepare_model/Supported_Frameworks_Layers.md).

Custom operations are operations that are not included in the list of known operations. If your model contains any
operation that is not in the list of known operations, the Model Optimizer is not able to generate an Intermediate
Representation (IR) for this model.

This guide illustrates the workflow for running inference on topologies featuring custom operations, allowing you to
plug in your own implementation for existing or completely new operation.

> **NOTE:** *Layer* — The legacy term for an *operation* which came from Caffe\* framework. Currently it is not used.
> Refer to the [Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™](../MO_DG/IR_and_opsets.md)
> for more information on the topic.

## Terms Used in This Guide

- *Intermediate Representation (IR)* — Neural Network used only by the Inference Engine in OpenVINO abstracting the
  different frameworks and describing the model topology, operations parameters and weights.

- *Operation* — The abstract concept of a math function that is selected for a specific purpose. Operations supported by
  OpenVINO™ are listed in the supported operation set provided in the [Available Operations Sets](../ops/opset.md).
  Examples of the operations are: [ReLU](../ops/activation/ReLU_1.md), [Convolution](../ops/convolution/Convolution_1.md),
  [Add](../ops/arithmetic/Add_1.md), etc.

- *Kernel* — The implementation of a operation function in the OpenVINO™ plugin, in this case, the math programmed (in
  C++ and OpenCL) to perform the operation for a target hardware (CPU or GPU).

- *Inference Engine Extension* — Device-specific module implementing custom operations (a set of kernels).

## Custom Operation Support Overview

There are three steps to support inference of a model with custom operation(s):
1. Add support for a custom operation in the [Model Optimizer](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) so
the Model Optimizer can generate the IR with the operation.
2. Create an operation set and implement a custom nGraph operation in it as described in the
[Custom nGraph Operation](../IE_DG/Extensibility_DG/AddingNGraphOps.md).
3. Implement a customer operation in one of the [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)
plugins to support inference of this operation using a particular target hardware (CPU, GPU or VPU).

To see the operations that are supported by each device plugin for the Inference Engine, refer to the
[Supported Devices](../IE_DG/supported_plugins/Supported_Devices.md).

> **NOTE:** If a device doesn't support a particular operation, an alternative to creating a new operation is to target
> an additional device using the HETERO plugin. The [Heterogeneous Plugin](../IE_DG/supported_plugins/HETERO.md) may be
> used to run an inference model on multiple devices allowing the unsupported operations on one device to "fallback" to
> run on another device (e.g., CPU) that does support those operations.

### Custom Operation Support for the Model Optimizer

Model Optimizer model conversion pipeline is described in details in "Model Conversion Pipeline" section on the
[Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md).
It is recommended to read that article first for a better understanding of the following material.

Model Optimizer provides extensions mechanism to support new operations and implement custom model transformations to
generate optimized IR. This mechanism is described in the "Model Optimizer Extensions" section on the
[Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md).

Two types of the Model Optimizer extensions should be implemented to support custom operation at minimum:
1. Operation class for a new operation. This class stores information about the operation, its attributes, shape
inference function, attributes to be saved to an IR and some others internally used attributes. Refer to the
"Model Optimizer Operation" section on the
[Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) for the
detailed instruction on how to implement it.
2. Operation attributes extractor. The extractor is responsible for parsing framework-specific representation of the
operation and uses corresponding operation class to update graph node attributes with necessary attributes of the
operation. Refer to the "Operation Extractor" section on the
[Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) for the
detailed instruction on how to implement it.

> **NOTE:** In some cases you may need to implement some transformation to support the operation. This topic is covered
> in the "Graph Transformation Extensions" section on the
> [Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md).

## Custom Operations Extensions for the Inference Engine

Inference Engine provides extensions mechanism to support new operations. This mechanism is described in the
[Inference Engine Extensibility Mechanism](../IE_DG/Extensibility_DG/Intro.md).

Each device plugin includes a library of optimized implementations to execute known operations which must be extended to
execute a custom operation. The custom operation extension is implemented according to the target device:

- Custom Operation CPU Extension
   - A compiled shared library (`.so`, `.dylib` or `.dll`) needed by the CPU Plugin for executing the custom operation
   on a CPU. Refer to the [How to Implement Custom CPU Operations](../IE_DG/Extensibility_DG/CPU_Kernel.md) for more
   details.
- Custom Operation GPU Extension
   - OpenCL source code (.cl) for the custom operation kernel that will be compiled to execute on the GPU along with a
   operation description file (.xml) needed by the GPU Plugin for the custom operation kernel. Refer to the
   [How to Implement Custom GPU Operations](../IE_DG/Extensibility_DG/GPU_Kernel.md) for more details.
- Custom Operation VPU Extension
   - OpenCL source code (.cl) for the custom operation kernel that will be compiled to execute on the VPU along with a
   operation description file (.xml) needed by the VPU Plugin for the custom operation kernel. Refer to the
   [How to Implement Custom Operations for VPU](../IE_DG/Extensibility_DG/VPU_Kernel.md) for more details.

Also, it is necessary to implement nGraph custom operation according to the
[Custom nGraph Operation](../IE_DG/Extensibility_DG/AddingNGraphOps.md) so the Inference Engine can read an IR with this
operation and correctly infer output tensors shape and type.

## Enabling Magnetic Resonance Image Reconstruction Model
This chapter provides a step-by-step instruction on how to enable the magnetic resonance image reconstruction model
implemented in the [repository](https://github.com/rmsouza01/Hybrid-CS-Model-MRI/) using a custom operation on CPU. The
example is prepared for a model generated from the repository with hash `2ede2f96161ce70dcdc922371fe6b6b254aafcc8`.

### Download and Convert the Model to a Frozen TensorFlow\* Model Format
The original pre-trained model is provided in the hdf5 format which is not supported by OpenVINO directly and needs to
be converted to TensorFlow\* frozen model format first.

1. Download repository `https://github.com/rmsouza01/Hybrid-CS-Model-MRI`:<br
```bash
    git clone https://github.com/rmsouza01/Hybrid-CS-Model-MRI
    git checkout 2ede2f96161ce70dcdc922371fe6b6b254aafcc8
```

2. Convert pre-trained `.hdf5` to a frozen `.pb` graph using the following script (tested with TensorFlow==1.15.0 and
Keras==2.2.4) which should be executed from the root of the cloned repository:<br>
```py
    import keras as K
    import numpy as np
    import Modules.frequency_spatial_network as fsnet
    import tensorflow as tf

    under_rate = '20'

    stats = np.load("Data/stats_fs_unet_norm_" + under_rate + ".npy")
    var_sampling_mask = np.load("Data/sampling_mask_" + under_rate + "perc.npy")

    model = fsnet.wnet(stats[0], stats[1], stats[2], stats[3], kshape = (5,5), kshape2=(3,3))
    model_name = "Models/wnet_" + under_rate + ".hdf5"
    model.load_weights(model_name)

    inp = np.random.standard_normal([1, 256, 256, 2]).astype(np.float32)
    np.save('inp', inp)

    sess = K.backend.get_session()
    sess.as_default()
    graph_def = sess.graph.as_graph_def()
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['conv2d_44/BiasAdd'])
    with tf.gfile.FastGFile('wnet_20.pb', 'wb') as f:
        f.write(graph_def.SerializeToString())    
```
   
As a result the TensorFlow\* frozen model file "wnet_20.pb" is generated.

### Convert the Frozen TensorFlow\* Model to Intermediate Representation

Firstly, open the model in the TensorBoard or other TensorFlow* model visualization tool. The model supports dynamic
batch dimension because the value for the batch dimension is not hardcoded in the model. Model Optimizer need to set all
dynamic dimensions to some specific value to create the IR, therefore specify the command line parameter `-b 1` to set
the batch dimension equal to 1. The actual batch size dimension can be changed at runtime using the Inference Engine API
described in the [Using Shape Inference](../IE_DG/ShapeInference.md). Also refer to
[Converting a Model Using General Conversion Parameters](../MO_DG/prepare_model/convert_model/Converting_Model_General.md)
and [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
for more details and command line parameters used for the model conversion.

```bash
./<MO_INSTALL_DIR>/mo.py --input_model <PATH_TO_MODEL>/wnet_20.pb -b 1
```

Model Optimizer produces the following error:
```bash
[ ERROR ]  List of operations that cannot be converted to Inference Engine IR:
[ ERROR ]      Complex (1)
[ ERROR ]          lambda_2/Complex
[ ERROR ]      IFFT2D (1)
[ ERROR ]          lambda_2/IFFT2D
[ ERROR ]      ComplexAbs (1)
[ ERROR ]          lambda_2/Abs
[ ERROR ]  Part of the nodes was not converted to IR. Stopped.
```

The error means that the Model Optimizer doesn't know how to handle 3 types of TensorFlow\* operations: "Complex",
"IFFT2D" and "ComplexAbs". In order to see more details about the conversion process run the model conversion with
additional parameter `--log_level DEBUG`. It is worth to mention the following lines from the detailed output:

```bash
[ INFO ]  Called "tf_native_tf_node_infer" for node "lambda_2/Complex"
[ <TIMESTAMP> ] [ DEBUG ] [ tf:228 ]  Added placeholder with name 'lambda_2/lambda_3/strided_slice_port_0_ie_placeholder'
[ <TIMESTAMP> ] [ DEBUG ] [ tf:228 ]  Added placeholder with name 'lambda_2/lambda_4/strided_slice_port_0_ie_placeholder'
[ <TIMESTAMP> ] [ DEBUG ] [ tf:241 ]  update_input_in_pbs: replace input 'lambda_2/lambda_3/strided_slice' with input 'lambda_2/lambda_3/strided_slice_port_0_ie_placeholder'
[ <TIMESTAMP> ] [ DEBUG ] [ tf:249 ]  Replacing input '0' of the node 'lambda_2/Complex' with placeholder 'lambda_2/lambda_3/strided_slice_port_0_ie_placeholder'
[ <TIMESTAMP> ] [ DEBUG ] [ tf:241 ]  update_input_in_pbs: replace input 'lambda_2/lambda_4/strided_slice' with input 'lambda_2/lambda_4/strided_slice_port_0_ie_placeholder'
[ <TIMESTAMP> ] [ DEBUG ] [ tf:249 ]  Replacing input '1' of the node 'lambda_2/Complex' with placeholder 'lambda_2/lambda_4/strided_slice_port_0_ie_placeholder'
[ <TIMESTAMP> ] [ DEBUG ] [ tf:148 ]  Inferred shape of the output tensor with index '0' of the node 'lambda_2/Complex': '[  1 256 256]'
[ <TIMESTAMP> ] [ DEBUG ] [ infer:145 ]  Outputs:
[ <TIMESTAMP> ] [ DEBUG ] [ infer:32 ]  output[0]: shape = [  1 256 256], value = <UNKNOWN>
[ <TIMESTAMP> ] [ DEBUG ] [ infer:129 ]  --------------------
[ <TIMESTAMP> ] [ DEBUG ] [ infer:130 ]  Partial infer for lambda_2/IFFT2D
[ <TIMESTAMP> ] [ DEBUG ] [ infer:131 ]  Op: IFFT2D
[ <TIMESTAMP> ] [ DEBUG ] [ infer:132 ]  Inputs:
[ <TIMESTAMP> ] [ DEBUG ] [ infer:32 ]  input[0]: shape = [  1 256 256], value = <UNKNOWN>
```

This is a part of the log of the partial inference phase of the model conversion. See the "Partial Inference" section on
the [Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) for
more information about this phase. Model Optimizer inferred output shape for the unknown operation of type "Complex"
using a "fallback" to TensorFlow\*. However, it is not enough to generate the IR because Model Optimizer doesn't know
which  attributes of the operation should be saved to IR. So it is necessary to implement Model Optimizer extensions to
support these operations.

Before going into the extension development it is necessary to understand what these unsupported operations do according
to the TensorFlow\* framework specification.

* "Complex" - returns a tensor of complex type constructed from two real input tensors specifying real and imaginary
part of a complex number.
* "IFFT2D" - returns a tensor with inverse 2-dimensional discrete Fourier transform over the inner-most 2 dimensions of
 an input.
* "ComplexAbs" - returns a tensor with absolute values of input tensor with complex numbers.

The part of the model with all three unsupported operations is depicted below:

![Unsupported sub-graph](img/unsupported_subgraph.png)

This model uses complex numbers during the inference but Inference Engine does not support tensors of this data type. So
it is necessary to find a way how to avoid using tensors of such a type in the model. Fortunately, the complex tensor
appear as a result of "Complex" operation, is used as input in the "IFFT2D" operation then is passed to "ComplexAbs"
which produces real value tensor as output. So there are just 3 operations consuming/producing complex tensors in the
model.

Let's design an OpenVINO operation "FFT" which get a single real number tensor describing the complex number and
produces a single real number tensor describing output complex tensor. This way the fact that the model uses complex
numbers is hidden inside the "FFT" operation implementation. The operation gets a tensor of shape `[N, H, W, 2]` and
produces the output tensor with the same shape, where the innermost dimension contains pairs of real numbers describing
the complex number (its real and imaginary part). As we will see further this operation will allow us to support the
model. The implementation of the Model Optimizer operation should be saved to `mo_extensions/ops/FFT.py` file:

@snippet FFT.py fft:operation

The attribute `inverse` is a flag specifying type of the FFT to apply: forward or inverse.

See the "Model Optimizer Operation" section on the
[Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) for the
detailed instruction on how to implement the operation.

Now it is necessary to implement extractor for the "IFFT2D" operation according to the
"Operation Extractor" section on the 
[Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md). The
following snippet provides two extractors: one for "IFFT2D", another one for "FFT2D", however only on of  them is used
in this example. The implementation should be saved to the file `mo_extensions/front/tf/FFT_ext.py`.

@snippet FFT_ext.py fft_ext:extractor

> **NOTE:** The graph is in inconsistent state after extracting node attributes because according to original operation
> "IFFT2D" semantic it should have an input consuming a tensor of complex numbers, but the extractor instantiated an
> operation "FFT" which expects a real tensor with specific layout. But the inconsistency will be resolved during
> applying front phase transformations discussed below.

The output shape of the operation "AddV2" from the picture above is `[N, H, W, 2]`. Where the innermost dimension
contains pairs of real numbers describing the complex number (its real and imaginary part). The following "StridedSlice"
operations split the input tensor into 2 parts to get a tensor of real and a tensor of imaginary parts which are then
consumed with the "Complex" operation to produce a tensor of complex numbers. These "StridedSlice" and "Complex"
operations can be removed so the "FFT" operation will get a real value tensor encoding complex numbers. To achieve this
we implement the front phase transformation which searches for a pattern of two "StridedSlice" operations with specific
attributes producing data to "Complex" operation and removes it from the graph. Refer to the
"Pattern-Defined Front Phase Transformations" section on the
[Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md) for more
information on how this type of transformation works. The code snippet should be saved to the file
`mo_extensions/front/tf/Complex.py`.

@snippet Complex.py complex:transformation

> **NOTE:** The graph is in inconsistent state because the "ComplexAbs" operation consumes complex value tensor but
>  "FFT" produces real value tensor.

Now lets implement a transformation which replace a "ComplexAbs" operation with a sub-graph of primitive operations
which calculate the result using the following formulae: \f$module(z) = \sqrt{real(z) \cdot real(z) + imag(z) \cdot imag(z)}\f$.
Original "IFFT2D" operation produces tensor of complex values, but the "FFT" operation produces a real value tensor with
the same format and shape as the input for the operation. So the input shape for the "ComplexAbs" will be `[N, H, W, 2]`
with the innermost dimension containing tuple with real and imaginary part of a complex number. In order to calculate
absolute values for the complex tensor we do the following:
1. Raise all elements in the power of 2.
2. Calculate a reduced sum over the innermost dimension.
3. Calculate a square root.

The implementation should be saved to the file `mo_extensions/front/tf/ComplexAbs.py` and provided below:

@snippet ComplexAbs.py complex_abs:transformation

Now it is possible to convert the model using the following command line:
```bash
./<MO_INSTALL_DIR>/mo.py --input_model <PATH_TO_MODEL>/wnet_20.pb -b 1 --extensions mo_extensions/
```

The sub-graph corresponding to the originally non-supported one is depicted on the image below:

![Converted sub-graph](img/converted_subgraph.png)

> **NOTE:** Model Optimizer performed conversion of the model from NHWC to NCHW layout that is why the dimension with
> the value 2 moved to another position.

### Inference Engine Extension Implementation
Now it is necessary to implement the extension for the CPU plugin with operation "FFT" introduced previously. The code
below is based on the template extension described on the
[Inference Engine Extensibility Mechanism](../IE_DG/Extensibility_DG/Intro.md).

#### CMake Build File
The first step is to create a CMake configuration file which builds the extension. The content of the "CMakeLists.txt"
file is the following:

@snippet ../template_extension/CMakeLists.txt cmake:extension

The CPU FFT kernel implementation uses OpenCV to perform the FFT that is why the extension library is linked with
"opencv_core" which comes with the OpenVINO.

#### Custom nGraph Operation "FFT" Implementation
The next step is to create the nGraph operation FFT. The header file "fft_op.hpp" has the following content:

@snippet ../template_extension/fft_op.hpp fft_op:header

The operation has just one boolean attribute `inverse`. Implementation of the necessary nGraph operation functions are
in the "fft_op.cpp" file with the following content:

@snippet ../template_extension/fft_op.cpp fft_op:implementation

Refer to the [Custom nGraph Operation](../IE_DG/Extensibility_DG/AddingNGraphOps.md) for more details.

#### CPU FFT Kernel Implementation
The operation implementation for CPU plugin uses OpenCV to perform the FFT. The header file "fft_kernel.hpp" has the
following content:

@snippet ../template_extension/fft_kernel.hpp fft_kernel:header

The "fft_kernel.cpp" with the implementation of the CPU has the following content:

@snippet ../template_extension/fft_kernel.cpp fft_kernel:implementation

Refer to the [How to Implement Custom CPU Operations](../IE_DG/Extensibility_DG/CPU_Kernel.md) for more details.

#### Extension Library Implementation
The last step is to create an extension library "extension.cpp" and "extension.hpp" which will include the FFT
operation for the CPU plugin. The code of  the library is described in the [Extension Library](../IE_DG/Extensibility_DG/Extension.md).

### Building and Running the Custom Extension
In order to build the extension run the following:<br>
```bash
mkdir build && cd build
source /opt/intel/openvino/bin/setupvars.sh
cmake .. -DCMAKE_BUILD_TYPE=Release
make --jobs=$(nproc)
```

The result of this command is a compiled shared library (`.so`, `.dylib` or `.dll`). It should be loaded in the
application using `Core` class instance method `AddExtension` like this
`core.AddExtension(make_so_pointer<IExtension>(compiled_library_file_name), "CPU");`.

To test that the extension is implemented correctly we can run the "mri_reconstruction_demo.py" with the following content:

@snippet mri_reconstruction_demo.py mri_demo:demo

The script can be executed using the following command line:
```bash
python3 mri_reconstruction_demo.py \
        -m <PATH_TO_IR>/wnet_20.xml \
        -i <PATH_TO_SAMPLE_MRI_IMAGE>.npy \
        -p <Hybrid-CS-Model-MRI_repo>/Data/sampling_mask_20perc.npy \
        -l <PATH_TO_BUILD_DIR>/libtemplate_extension.so \
        -d CPU
```

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
- [Model Optimizer Extensibility](../MO_DG/prepare_model/customize_model_optimizer/Customize_Model_Optimizer.md)
- [Inference Engine Extensibility Mechanism](../IE_DG/Extensibility_DG/Intro.md)
- [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md)
- [Overview of OpenVINO™ Toolkit Pre-Trained Models](@ref omz_models_intel_index)
- For IoT Libraries and Code Samples see the [Intel® IoT Developer Kit](https://github.com/intel-iot-devkit).

## Converting Models:

- [Convert Your Caffe* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)
- [Convert Your Kaldi* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)
- [Convert Your TensorFlow* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)
- [Convert Your MXNet* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)
- [Convert Your ONNX* Model](../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md)
