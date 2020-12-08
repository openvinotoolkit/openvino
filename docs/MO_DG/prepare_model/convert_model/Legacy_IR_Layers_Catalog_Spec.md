# Intermediate Representation Notation Reference Catalog {#openvino_docs_MO_DG_prepare_model_convert_model_Legacy_IR_Layers_Catalog_Spec}

> **NOTE**: This IR Notation Reference is no longer supported since the new concept of operation sets is introduced in OpenVINO 2020.1 version. For a complete list of supported operations, see the [Intermediate Representation and Operation Sets](../../IR_and_opsets.md) topic.

## Table of Сontents <a name="toc"></a>

* <a href="#Activation">Activation Layer</a>
* <a href="#ArgMax">ArgMax Layer</a>
* <a href="#BatchNormalization">BatchNormalization Layer</a>
* <a href="#BinaryConvolution">BinaryConvolution Layer</a>
* <a href="#Bucketize">Bucketize Layer</a>
* <a href="#Broadcast">Broadcast Layer</a>
* <a href="#Clamp">Clamp Layer</a>
* <a href="#Concat">Concat Layer</a>
* <a href="#Const">Const Layer</a>
* <a href="#Convolution">Convolution Layer</a>
* <a href="#Crop_1">Crop (Type 1) Layer</a>
* <a href="#Crop_2">Crop (Type 2) Layer</a>
* <a href="#Crop_3">Crop (Type 3) Layer</a>
* <a href="#CTCGreadyDecoder">CTCGreadyDecoder Layer</a>
* <a href="#Deconvolution">Deconvolution Layer</a>
* <a href="#DeformableConvolution">DeformableConvolution Layer</a>
* <a href="#DepthToSpace">DepthToSpace Layer</a>
* <a href="#DetectionOutput">DetectionOutput Layer</a>
* <a href="#Erf">Erf Layer</a>
* <a href="#Eltwise">Eltwise Layer</a>
* <a href="#Fill">Fill Layer</a>
* <a href="#Flatten">Flatten Layer</a>
* <a href="#FullyConnected">FullyConnected Layer</a>
* <a href="#Gather">Gather Layer</a>
* <a href="#GRN">GRN Layer</a>
* <a href="#GRUCell">GRUCell Layer</a>
* <a href="#Input">Input Layer</a>
* <a href="#Interp">Interp Layer</a>
* <a href="#LSTMCell">LSTMCell Layer</a>
* <a href="#Memory">Memory Layer</a>
* <a href="#MVN">MVN Layer</a>
* <a href="#NonMaxSuppression">NonMaxSuppression Layer</a>
* <a href="#Norm">Norm Layer</a>
* <a href="#Normalize">Normalize Layer</a>
* <a href="#OneHot">OneHot Layer</a>
* <a href="#Pad">Pad Layer</a>
* <a href="#Permute">Permute Layer</a>
* <a href="#Pooling">Pooling Layer</a>
* <a href="#Power">Power Layer</a>
* <a href="#PReLU">PReLU Layer</a>
* <a href="#PriorBox">PriorBox Layer</a>
* <a href="#PriorBoxClustered">PriorBoxClustered Layer</a>
* <a href="#Proposal">Proposal Layer</a>
* <a href="#PSROIPooling">PSROIPooling Layer</a>
* <a href="#FakeQuantize">FakeQuantize Layer</a>
* <a href="#Range">Range Layer</a>
* <a href="#RegionYolo">RegionYolo Layer</a>
* <a href="#ReLU">ReLU Layer</a>
* <a href="#ReorgYolo">ReorgYolo Layer</a>
* <a href="#Resample_1">Resample (Type 1) Layer</a>
* <a href="#Resample_2">Resample (Type 2) Layer</a>
* <a href="#Reshape">Reshape Layer</a>
* <a href="#ReverseSequence">ReverseSequence Layer</a>
* <a href="#RNNCell">RNNCell Layer</a>
* <a href="#ROIPooling">ROIPooling Layer</a>
* <a href="#ExperimentalDetectronROIFeatureExtractor">ExperimentalDetectronROIFeatureExtractor layer</a>
* <a href="#ExperimentalSparseWeightedSum">ExperimentalSparseWeightedSum layer</a>
* <a href="#ScaleShift">ScaleShift Layer</a>
* <a href="#Select">Select Layer</a>
* <a href="#Shape">Shape Layer</a>
* <a href="#ShuffleChannels">ShuffleChannels Layer</a>
* <a href="#SimplerNMS">SimplerNMS Layer</a>
* <a href="#Slice">Slice Layer</a>
* <a href="#SoftMax">SoftMax Layer</a>
* <a href="#SparseFillEmptyRows">SparseFillEmptyRows Layer</a>
* <a href="#SparseSegmentMean">SparseSegmentMean Layer</a>
* <a href="#SparseSegmentSqrtN">SparseSegmentSqrtN Layer</a>
* <a href="#SparseSegmentSum">SparseSegmentSum Layer</a>
* <a href="#SparseToDense">SparseToDense Layer</a>
* <a href="#Split">Split Layer</a>
* <a href="#Squeeze">Squeeze Layer</a>
* <a href="#StridedSlice">StridedSlice Layer</a>
* <a href="#TensorIterator">TensorIterator Layer</a>
* <a href="#Tile">Tile Layer</a>
* <a href="#TopK">TopK Layer</a>
* <a href="#Unique">Unique Layer</a>
* <a href="#Unsqueeze">Unsqueeze Layer</a>

## Activation Layer <a name="Activation"></a>
<a href="#toc">Back to top</a>

**Name**: *Activation*

**Category**: *Activation*

**Short description**: *Activation* layer represents an activation function of each neuron in a layer, which is used to add non-linearity to the computational flow.

**Detailed description**: [Reference](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)

**Parameters**: *Activation* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *type*

  * **Description**: *type* represents particular activation function. For example, *type* equal to `sigmoid` means that the neurons of this layer have a sigmoid activation function.
  * **Range of values**:
    * *sigmoid* - sigmoid activation function. Learn more from the **Detailed description** section.
    * *tanh* - tanh activation function. Learn more from the **Detailed description** section.
    * *elu* - elu activation function. Learn more from the **Detailed description** section.
    * *relu6* - relu6 activation function
    * *not* - logical NOT function
    * *exp* - exponent function
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*

**Mathematical Formulation**

*   Sigmoid function:
    \f[
    f(x) = \frac{1}{1+e^{-x}}
    \f]
*   Tahn function:
    \f[
    f (x) = \frac{2}{1+e^{-2x}} - 1 = 2sigmoid(2x) - 1
    \f]
*   Elu function:
    \f[
    f(x) = \left\{\begin{array}{ll}
	e^{x} - 1 \quad \mbox{if } x < 0 \\
	x \quad \mbox{if } x \geq  0
    \end{array}\right.
    \f]
*   Relu6 function:
    \f[
    f(x) = min(max(0, x), 6)
    \f]

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Example**

```xml
<layer ... type="Activation" ... >
    <data type="sigmoid" />
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## ArgMax Layer <a name="ArgMax"></a>
<a href="#toc">Back to top</a>

**Name**: *ArgMax*

**Category**: *Layer*

**Short description**: *ArgMax* layer computes indexes and values of the *top_k* maximum values for each datum across all dimensions *CxHxW*.

**Detailed description**: *ArgMax* layer is used after a classification layer to produce a prediction. If the parameter *out_max_val* is 1, output is a vector of pairs `(max_ind, max_val)` for each batch. The *axis* parameter specifies an axis along which to maximize.

**Parameters**: *ArgMax* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *out_max_val*

  * **Description**: If *out_max_val* is 1, the output is a list of pairs `(max_ind, max_val)`. If *out_max_val* is 0, the output is a list of indexes of size *top_k*.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *top_k*

  * **Description**: *top_k* is the number of elements to save in output.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *axis*

  * **Description**: If *axis* is set, maximizes along the specified axis, else maximizes the flattened trailing dimensions for each index of the first / num dimension.
  * **Range of values**: an integer. Negative value means counting dimension from the end.
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *no*

**Inputs**:

*   **1**: 4D input blob. Required.

**Mathematical Formulation**

*ArgMax* generally does the following with the input blobs:
\f[
o_{i} = \left\{
x| x \in S  \wedge \forall y \in S : f(y) \leq f(x)
\right\}
\f]

**Example**

```xml
<layer ... type="ArgMax" ... >
    <data top_k="10" out_max_val="1" axis="-1"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## BatchNormalization Layer <a name="BatchNormalization"></a>
<a href="#toc">Back to top</a>

**Name**: *BatchNormalization*

**Category**: *Normalization*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/batchnorm.html)

**Detailed description**: [Reference](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

**Parameters**: *BatchNormalization* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *epsilon*

  * **Description**: *epsilon* is the number to be added to the variance to avoid division by zero when normalizing a value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: 4D input blob. Required.

**Mathematical Formulation**

*BatchNormalization* normalizes the output in each hidden layer.
*   **Input**: Values of \f$x\f$ over a mini-batch:
    \f[
    \beta = \{ x_{1...m} \}
    \f]
*   **Parameters to learn**: \f$ \gamma, \beta\f$
*   **Output**:
    \f[
    \{ o_{i} = BN_{\gamma, \beta} ( b_{i} ) \}
    \f]
*   **Mini-batch mean**:
    \f[
    \mu_{\beta} \leftarrow \frac{1}{m}\sum_{i=1}^{m}b_{i}
    \f]
*   **Mini-batch variance**:
    \f[
    \sigma_{\beta }^{2}\leftarrow \frac{1}{m}\sum_{i=1}^{m} ( b_{i} - \mu_{\beta} )^{2}
    \f]
*   **Normalize**:
    \f[
    \hat{b_{i}} \leftarrow \frac{b_{i} - \mu_{\beta}}{\sqrt{\sigma_{\beta }^{2} + \epsilon }}
    \f]
*   **Scale and shift**:
    \f[
    o_{i} \leftarrow \gamma\hat{b_{i}} + \beta = BN_{\gamma ,\beta } ( b_{i} )
    \f]

**Example**

```xml
<layer ... type="BatchNormalization" ... >
    <data epsilon="9.99e-06" />
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *


## BinaryConvolution Layer <a name="BinaryConvolution"></a>
<a href="#toc">Back to top</a>

**Name**: *BinaryConvolution*

**Category**: *Layer*

**Short description**: *BinaryConvolution* convolution with binary weights

**Parameters**: *BinaryConvolution* layer parameters are specified in the `data` node, which is a child of the `layer` node. The layer has the same parameters as a regular *Convolution* layer and several unique parameters.

* **Parameter name**: *input*

  * **Description**: *input* is the number of input channels.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *mode*

  * **Description**: *mode* defines how input tensor 0/1 values and weights 0/1 are interpreted as real numbers and how the result is computed.
  * **Range of values**:
    * *xnor-popcount*
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *pad_value*

  * **Description**: *pad_value* is a floating-point value used to fill pad area.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: 4D input blob containing integer or floats; filled with 0/1 values. 0 means -1, 1 means 1 for `mode="xnor-popcount"`. Required.

* * *

## Bucketize Layer <a name="Bucketize"></a>
<a href="#toc">Back to top</a>

**Name**: *Bucketize*

**Category**: *Layer*

**Short description**: *Bucketize* bucketizes the input based on boundaries. This is an equivalent to np.digitize.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/bucketize)

* **Parameter name**: *with_right_bound*

  * **Description**: Indicates whether the intervals include the right or the left bucket edge.
  * **Range of values**: *True* or *False*
  * **Type**: `bool`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: N-D tensor. Input tensor for the bucketization. It contains with float or integer types. Required.
*   **2**: 1-D tensor. Sorted boundaries of the buckets. It contains with a float type. Required.

**Outputs**:

*   **1**: Output tensor with bucket indices for each element of the first input tensor. If the second input is empty, the bucket indice for all elements is equal to 0. The output tensor shape is the same as the first input tensor shape.

* * *

## Clamp Layer <a name="Clamp"></a>
<a href="#toc">Back to top</a>

**Name**: *Clamp*

**Category**: *Layer*

**Short description**: *Clamp* layer represents clipping activation operation.

**Detailed description**: [Reference](https://www.tensorflow.org/versions/r1.2/api_docs/MO_DG/prepare_model/python/tf/clip_by_value)

**Parameters**: *Clamp* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *min*

  * **Description**: *min* is the lower bound of values in the output. Any value in the input that is smaller than the bound is replaced by the *min* value. For example, *min* equal to 10.0 means that any value in the input that is smaller than the bound is replaced by 10.0.
  * **Range of values**: a non-negative floating-point number
  * **Type**: `float`
  * **Default value**: 0.0
  * **Required**: *yes*

* **Parameter name**: *max*

  * **Description**: *max* is the upper bound of values in the output. Any value in the input that is greater than the bound, is replaced by the *max* value. For example, *max* equal to 50.0 means that any value in the input that is greater than the bound is replaced by 50.0.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: 6.0
  * **Required**: *yes*

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Mathematical Formulation**

*Clamp* generally does the following with the input blobs:
\f[
out_i=\left\{\begin{array}{ll}
	max\_value \quad \mbox{if } \quad input_i>max\_value \\
	min\_value \quad \mbox{if } \quad input_i
\end{array}\right.
\f]

**Example**

```xml
<layer ... type="Clamp" ... >
    <data min="10" max="50" />
    <input> ... </input>
    <output> ... </output>
</layer>
```


* * *

## Broadcast <a name="Broadcast"></a>
<a href="#toc">Back to top</a>

**Category**: Layer

**Short description**: *Broadcast* replicates data on the first input to fit a given shape.

**Detailed description**:

*Broadcast* takes the first tensor and, following the [NumPy broadcasting rules specification](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html), builds a new tensor with shape matching the second input tensor. The second input value represents desired output shape.

**Parameters**: *Broadcast* layer does not have parameters.

**Inputs**:

*   **1**: Source tensor that is being broadcasted. Required.

*   **2**: 1D tensor describing output shape. Required.


**Outputs**:

*   **1**: Output tensor with replicated content from the first tensor with shape defined by the second input.

**Example**

```xml
<layer ... type="Broadcast" ...>
    <input>
        <port id="0">
            <dim>16</dim>
            <dim>1</dim>
            <dim>1</dim>
       </port>
        <port id="1">
            <dim>4</dim>   <!--The tensor contains 4 elements: 1, 16, 50, 50 -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>16</dim>
            <dim>50</dim>
            <dim>50</dim>
        </port>
    </output>
</layer>
```

* * *


## Concat Layer <a name="Concat"></a>
<a href="#toc">Back to top</a>

**Name**: *Concat*

**Category**: *Layer*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/concat.html)

**Parameters**: *Concat* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* is the number of axis over which input blobs are concatenated. For example, *axis* equal to 1 means that input blobs are concatenated over the first axis.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *yes*

**Inputs**:

*   **1**: Multidimensional input blob. Required.

*   **2**: Multidimensional input blob. Required.

**Mathematical Formulation**

*Axis* parameter specifies a blob dimension to concatenate values over. For example, for two input blobs *B1xC1xH1xW1* and *B2xC2xH2xW2*, if `axis="1"`, the output blob is *B1xC1+C2xH1xW1*. This is only possible if *B1=B2*, *H1=H2*, *W1=W2*.

**Example**

```xml
<layer ... type="Concat" ... >
    <data axis="1"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Const Layer <a name="Const"></a>
<a href="#toc">Back to top</a>

**Name**: *Const*

**Category**: *Layer*

**Short description**: *Const* layer produces a blob with a constant value specified in the *blobs* section.

**Parameters**: *Const* layer does not have parameters.

**Example**

```xml
<layer ... type="Const" ...>
    <output>
        <port id="1">
            <dim>3</dim>
            <dim>100</dim>
        </port>
    </output>
    <blobs>
        <custom offset="..." size="..."/>
    </blobs>
</layer>
```

* * *

## Convolution Layer <a name="Convolution"></a>
<a href="#toc">Back to top</a>

**Name**: *Convolution*

**Category**: *Layer*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/convolution.html)

**Detailed description**: [Reference](http://cs231n.github.io/convolutional-networks/#conv)

**Parameters**: *Convolution* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the `(z, y, x)` axes for 3D convolutions and `(y, x)` axes for 2D convolutions. For example, *strides* equal to "4,2,1" means sliding the filter four pixels at a time over depth dimension, two pixels over height dimension, and one pixel over width dimension.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 1 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *pads_begin*

  * **Description**: *pads_begin* is the number of pixels to add to the beginning of each axis. For example, *pads_begin* equal to "1,2" means adding one pixel to the top of the input and two pixels to the left of the input.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 0 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *pads_end*

  * **Description**: *pads_end* is the number of pixels to add to the end of each axis. For example, *pads_end* equal to "1,2" means adding one pixel to the bottom of the input and two pixels to the right of the input.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 0 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *kernel*

  * **Description**: *kernel* is a size of each filter. For example, *kernel* equal to "2,3" means that each filter has height equal to 2 and width equal to 3.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *output*

  * **Description**: *output* is a number of output feature maps in the output. If *group* parameter value is greater than 1, *output* still matches the number of output features regardless of *group* value. For example, *output* equal to 1 means that there is one output feature map in a layer.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *group*

  * **Description**: *group* is the number of groups which *output* and *input* should be split into. For example, *group* equal to 1 means that all filters are applied to the whole input (usual convolution), *group* equal to 2 means that both *input* and *output* channels are separated into two groups and the *i-th output* group is connected to the *i-th input* group channel. *group* equal to a number of output feature maps implies depth-wise separable convolution. For more information, see the [Reference](https://medium.com/towards-data-science/types-of-convolutions-in-deep-learning-717013397f4d#6f51).
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *dilations*

  * **Description**: *dilations* is a distance in width and height between elements (weights) in the filter. For example, *dilations* equal to "1,1" means that all elements in the filter are neighbors, so it is the same as the usual convolution. *dilations* equal to "2,2" means that all elements in the filter are matched to the elements in the input matrix separated by one pixel.
  * **Range of values**: a non-negative integer
  * **Type**: `int[]`
  * **Default value**: a list of 1 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *auto_pad*

  * **Description**: *auto_pad* defines how the padding is calculated. Possible values:
    * Not specified: use explicit padding values
    * *same_upper/same_lower*: add paddings to the input to match the output size. In case of odd padding value, an extra padding is added to the beginning if `auto_pad="same_upper"` or to the end if `auto_pad="same_lower"`.
    * *valid*: do not use padding
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *no*

**Inputs**:

*   **1**: 4D or 5D input blob. Required.

**Weights Layout**

Weights layout is GOIYX (GOIZYX for 3D convolution), which means that *X* changes the fastest, then *Y*, *Input* and *Output*, *Group*.


**Mathematical Formulation**

*   For the convolutional layer, the number of output features in each dimension is calculated as:
\f[
n_{out} = \left ( \frac{n_{in} + 2p - k}{s} \right ) + 1
\f]
*   The receptive field in each layer is calculated as:
    *   Jump in the output feature map:
        \f[
        j_{out} = j_{in} * s
        \f]
    *   Size of the receptive field of output feature:
        \f[
        r_{out} = r_{in} + ( k - 1 ) * j_{in}
        \f]
    *   Center position of the receptive field of the first output feature:
        \f[
        start_{out} = start_{in} + ( \frac{k - 1}{2} - p ) * j_{in}
        \f]
    *   Output is calculated as:
        \f[
        out = \sum_{i = 0}^{n}w_{i}x_{i} + b
        \f]

**Example**

```xml
<layer ... type="Convolution" ... >
        <data auto_pad="same_upper" dilations="1,1" group="3" kernel="7,7" output="24" pads_begin="2,2" pads_end="3,3" strides="2,2"/>
        <input> ... </input>
        <output> ... </output>
        <weights ... />
        <biases ... />
    </layer>
```

* * *

## Crop (Type 1) Layer <a name="Crop_1"></a>
<a href="#toc">Back to top</a>

**Name**: *Crop*

**Category**: *Layer*

**Short description**: *Crop* layer changes selected dimensions of the input blob according to the specified parameters.

**Parameters**: *Crop* layer parameters are specified in the `data` section, which is a child of the `layer` node. *Crop* **Type 1** layer takes two input blobs, and the shape of the second blob specifies the *Crop* size. The *Crop* layer of this type supports shape inference.

* **Parameter name**: *axis*

  * **Description**: *axis* is the number of a dimension to crop. For example, *axis* equal to [1] means that the first dimension is cropped.
  * **Range of values**: a list of unique integers, where each element is greater than or equal to 0 and less than input shape length
  * **Type**: `int[]`
  * **Default value**: `[1]`
  * **Required**: *yes*

* **Parameter name**: *offset*

  * **Description**: *offset* is the starting point for crop in the input blob. For example, *offset* equal to 2 means that crop starts from the second value of a specified axis.
  * **Range of values**: a list of integers of the length equal to the length of the *axis* attribute. In the list, *offset[i]* is greater than or equal to 0 and less than or equal to *input_shape[axis[i]] - crop_size[axis[i]]*, where *crop_size* is the shape of the second input.
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: Multidimensional input blob
* **2**: Shape of this input will be used for crop

**Example**

```xml
<layer id="39" name="score_pool4c" precision="FP32" type="Crop">
    <data axis="2,3" offset="0,0"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>21</dim>
            <dim>44</dim>
            <dim>44</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>21</dim>
            <dim>34</dim>
            <dim>34</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>21</dim>
            <dim>34</dim>
            <dim>34</dim>
        </port>
    </output>
</layer>
```

* * *

## Crop (Type 2) Layer <a name="Crop_2"></a>
<a href="#toc">Back to top</a>

**Name**: *Crop*

**Category**: *Layer*

**Short description**: *Crop* layer changes selected dimensions of the input blob according to the specified parameters.

**Parameters**: Specify parameters for the *Crop* layer in the `data` section, which is a child of the `layer` node. *Crop* **Type 2** layer takes one input blob to crop. The *Crop* layer of this type supports shape inference only when shape propagation is applied to dimensions not specified in the *axis* attribute.

* **Parameter name**: *axis*

  * **Description**: *axis* is the number of a dimension to crop. For example, *axis* equal to [1] means that the first dimension is cropped.
  * **Range of values**: a list of unique integers, where each element is greater than or equal to 0 and less than input shape length
  * **Type**: `int[]`
  * **Default value**: `[1]`
  * **Required**: *yes*

* **Parameter name**: *offset*

  * **Description**: *offset* is the starting point for crop in the input blob. For example, *offset* equal to 2 means that cropping starts from the second value of the specified axis.
  * **Range of values**: a list of integers with the length equal to the length of *axis* attribute, where *offset[i]* is greater than or equal to 0 and less or equal to *input_shape[axis[i]] - dim[i]*
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *dim*

  * **Description**: *dim* is the resulting size of the output blob for the specified axis. For example, *dim* equal to [88] means that the output blob gets the dimension equal to 88 for the specified axis.
  * **Range of values**: a list of integers
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

**Example**

```xml
<layer id="39" name="score_pool4c" precision="FP32" type="Crop">
    <data axis="2,3" offset="0,0" dim="34,34"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>21</dim>
            <dim>44</dim>
            <dim>44</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>21</dim>
            <dim>34</dim>
            <dim>34</dim>
        </port>
    </output>
</layer>
```

* * *

## Crop (Type 3) Layer <a name="Crop_3"></a>
<a href="#toc">Back to top</a>

**Name**: *Crop*

**Category**: *Layer*

**Short description**: *Crop* layer changes selected dimensions of the input blob according to the specified parameters.

**Parameters**: *Crop* layer parameters are specified in the `data` section, which is a child of the `layer` node. *Crop* **Type 3** layer takes one input blob to crop. The *Crop* layer of this type supports shape inference.

* **Parameter name**: *axis*

  * **Description**: *axis* is the number of a dimension to crop. For example, *axis* equal to [1] means that the first dimension is cropped.
  * **Range of values**: a list of unique integers, where each element is greater than or equal to 0 and less than input shape length
  * **Type**: `int[]`
  * **Default value**: `[1]`
  * **Required**: *yes*

* **Parameter name**: *crop_begin*

  * **Description**: *crop_begin* specifies the starting offset for crop in the input blob for a specified axes.
  * **Range of values**: a list of integers, where *crop_begin[i]* is greater than or equal to 0 and less than *input_shape[axis[i]] - crop_end[i]*
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *crop_end*

  * **Description**: *crop_end* specifies the ending offset for crop in the input blob for the specified axes.
  * **Range of values**: a list of integers, where *crop_end[i]* is greater than or equal to 0 and less than *input_shape[axis[i]] - crop_begin[i]*
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

**Example**

```xml
<layer id="39" name="score_pool4c" precision="FP32" type="Crop">
    <data axis="2,3" crop_begin="4,4" crop_end="6,6"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>21</dim>
            <dim>44</dim>
            <dim>44</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>21</dim>
            <dim>34</dim>
            <dim>34</dim>
        </port>
    </output>
</layer>
```

* * *

## CTCGreedyDecoder Layer <a name="CTCGreedyDecoder"></a>
<a href="#toc">Back to top</a>

**Name**: *CTCGreedyDecoder*

**Category**: *Layer*

**Short description**: *CTCGreedyDecoder* performs greedy decoding on the logits given in input (best path).

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder)

**Parameters**: *CTCGreedyDecoder* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *ctc_merge_repeated*

  * **Description**: *ctc_merge_repeated* is a flag for merging repeated labels during the CTC calculation.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Mathematical Formulation**

Given an input sequence \f$X\f$ of length \f$T\f$, *CTCGreadyDecoder* assumes the probability of a length \f$T\f$ character sequence \f$C\f$ is given by
\f[
p(C|X) = \prod_{t=1}^{T} p(c_{t}|X)
\f]

**Example**

```xml
<layer ... type="CTCGreadyDecoder" ... >
    <data stride="1"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Deconvolution Layer <a name="Deconvolution"></a>
<a href="#toc">Back to top</a>

**Name**: *Deconvolution*

**Category**: *Layer*

**Short description**: *Deconvolution* layer is applied for upsampling the output to the higher image resolution.

**Detailed description**: [Reference](https://distill.pub/2016/deconv-checkerboard/)

**Parameters**: *Deconvolution* layer parameters should be specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the filter on the feature map over the `(z, y, x)` axes for 3D deconvolutions and `(y, x)` axes for 2D deconvolutions. For example, *strides* equal to "4,2,1" means sliding the filter four pixels at a time over depth dimension, two pixels over height dimension, and one pixel over width dimension.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 1 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *pads_begin*

  * **Description**: *pads_begin* is the number of pixels to add to the beginning of each axis. For example, *pads_begin* equal to "1,2" means adding one pixel to the top of the input and two pixels to the left of the input.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 0 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *pads_end*

  * **Description**: *pads_end* is the number of pixels to add to the end of each axis. For example, *pads_end* equal to "1,2" means adding one pixel to the bottom of the input and two pixels to the right of the input.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 1 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *kernel*

  * **Description**: *kernel* is a size of each filter. For example, *kernel* equal to "2,3" means that each filter has height equal to 2 and width equal to 3.
  * **Range of values**: a list of positive integers
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *output*

  * **Description**: *output* is the number of output feature maps in the output. If *group* parameter value is greater than 1, *output* still matches the number of output features regardless of the *group* value. For example, *output* equal to 1 means that there is one output feature map in a layer.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *group*

  * **Description**: *group* denotes the number of groups to which *output* and *input* should be split. For example, *group* equal to 1 means that all filters are applied to the whole input (usual convolution), *group* equal to 2 means that both *input* and *output* channels are separated into 2 groups and *i-th output* group is connected to *i-th input* group channels. *group* equal to a number of output feature maps implies depth-wise separable convolution. For more information, see the [Reference](https://medium.com/towards-data-science/types-of-convolutions-in-deep-learning-717013397f4d#6f51).
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *dilations*

  * **Description**: *dilations* is the distance in width and height between elements (weights) in the filter. For example, *dilation* equal to "1,1" means that all elements in the filter are neighbors, so it is the same as the usual convolution. *dilation* equal to "2,2" means that all elements in the filter are matched to the elements in the input matrix separated by one pixel.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 1 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *auto_pad*

  * **Description**: *auto_pad* defines how the padding is calculated.
  * **Range of values**:
    * Not specified: use explicit padding values.
    * *same_upper/same_lower*: add paddings to the input to match the output size. In case of odd padding value, an extra padding is added to the beginning if `auto_pad="same_upper"` or to the end if `auto_pad="same_lower"`.
    * *valid*: do not use padding
  * **Type**: string
  * **Default value**: None
  * **Required**: *no*

**Inputs**:

*   **1**: 4D or 5D blob with input data. Required.

**Weights Layout**

Weights layout is GOIYX, which means that *X* changes the fastest, then *Y*, *Input* and *Output*, *Group*.


**Mathematical Formulation**

*Deconvolution* is also called transpose convolution and performs operation that is reverse to convolution.
The number of output features for each dimensions is calculated as:
\f[S_{o}=stride(S_{i} - 1 ) + S_{f} - 2pad \f]
Where \f$S\f$ is the size of output, input, and filter.
Output is calculated in the same way as for convolution layer:
\f[out = \sum_{i = 0}^{n}w_{i}x_{i} + b\f]

**Example**

```xml
<layer ... type="Deconvolution" ...>
    <data auto_pad="valid" kernel="2,2,2" output="512" pads_begin="0,0,0" pads_end="0,0,0" strides="2,2,2"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>512</dim>
            <dim>8</dim>
            <dim>8</dim>
            <dim>8</dim>
        </port>
    </input>
    <output>
        <port id="3">
            <dim>1</dim>
            <dim>512</dim>
            <dim>16</dim>
            <dim>16</dim>
            <dim>16</dim>
        </port>
    </output>
    <blobs>
        <weights offset="..." size="..."/>
        <biases offset="..." size="..."/>
    </blobs>
</layer>
```

* * *

## DeformableConvolution Layer <a name="DeformableConvolution"></a>
<a href="#toc">Back to top</a>

**Name**: *DeformableConvolution*

**Category**: *Layer*

**Short description**: *DeformableConvolution* convolution layer enhances the transformation modeling capacity of CNNs.

**Detailed description**: [Reference](https://arxiv.org/abs/1703.06211)

**Parameters**: *DeformableConvolution* layer parameters are specified in the `data` node, which is a child of the `layer` node. The layer has the same parameters as a regular *Convolution* layer and several unique parameters.

* **Parameter name**: *num_deformable_group*

  * **Description**: *num_deformable_group* is the number of deformable group partitions.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: 4D or 5D blob with input data. Required.
*   **2**: Input offset to the DeformableConvolution

**Weights Layout**

Weights layout is GOIYX (GOIZYX for 3D convolution), which means that *X* changes the fastest, then *Y*, *Input* and *Output*, *Group*.

**Example**

```xml
<layer ... type="DeformableConvolution">
	<data deformable_group="4" dilations="2,2" group="1" kernel="3,3" output="512" pads_begin="2,2" pads_end="2,2" strides="1,1"/>
	<input>
		<port id="0">
			<dim>1</dim>
			<dim>512</dim>
			<dim>40</dim>
			<dim>27</dim>
		</port>
		<port id="1">
			<dim>1</dim>
			<dim>72</dim>
			<dim>40</dim>
			<dim>27</dim>
		</port>
	</input>
	<output>
		<port id="3">
			<dim>1</dim>
			<dim>512</dim>
			<dim>40</dim>
			<dim>27</dim>
		</port>
	</output>
	<blobs>
		<weights offset="121799456" size="9437184"/>
	</blobs>
</layer>
```

* * *

## DepthToSpace Layer <a name="DepthToSpace"></a>
<a href="#toc">Back to top</a>

**Name**: *DepthToSpace*

**Category**: *Layer*

**Short description**: *DepthToSpace* layer rearranges data from the depth dimension of the input blob into spatial dimensions.

**Detailed description**:  *DepthToSpace* layer outputs a copy of the input blob, where values from the depth dimension (features) are moved to spatial blocks. Refer to the [ONNX* specification](https://github.com/onnx/onnx/blob/master/docs/Operators.md#DepthToSpace) for an example of the 4D input blob case.

**Parameters**: *DepthToSpace* layer parameters are specified parameters in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *block_size*

  * **Description**: *block_size* specifies the size of the value block to be moved. The depth dimension size must be evenly divided by `block_size ^ (len(input.shape) - 2)`.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: 3D+ blob with input data. Required.

**Mathematical Formulation**

The operation is equivalent to the following transformation of the input blob *x* with *K* spatial dimensions of shape *[N, C, D1, D2, D3 , ... , DK]*:

```
x' = reshape(x, [N, block_size, block_size, ... , block_size, D1 * block_size, D2 * block_size, ... Dk * block_size])
x'' = transpose(x', [0, K + 1, K + 2, 1, K + 3, 2, K + 4, 3, ... K + K + 1, K])
y = reshape(x'', [N, C / block_size ^ K, D1 * block_size, D2 * block_size, D3 * block_size, ... , DK * block_size])

```
**Example**

```xml
<layer ... type="DepthToSpace">
    <data block_size="2"/>
    <input>
        <port id="0">
            <dim>5</dim>
            <dim>4</dim>
            <dim>2</dim>
            <dim>3</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>5</dim>  <!-- input.shape[0] -->
            <dim>1</dim>  <!-- input.shape[1] / (block_size ^ 2) -->
            <dim>4</dim>  <!-- input.shape[2] * block_size -->
            <dim>6</dim>  <!-- input.shape[3] * block_size -->
        </port>
    </output>
</layer>
```

* * *

## DetectionOutput Layer <a name="DetectionOutput"></a>
<a href="#toc">Back to top</a>

**Name**: *DetectionOutput*

**Category**: *Layer*

**Short description**: *DetectionOutput* layer performs non-maximum suppression to generate the detection output using information on location and confidence predictions.

**Detailed description**: [Reference](https://arxiv.org/pdf/1512.02325.pdf). The layer has three required inputs: blob with box logits, blob with confidence predictions, and blob with box coordinates (proposals). It can have two additional inputs with additional confidence predictions and box coordinates described in the [article](https://arxiv.org/pdf/1711.06897.pdf). The five input version of the layer is supported with MYRIAD plugin only. The output blob contains information about filtered detections described with seven element tuples: *[batch_id, class_id, confidence, x_1, y_1, x_2, y_2]*. The first tuple with *batch_id* equal to -1 means end of output.

**Parameters**: *DetectionOutput* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *num_classes*

  * **Description**: *num_classes* is the number of classes to be predicted.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *background_label_id*

  * **Description**: *background_label_id* is the background label ID. If there is no background class, set it to -1.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *top_k*

  * **Description**: *top_k* is the maximum number of results to keep per batch after NMS step. -1 means keeping all bounding boxes.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: -1
  * **Required**: *no*

* **Parameter name**: *variance_encoded_in_target*

  * **Description**: *variance_encoded_in_target* is a flag that specifies if variance is encoded in target. If flag is 0 (that is, `false`), you need to adjust the predicted offset accordingly.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *keep_top_k*

  * **Description**: *keep_top_k* is the maximum number of bounding boxes per batch to keep after NMS step. -1 means keeping all bounding boxes.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: -1
  * **Required**: *yes*

* **Parameter name**: *code_type*

  * **Description**: *code_type* is a coding method for bounding boxes.
  * **Range of values**: `"caffe.PriorBoxParameter.CENTER_SIZE"`, `"caffe.PriorBoxParameter.CORNER"`
  * **Type**: `string`
  * **Default value**: `caffe.PriorBoxParameter.CORNER`
  * **Required**: *no*

* **Parameter name**: *share_location*

  * **Description**: *share_location* is a flag that specifies if bounding boxes are shared among different classes.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *nms_threshold*

  * **Description**: *nms_threshold* is the threshold to be used in the NMS stage.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *confidence_threshold*

  * **Description**: *confidence_threshold* is a threshold to filter out detections with smaller confidence. If not set, all boxes are used.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: `-FLT_MAX`
  * **Required**: *no*

* **Parameter name**: *clip_after_nms*

  * **Description**: *clip_after_nms* is a flag that specifies whether to perform clip bounding boxes after non-maximum suppression or not.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *clip_before_nms*

  * **Description**: *clip_before_nms* is a flag that specifies whether to clip bounding boxes before non-maximum suppression or not.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *decrease_label_id*

  * **Description**: *decrease_label_id* is a flag that denotes how to perform NMS.
  * **Range of values**:
    * *0* - perform NMS like in Caffe\*
    * *1* - perform NMS like in MxNet\*
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *normalized*

  * **Description**: *normalized* is a flag that specifies whether input blobs with boxes are normalized. If blobs are not normalized, the *input_height* and *input_width* parameters are used to normalize box coordinates.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *input_height*

  * **Description**: *input_height* is the height of an input image. If the *normalized* is 1, *input_height* is ignored.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *input_width*

  * **Description**: *input_width* is the width of an input image. If the *normalized* is 1, *input_width* is ignored.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *objectness_score*

  * **Description**: *objectness_score* is the threshold to sort out confidence predictions. Used only when the *DetectionOutput* layer has five inputs.
  * **Range of values**: a non-negative floating-point number
  * **Type**: `float`
  * **Default value**: 0.0
  * **Required**: *no*

**Inputs**:

* **1**: 2D input blob with box logits. Required.
* **2**: 2D input blob with class predictions. Required.
* **3**: 3D input blob with proposals. Required.
* **4**: 2D input blob with additional class predictions information described in the [article](https://arxiv.org/pdf/1711.06897.pdf). Optional.
* **5**: 2D input blob with additional box predictions information described in the [article](https://arxiv.org/pdf/1711.06897.pdf). Optional.

**Mathematical Formulation**

At each feature map cell, *DetectionOutput* predicts the offsets relative to the default box shapes in the cell, as well as the per-class scores that indicate the presence of a class instance in each of those boxes. Specifically, for each box out of k at a given location, *DetectionOutput* computes class scores and the four offsets relative to the original default box shape. This results are a total of \f$(c + 4)k\f$ filters that are applied around each location in the feature map, yielding \f$(c + 4)kmn\f$ outputs for a *m \* n* feature map.

**Example**

```xml
<layer ... type="DetectionOutput" ... >
    <data num_classes="21" share_location="1" background_label_id="0" nms_threshold="0.450000" top_k="400" input_height="1" input_width="1" code_type="caffe.PriorBoxParameter.CENTER_SIZE" variance_encoded_in_target="0" keep_top_k="200" confidence_threshold="0.010000"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *


## Erf Layer<a name="Erf"></a>
<a href="#toc">Back to top</a>

**Name**: *Erf*

**Category**: *Layer*

**Short description**: *Erf* layer computes the Gauss error function of input element-wise.

**Detailed Description**: [Reference](https://www.tensorflow.org/api_docs/python/tf/math/erf)

**Parameters**: *Erf* layer does not have parameters.

**Inputs**:

*   **1**: Input tensor X of any floating-point type. Required.

**Outputs**:

*   **1**: Result of Erf function applied on input tensor x. Floating point tensor with shape and type matching input tensor. Required.

**Mathematical Formulation**

For each element from an input tensor, *Erf* layer calculates corresponding
element in the output tensor by the formula:
\f[
erf(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt
\f]

**Example**

```xml
<layer ... type="Erf" ... >
    <input>
        <port id="0">
            <dim>5</dim>
            <dim>4</dim>
        </port>
    </input>
    <output>
         <port id="1">
            <dim>5</dim>
            <dim>4</dim>
        </port>
    </output>
</layer>
```



* * *


## Eltwise Layer <a name="Eltwise"></a>
<a href="#toc">Back to top</a>

**Name**: *Eltwise*

**Category**: *Layer*

**Short description**: *Eltwise* layer performs element-wise operation specified in parameters, over given inputs.

**Parameters**: *Eltwise* layer parameters are specified in the `data` node, which is a child of the `layer` node. *Eltwise* accepts two inputs of arbitrary number of dimensions. The operation supports broadcasting input blobs according to the [NumPy specification](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

* **Parameter name**: *operation*

  * **Description**: *operation* is a mathematical operation to be performed over inputs.
  * **Range of values**:
    * *sum* - summation
    * *sub* - subtraction
    * *mul* - multiplication
    * *div* - division
    * *max* - maximum
    * *min* - minimum
    * *squared_diff* - squared difference
    * *floor_mod* - reminder of division
    * *pow* - power
    * *logical_and* - logical AND
    * *logical_or* - logical OR
    * *logical_xor* - logical XOR
    * *less* - less
    * *less_equal* - less or equal
    * *greater* - greater
    * *greater_equal* - greater equal
    * *equal* - equal
    * *not_equal* - not equal
  * **Type**: string
  * **Default value**: *sum*
  * **Required**: *no*

**Inputs**

* **1**: Multidimensional input blob. Required.
* **2**: Multidimensional input blob. Required.

**Mathematical Formulation**
*Eltwise* does the following with the input blobs:
\f[
o_{i} = f(b_{i}^{1}, b_{i}^{2})
\f]
where \f$b_{i}^{1}\f$ - first blob \f$i\f$-th element, \f$b_{i}^{2}\f$ - second blob \f$i\f$-th element, \f$o_{i}\f$ - output blob \f$i\f$-th element, \f$f(a, b)\f$ - is a function that performs an operation over its two arguments \f$a, b\f$.

**Example**

```xml
<layer ... type="Eltwise" ... >
    <data operation="sum"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Fill Layer <a name="Fill"></a>
<a href="#toc">Back to top</a>

**Name**: *Fill*

**Category**: *Layer*

**Short description**: *Fill* layer generates a blob of the specified shape filled with the specified value.

**Parameters**: *Fill* layer does not have parameters.

**Inputs**:

*   **1**: 1D blob with an output blob shape. Required.

*   **2**: 0D blob (constant) with the value for fill. Required.

**Example**

```xml
<layer ... type="Fill">
    <input>
        <port id="0">
            <dim>2</dim> <!-- value: [3,4]-->
        </port>
        <port id="1"/> <!-- value 5-->
    </input>
    <output>
        <port id="2">  <!-- output value is: [[5,5,5,5],[5,5,5,5],[5,5,5,5]]-->
            <dim>3</dim>
            <dim>4</dim>
        </port>
    </output>
</layer>
```

* * *

## Flatten Layer <a name="Flatten"></a>
<a href="#toc">Back to top</a>

**Name**: *Flatten*

**Category**: *Layer*

**Short description**: *Flatten* layer performs flattening of specific dimensions of the input blob.

**Parameters**: *Flatten* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* specifies the first axis to flatten.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *end_axis*

  * **Description**: *end_axis* specifies the last dimension to flatten. The value can be negative meaning counting axes from the end.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: -1
  * **Required**: *no*

**Inputs**

* **1**: Multidimensional input blob. Required.

**Example**

```xml
<layer ... type="Flatten" ...>
    <data axis="1" end_axis="-1"/>
    <input>
        <port id="0">
            <dim>7</dim>
            <dim>19</dim>
            <dim>19</dim>
            <dim>12</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>7</dim>
            <dim>4332</dim>
        </port>
    </output>
</layer>
```

* * *

## FullyConnected Layer <a name="FullyConnected"></a>
<a href="#toc">Back to top</a>

**Name**: *FullyConnected*

**Category**: *Layer*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/innerproduct.html)

**Detailed description**: [Reference](http://cs231n.github.io/convolutional-networks/#fc)

**Parameters**: *FullyConnected* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *out-size*

  * **Description**: *out-size* is the length of the output vector. For example, *out-size* equal to 4096 means that the output vector length is 4096.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: 2D or 4D input blob. Required.

**Weights Layout**

OI, which means that Input changes the fastest, then Output.

**Mathematical Formulation**

*   If previous layer is *FullyConnected*:
    \f[
    y_{i} = f( z_{i} ) \quad with \quad z_{i} = \sum_{j=1}^{m_{1}^{( l-1 )}}w_{i,j}^{( l )}y_{i}^{ ( l -1  )}
    \f]
*   Otherwise:
    \f[
    y_{i} = f( z_{i} ) \quad with \quad z_{i}^{ ( l )} = \sum_{j=1}^{m_{1}^{( l-1 )}}\sum_{r=1}^{m_{2}^{ ( l-1  )}}\sum_{s=1}^{m_{3}^{ ( l-1 )}}w_{i,j,r,s}^{ ( l )} ( Y_{i}^{ (l-1) })_{r,s}
    \f]

**Example**

```xml
<layer ... type="FullyConnected" ... >
        <data out-size="4096"/>
        <input> ... </input>
        <output> ... </output>
    </layer>
```

* * *

## Gather Layer <a name="Gather"></a>
<a href="#toc">Back to top</a>

**Name**: *Gather*

**Category**: *Layer*

**Short description**: *Gather* layer takes slices of data in the second input blob according to the indexes specified in the first input blob. The output blob shape is `input2.shape[:axis] + input1.shape + input2.shape[axis + 1:]`.

**Parameters**: *Gather* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* is a dimension index to gather data from. For example, *axis* equal to 1 means that gathering is performed over the first dimension.
  * **Range of values**: an integer in the range `[-len(input2.shape), len(input2.shape) - 1]`.
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Mathematical Formulation**

\f[
    output[:, ... ,:, i, ... , j,:, ... ,:] = input2[:, ... ,:, input1[i, ... ,j],:, ... ,:]
\f]


**Inputs**

* **1**:  Multidimensional input blob with indexes to gather. The values for indexes are in the range `[0, input1[axis] - 1]`.
* **2**:  Multidimensional input blob with arbitrary data.

**Example**

```xml
<layer id="1" name="gather_node" precision="FP32" type="Gather">
    <data axis=1 />
	<input>
		<port id="0">
			<dim>15</dim>
			<dim>4</dim>
			<dim>20</dim>
			<dim>28</dim>
		</port>
		<port id="1">
			<dim>6</dim>
			<dim>12</dim>
			<dim>10</dim>
			<dim>24</dim>
		</port>
	</input>
	<output>
		<port id="2">
			<dim>6</dim>
			<dim>15</dim>
			<dim>4</dim>
			<dim>20</dim>
			<dim>28</dim>
			<dim>10</dim>
			<dim>24</dim>
		</port>
	</output>
</layer>
```

* * *

## GRN Layer <a name="GRN"></a>
<a href="#toc">Back to top</a>

**Name**: *GRN*

**Category**: *Normalization*

**Short description**: *GRN* is the Global Response Normalization with L2 norm (across channels only).

**Parameters**: *GRN* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *bias*

  * **Description**: *bias* is added to the variance.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: 2D, 3D or 4D input blob. Required.

**Mathematical Formulation**

*GRN* computes the L2 norm by channels for input blob. *GRN* generally does the following with the input blob:
\f[
output_{i} = \frac{input_{i}}{\sqrt{\sum_{i}^{C} input_{i}}}
\f]

**Example**

```xml
<layer ... type="GRN" ... >
    <data bias="1.0"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *
## GRUCell Layer <a name="GRUCell"></a>
<a href="#toc">Back to top</a>

**Name**: *GRUCell*

**Category**: *Layer*

**Short description**: *GRUCell* layer computes the output using the formula described in the [paper](https://arxiv.org/abs/1406.1078).

**Parameters**: *GRUCell* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *activations*

  * **Description**: *activations* specifies activation functions for gates.
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh*
  * **Required**: *no*

* **Parameter name**: *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* parameters of functions
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: None
  * **Required**: *no*

* **Parameter name**: *clip*

  * **Description**: *clip* specifies bound values *[-C, C]* for tensor clipping. Clipping is performed before activations.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *no*

* **Parameter name**: *linear_before_reset*

  * **Description**: *linear_before_reset* flag denotes if the layer behaves according to the modification of *GRUCell* described in the formula in the [ONNX documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU).
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

**Inputs**

* **1**: `X` - 2D ([batch_size, input_size]) input data. Required.

* **2**: `Hi` - 2D ([batch_size, hidden_size]) input hidden state data. Required.

**Outputs**

* **1**: `Ho` - 2D ([batch_size, hidden_size]) output hidden state.

**Example**
```xml
<layer … type="GRUCell"…>
    <data hidden_size="128" linear_before_reset="1"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>16</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </input>
    <output>
        <port id="4">
            <dim>1</dim>
            <dim>128</dim>
        </port>
    </output>
    <blobs>
        <weights offset="…" size="…"/>
        <biases offset="…" size="…"/>
    </blobs>
</layer>
```

* * *
## Input Layer <a name="Input"></a>
<a href="#toc">Back to top</a>

**Name**: *Input*

**Category**: *Layer*

**Short description**: *Input* layer specifies input to the model.

**Parameters**: *Input* layer does not have parameters.

**Example**

```xml
<layer ... type="Input" ...>
    <output>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </output>
</layer>
```

* * *

## Interp Layer <a name="Interp"></a>
<a href="#toc">Back to top</a>

**Name**: *Interp*

**Category**: *Layer*

**Short description**: *Interp* layer performs bilinear interpolation of the input blob by the specified parameters.

**Parameters**: *Interp* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *height*

  * **Description**: *height* specifies output height. If  the parameter is not set, other parameters are used for output size calculation.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *no*

* **Parameter name**: *width*

  * **Description**: *width* specifies output width. If the parameter is not set, other parameters are used for output size calculation.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *no*

* **Parameter name**: *align_corners*

  * **Description**: *align_corners* is a flag that specifies whether to align corners or not.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *pad_beg*

  * **Description**: *pad_beg* specify the number of pixels to add to the beginning of the image being interpolated.
  * **Range of values**: a non-negative integer number
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *yes*

* **Parameter name**: *pad_end*

  * **Description**: *pad_end* specify the number of pixels to add to the end of the image being interpolated.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *yes*

**Inputs**

* **1**: 4D input blob. Required.

**Example**

```xml
<layer ... type="Interp" ...>
    <data align_corners="0" pad_beg="0" pad_end="0"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>2</dim>
            <dim>48</dim>
            <dim>80</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>2</dim>
            <dim>96</dim>
            <dim>160</dim>
        </port>
    </output>
</layer>
```

* * *

## LSTMCell Layer <a name="LSTMCell"></a>
<a href="#toc">Back to top</a>

**Name**: *LSTMCell*

**Category**: *Layer*

**Short description**: *LSTMCell* layer computes the output using the formula described in the original paper [Long Short-Term Memory](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.676.4320&rep=rep1&type=pdf).

**Parameters**: *LSTMCell* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *activations*

  * **Description**: *activations* specifies activation functions for gates.
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh,tanh*
  * **Required**: *no*

* **Parameter name**: *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* parameters of functions.
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: None
  * **Required**: *no*

* **Parameter name**: *clip*

  * **Description**: *clip* specifies bound values *[-C, C]* for tensor clipping. Clipping is performed before activations.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *no*

**Inputs**

* **1**: `X` - 2D ([batch_size, input_size]) input data. Required.

* **2**: `Hi` - 2D ([batch_size, hidden_size]) input hidden state data. Required.

* **3**: `Ci` - 2D ([batch_size, hidden_size]) input cell state data. Required.


**Outputs**

* **1**: `Ho` - 2D ([batch_size, hidden_size]) output hidden state.

* **2**: `Co` - 2D ([batch_size, hidden_size]) output cell state.

**Mathematical Formulation**

```
Formula:
  *  - matrix mult
 (.) - eltwise mult
 [,] - concatenation
sigm - 1/(1 + e^{-x})
tanh - (e^{2x} - 1)/(e^{2x} + 1)
   f = sigm(Wf*[Hi, X] + Bf)
   i = sigm(Wi*[Hi, X] + Bi)
   c = tanh(Wc*[Hi, X] + Bc)
   o = sigm(Wo*[Hi, X] + Bo)
  Co = f (.) Ci + i (.) c
  Ho = o (.) tanh(Co)
```

**Example**

```xml
<layer ... type="LSTMCell" ... >
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Memory Layer <a name="Memory"></a>
<a href="#toc">Back to top</a>

**Name**: *Memory*

**Category**: *Layer*

**Short description**: *Memory* layer represents the delay layer in terms of LSTM terminology. For more information about LSTM topologies, please refer to this [article](http://colah.github.io/posts/2015-08-Understanding-LSTMs).

**Detailed description**: *Memory* layer saves the state between two infer requests. In the topology, it is the single layer, however, in the Intermediate Representation, it is always represented as a pair of **Memory** layers. One of these layers does not have outputs and another does not have inputs (in terms of the Intermediate Representation).

**Parameters**: *Memory* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *id*

  * **Description**: *id* is the ID of the pair of *Memory* layers. Two layers with the same value of the *id* parameter are paired.
  * **Range of values**: any combination of Latin characters, numbers, and underscores (`_`) in the `string` format
  * **Type**: string
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *index*

  * **Description**: *index* specifies whether the given layer is input or output. For example, *index* equal to 0 means the layer is output.
  * **Range of values**:
    * 0 - current layer is output
    * 1 - current layer is input
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *size*

  * **Description**: *size* is the size of the group. For example, *size* equal to 2 means this group is a pair.
  * **Range of values**: only *2* is supported
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Mathematical Formulation**

*Memory* saves data from the input blob.

**Example**

```xml
<layer ... type="Memory" ... >
    <data id="r_27-28" index="0" size="2" />
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## MVN Layer <a name="MVN"></a>
<a href="#toc">Back to top</a>

**Name**: *MVN*

**Category**: *Normalization*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/mvn.html)

**Parameters**: *MVN* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *across_channels*

  * **Description**: *across_channels* is a flag that specifies whether mean values are shared across channels. For example, *across_channels* equal to 0 means that mean values are not shared across channels.
  * **Range of values**:
    * 0 - do not share mean values across channels
    * 1 - share mean values across channels
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *normalize_variance*

  * **Description**: *normalize_variance* is a flag that specifies whether to perform variance normalization.
  * **Range of values**:
    * 0 - do not normalize variance
    * 1 - normalize variance
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *eps*

  * **Description**: *eps* is the number to be added to the variance to avoid division by zero when normalizing the value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: 4D or 5D input blob. Required.

**Mathematical Formulation**

*MVN* subtracts mean value from the input blob:
\f[
o_{i} = i_{i} - \frac{\sum{i_{k}}}{C * H * W}
\f]
If *normalize_variance* is set to 1, the output blob is divided by variance:
\f[
o_{i}=\frac{o_{i}}{\sum \sqrt {o_{k}^2}+\epsilon}
\f]

**Example**

```xml
<layer ... type="MVN">
    <data across_channels="1" eps="9.999999717180685e-10" normalize_variance="1"/>
    <input>
        ...
    </input>
    <output>
        ...
    </output>
</layer>
```

* * *

## NonMaxSuppression Layer <a name="NonMaxSuppression"></a>
<a href="#toc">Back to top</a>

**Name**: *NonMaxSuppression*

**Category**: *Layer*

**Short description**: *NonMaxSuppression* performs non-maximum suppression of the input boxes and return indices of the selected boxes.

**Detailed description**: [Reference](https://github.com/onnx/onnx/blob/rel-1.5.0/docs/Operators.md#NonMaxSuppression)

**Parameters**: *NonMaxSuppression* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *center_point_box*

  * **Description**: *center_point_box* is flag that specifies the format of the box data. 
  * **Range of values**:
    * false (0) - the box data is supplied as `[y1, x1, y2, x2]` where `(y1, x1)` and `(y2, x2)` are the coordinates of any diagonal pair of box corners.
    * true (1) - the box data is supplied as `[x_center, y_center, width, height]`.
  * **Type**: `bool`
  * **Default value**: false
  * **Required**: *no*


**Inputs**

* **1**: 3D floating point blob with the boxes data of shape [batch_size, num_boxes, 4]. Required.
* **2**: 3D floating point blob with the boxes scores of shape [batch_size, num_classes, num_boxes]. Required.
* **3**: 1D integer blob with of shape [1] representing maximum number of boxes to be selected per class. Optional. If not specified then all boxes will be selected.
* **4**: 1D floating point blob with of shape [1] representing intersection over union threshold. Optional. If not specified then it is equal to 1.0.
* **5**: 1D floating point blob with of shape [1] representing box score threshold. Optional. If not specified then it is equal to 0.0.


**Mathematical Formulation**

\f[o_{i} = \left( 1 + \left( \frac{\alpha}{n} \right)\sum_{i}x_{i}^{2} \right)^{\beta}\f]
Where \f$n\f$ is the size of each local region.

**Example**

```xml
<layer ... type="Norm" ... >
    <data alpha="9.9999997e-05" beta="0.75" local-size="5" region="across"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Norm Layer <a name="Norm"></a>
<a href="#toc">Back to top</a>

**Name**: *Norm*

**Category**: *Normalization*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/lrn.html)

**Detailed description**: [Reference](http://yeephycho.github.io/2016/08/03/Normalizations-in-neural-networks/#Local-Response-Normalization-LRN)

**Parameters**: *Norm* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *alpha*

  * **Description**: *alpha* is a scaling parameter for the normalizing sum. For example, *alpha* equal to 0.0001 means that the normalizing sum is multiplied by 0.0001.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *beta*

  * **Description**: *beta* is an exponent for the normalizing sum. For example, *beta* equal to 0.75 means that the normalizing sum is raised to the power of 0.75.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *region*

  * **Description**: *region* is the strategy of local regions extension. For example, *region* equal to *across* means that the normalizing sum is performed over adjacent channels.
  * **Range of values**:
    * *across* - normalize sum over adjacent channels
    * *same* - normalize sum over nearby spatial locations
  * **Type**: string
  * **Default value**: `across`
  * **Required**: *yes*

* **Parameter name**: *local-size*

  * **Description**: *local-size* represents the side length of the region to be used for the normalization sum or number of channels depending on the strategy specified in the *region* parameter. For example, *local-size* equal to 5 for the *across* strategy means application of sum across 5 adjacent channels.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: 4D input blob. Required.

**Mathematical Formulation**

\f[o_{i} = \left( 1 + \left( \frac{\alpha}{n} \right)\sum_{i}x_{i}^{2} \right)^{\beta}\f]
Where \f$n\f$ is the size of each local region.

**Example**

```xml
<layer ... type="Norm" ... >
    <data alpha="9.9999997e-05" beta="0.75" local-size="5" region="across"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Normalize Layer <a name="Normalize"></a>
<a href="#toc">Back to top</a>

**Name**: *Normalize*

**Category**: *Normalization*

**Short description**: *Normalize* layer performs l-p normalization of 1 of input blob.

**Parameters**: *Normalize* layer parameters should be specified as the `data` node, which is a child of the `layer` node.

* **Parameter name**: *across_spatial*

  * **Description**: *across_spatial* is a flag that specifies if normalization is performed over CHW or HW. For example, *across_spatial* equal to 0 means that normalization is not shared across channels.
  * **Range of values**:
    * 0 - do not share normalization across channels
    * 1 - not supported
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *channel_shared*

  * **Description**: *channel_shared* is a flag that specifies if scale parameters are shared across channels. For example, *channel_shared* equal to 0 means that scale parameters are not shared across channels.
  * **Range of values**:
    * 0 -  do not share scale parameters across channels
    * 1 - not supported
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *eps*

  * **Description**: *eps* is the number to be added to the variance to avoid division by zero when normalizing the value. For example, *eps* equal to 0.001 means that 0.001 is used if all the values in normalization are equal to zero.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**

* **1**: 2D, 3D or 4D input blob. Required.

**Mathematical Formulation**

\f[
o_{i} = \sum_{i}^{H*W}\frac{\left ( n*C*H*W \right )` scale}{\sqrt{\sum_{i=0}^{C*H*W}\left ( n*C*H*W \right )^{2}}}
\f]

**Example**

```xml
<layer ... type="Normalize" ... >
    <data across_spatial="0" channel_shared="0" eps="0.000000"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## OneHot Layer <a name="OneHot"></a>
<a href="#toc">Back to top</a>

**Name**: *OneHot*

**Category**: *Layer*

**Short description**: *OneHot* layer fills the locations represented by indices specified in input with the value of *on_value* and fills all other locations with the value of *off_value*. If an index is out of range, the corresponding element is also filled with the *off_value*.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/python/tf/one_hot)

**Parameters**: *OneHot* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* is a new axis position in the output shape to fill with one-hot values.
  * **Range of values**: an integer. Negative value means counting dimension from the end.
  * **Type**: `int`
  * **Default value**: -1
  * **Required**: *no*

* **Parameter name**: *depth*

  * **Description**: *depth* is depth of a new one-hot dimension.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *on_value*

  * **Description**: *on_value* is the value that the locations represented by indices in input take.
  * **Range of values**: a floating-point number.
  * **Type**: `float`
  * **Default value**: 1.0
  * **Required**: *no*
  
* **Parameter name**: *off_value*

  * **Description**: *off_value* is the value that the locations not represented by indices in input take.
  * **Range of values**: a floating-point number.
  * **Type**: `float`
  * **Default value**: 0.0
  * **Required**: *no*

**Inputs**:

* **1**: Multidimensional input tensor with indices of type T (can be 0D). Required.

**Outputs**:

* **1** Multidimensional output tensor. If the input indices have rank N, the output will have rank N+1.
 A new axis of the size `depth` is created at the dimension `axis`.

**Examples**

```xml
<layer ... type="OneHot" ...>
    <data axis="-1" depth="3" on_value="1.0" off_value="0.0"/>
    <input>
        <port id="0">    <!-- indices value [0, 1, 2] -->
            <dim>3</dim> 
        </port>
    </input>
    <output>
        <port id="2">    <!-- output value # [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]] -->
            <dim>3</dim>
            <dim>3</dim>
        </port>
    </output>
</layer>
```

* * *

## Pad Layer <a name="Pad"></a>
<a href="#toc">Back to top</a>

**Name**: *Pad*

**Category**: *Layer*

**Short description**: *Pad* layer extends an input blob on edges. New element values are generated based on the *Pad* layer parameters described below.

**Parameters**: *Pad* layer parameters are specified in the `data` section, which is a child of the `layer` node. The parameters specify a number of elements to add along each axis and a rule by which new element values are generated: for example, whether they are filled with a given constant or generated based on the input blob content.

* **Parameter name**: *pads_begin*

  * **Description**: *pads_begin* specifies the number of padding elements at the beginning of each axis.
  * **Range of values**: a list of non-negative integers. The length of the list must be equal to the number of dimensions in the input blob.
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *pads_end*

  * **Description**: *pads_end* specifies the number of padding elements at the end of each axis.
  * **Range of values**: a list of non-negative integers. The length of the list must be equal to the number of dimensions in the input blob.
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *pad_mode*

  * **Description**: *pad_mode* specifies the method used to generate new element values.
  * **Range of values**: Name of the method in string format:
    * `constant` - padded values are equal to the value of the *pad_value* layer parameter.
    * `edge` - padded values are copied from the respective edge of the input blob.
    * `reflect` - padded values are a reflection of the input blob; values on the edges are not duplicated. `pads_begin[D]` and `pads_end[D]` must be not greater than `input.shape[D] – 1` for any valid `D`.
    * `symmetric` - padded values are symmetrically added from the input blob. This method is similar to the `reflect`, but values on edges are duplicated. Refer to the examples below for more details. `pads_begin[D]` and `pads_end[D]` must be not greater than `input.shape[D]` for any valid `D`.
  * **Type**: string
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *pad_value*

  * **Description**: Use with the `pad_mode = "constant"` only. All new elements are filled with this value.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: 0.0
  * **Required**: *no*

**Inputs**

* **1**: Multidimensional input blob. Required.


**Outputs**

* **1**: Multidimensional input blob with dimensions `pads_begin[D] + input.shape[D] + pads_end[D]` for each `D` from `0` to `len(input.shape) - 1`.


**pad_mode Examples**  

The following examples illustrate how output blob is generated for the *Pad* layer for a given input blob:
```
INPUT =
[[ 1  2  3  4 ]
[  5  6  7  8 ]
[  9 10 11 12 ]]
```
with the following parameters:
```
pads_begin = [0, 1]
pads_end = [2, 3]
```
depending on the *pad_mode*.
* `pad_mode = "constant"`:
```
OUTPUT =
[[ 0  1  2  3  4  0  0  0 ]
[  0  5  6  7  8  0  0  0 ]
[  0  9 10 11 12  0  0  0 ]
[  0  0  0  0  0  0  0  0 ]
[  0  0  0  0  0  0  0  0 ]]
```
* `pad_mode = "edge"`:
```
OUTPUT =
[[ 1  1  2  3  4  4  4  4 ]
[  5  5  6  7  8  8  8  8 ]
[  9  9 10 11 12 12 12 12 ]
[  9  9 10 11 12 12 12 12 ]
[  9  9 10 11 12 12 12 12 ]]
```
* `pad_mode = "reflect"`:
```
OUTPUT =
[[ 2  1  2  3  4  3  2  1 ]
[  6  5  6  7  8  7  6  5 ]
[ 10  9 10 11 12 11 10  9 ]
[  6  5  6  7  8  7  6  5 ]
[  2  1  2  3  4  3  2  1 ]]
```
* `pad_mode = "symmetric"`:
```
OUTPUT =
[[ 1  1  2  3  4  4  3  2 ]
[  5  5  6  7  8  8  7  6 ]
[  9  9 10 11 12 12 11 10 ]
[  9  9 10 11 12 12 11 10 ]
[  5  5  6  7  8  8  7  6 ]]
```

**Example**

```xml
<layer ... type="Pad" ...>
    <data pads_begin="0,5,2,1" pads_end="1,0,3,7" pad_mode="constant" pad_value="666.0"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>32</dim>
            <dim>40</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>2</dim>     <!-- 2 = 0 + 1 + 1 = pads_begin[0] + input.shape[0] + pads_end[0] -->
            <dim>8</dim>     <!-- 8 = 5 + 3 + 0 = pads_begin[1] + input.shape[1] + pads_end[1] -->
            <dim>37</dim>    <!-- 37 = 2 + 32 + 3 = pads_begin[2] + input.shape[2] + pads_end[2] -->
            <dim>48</dim>    <!-- 48 = 1 + 40 + 7 = pads_begin[3] + input.shape[3] + pads_end[3] -->
        </port>
    </output>
</layer>
```

* * *

## Permute Layer <a name="Permute"></a>
<a href="#toc">Back to top</a>

**Name**: *Permute*

**Category**: *Layer*

**Short description**: *Permute* layer reorders input blob dimensions.

**Detailed description**: [Reference](http://caffe.help/manual/layers/tile.html)

**Parameters**: *Permute* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *order*

  * **Description**: *order* is a list of dimensions indexes for output blob. For example, *order* equal to "0,2,3,1" means that the output blob has the following dimensions: the first dimension from the input blob, the third dimension from the input blob, the fourth dimension from the input blob, the second dimension from the input blob.
  * **Range of values**: a list of positive integers separated by comma
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*


**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Mathematical Formulation**

*Permute* layer reorders input blob dimensions. Source indexes and destination indexes are bound by the formula:
\f[
src\_ind_{offset} = n * ordered[1] * ordered[2] * ordered[3] + (h * ordered[3] + w)
\f]
\f[
n \in ( 0, order[0] )
\f]
\f[
h \in ( 0, order[2] )
\f]
\f[
w \in ( 0, order[3] )
\f]

**Example**

```xml
<layer ... type="Permute" ... >
    <data order="0,2,3,1"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Pooling Layer <a name="Pooling"></a>
<a href="#toc">Back to top</a>

**Name**: *Pooling*

**Category**: *Pool*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/pooling.html)

**Detailed description**: [Reference](http://cs231n.github.io/convolutional-networks/#pool)

**Parameters**: *Pooling* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *strides*

  * **Description**: *strides* is a distance (in pixels) to slide the window on the feature map over the `(z, y, x)` axes for 3D poolings and `(y, x)` axes for 2D poolings. For example, *strides* equal to "4,2,1" means sliding the window four pixels at a time over depth dimension, two pixels over height dimension, and one pixel over width dimension.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 1 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *pads_begin*

  * **Description**: *pads_begin* is the number of pixels to add to the beginning along each axis. For example, *pads_begin* equal to "1,2" means adding one pixel to the top of the input and two pixels to the left of the input.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 0 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *pads_end*

  * **Description**: *pads_end* is the number of pixels to add to the ending along each axis. For example, *pads_end* equal "1,2" means adding one pixel to the bottom of the input and two pixels to the right of the input.
  * **Range of values**: a list of non-negative integers
  * **Type**: `int[]`
  * **Default value**: a list of 1 with length equal to the number of convolution kernel dimensions
  * **Required**: *no*

* **Parameter name**: *kernel*

  * **Description**: *kernel* is a size of each filter. For example, *kernel* equal to "2,3" means that each filter has height equal to 2 and width equal to 3.
  * **Range of values**: a list of positive integers
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *pool-method*

  * **Description**: *pool-method* is a type of pooling strategy for values.
  * **Range of values**:
    * *max* - choose the biggest value in a feature map for each window position
    * *avg* - take the average value in a feature map for each windows position
  * **Type**: string
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *exclude-pad*

  * **Description**: *exclude-pad* is a flag that specifies whether to ignore zeros in a padding area. For example, *exclude-pad* equal to *true* means that zero values in the padding are not used.
  * **Range of values**: *true* or *false*
  * **Type**: string
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *rounding_type*

  * **Description**: *rounding_type* is a type of rounding to apply.
  * **Range of values**:
    * *ceil*
    * *floor*
  * **Type**: string
  * **Default value**: *floor*
  * **Required**: *no*

* **Parameter name**: *auto_pad*

  * **Description**: *auto_pad* specifies how to calculate padding.
  * **Range of values**:
    * Not specified: use explicit padding values
    * *same_upper/same_lower*: the input is padded to match the output size. In case of odd padding value, an extra padding is added at the end (at the beginning).
    * *valid*: do not use padding
  * **Type**: string
  * **Default value**: None
  * **Required**: *no*

**Inputs**:

*   **1**: 4D or 5D input blob. Required.

**Mathematical Formulation**

*   For `pool-method="max"`:
    \f[
    output_{j} = MAX\{ x_{0}, ... x_{i}\}
    \f]
*   For `pool-method="avg"`:
    \f[
    output_{j} = \frac{\sum_{i = 0}^{n}x_{i}}{n}
    \f]

**Example**

```xml
<layer ... type="Pooling" ... >
        <data auto_pad="same_upper" exclude-pad="true" kernel="3,3" pads_begin="0,0" pads_end="1,1" pool-method="max" strides="2,2"/>
        <input> ... </input>
        <output> ... </output>
    </layer>
```

* * *

## Power Layer <a name="Power"></a>
<a href="#toc">Back to top</a>

**Name**: *Power*

**Category**: *Layer*

**Short description**: *Power* layer computes the output as `(shift + scale * x) ^ power` for each input element `x`.

**Parameters**: *Power* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *power*

  * **Description**: *power* is a parameter in the formula described above.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *scale*

  * **Description**: *scale* is a parameter in the formula described above.
  * **Range of values**: a floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *shift*

  * **Description**: *shift* is a parameter in the formula described above.
  * **Range of values**: a floating-point number
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Mathematical Formulation**

\f[
p = (shift + scale * x)^{power}
\f]

**Example**

```xml
<layer ... type="Power" ... >
    <data power="2" scale="0.1" shift="5"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## PReLU Layer <a name="PReLU"></a>
<a href="#toc">Back to top</a>

**Name**: *PReLU*

**Category**: *Activation*

**Short description**: *PReLU* is the Parametric Rectifier Linear Unit. The difference from *ReLU* is that negative slopes can vary across channels.

**Parameters**: *PReLU* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *channel_shared*

  * **Description**: *channel_shared* specifies whether a negative slope is shared across channels or not. If the *channel_shared* is equal to 0, the slope shape is equal to the number of channels, if the *channel_shared* is equal to 1, the slope is scalar.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

**Inputs**:

*   **1**: 4D or 5D input blob. Required.


**Mathematical Formulation**

*PReLU* accepts one input with four dimensions. The produced blob has the same dimensions as input.
*PReLU* does the following with the input blob:
\f[
o_{i} = max(0, x_{i}) + w_{i} * min(0,x_{i})
\f]
where \f$w_{i}\f$ is from weights blob.

**Example**

```xml
<layer ... type="PReLU" ... >
    <data channel_shared="1"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## PriorBox Layer <a name="PriorBox"></a>
<a href="#toc">Back to top</a>

**Name**: *PriorBox*

**Category**: *Layer*

**Short description**: *PriorBox* layer generates prior boxes of specified sizes and aspect ratios across all dimensions.

**Parameters**: *PriorBox* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *min_size*

  * **Description**: *min_size* is the minimum box size (in pixels). For example, *min_size* equal to `[15.0]` means that the minimum box size is 15.0.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *max_size*

  * **Description**: *max_size* is the maximum box size (in pixels). For example, *max_size* equal to `[15.0]` means that the maximum box size is 15.0.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *aspect_ratio*

  * **Description**: *aspect_ratio* is a variance of aspect ratios. Duplicate values are ignored. For example, *aspect_ratio* equal to "[2.0,3.0]" means that for the first box, aspect ratio is 2.0, for the second box, it is 3.0.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *no*

* **Parameter name**: *flip*

  * **Description**: *flip* is a flag that specifies whether each *aspect_ratio* is duplicated and flipped. For example, *flip* equal to 1 and *aspect_ratio* equal to "4.0,2.0" mean that *aspect_ratio* is equal to "4.0,2.0,0.25,0.5".
  * **Range of values**:
    * 0 - flip each *aspect_ratio*
    * 1 - do not flip each *aspect_ratio*
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *clip*

  * **Description**: *clip* is a flag that specifies if each value in the output blob is clipped to *[0,1]* interval.
  * **Range of values**:
    * 0 - do not perform clipping
    * 1 - clip each value in the output blob to *[0,1]* interval
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *step*

  * **Description**: *step* is the distance between box centers. For example, *step* equal to `85.0` means that the distance between neighborhood prior boxes centers is 85.0.
  * **Range of values**: a non-negative floating-point number
  * **Type**: `float`
  * **Default value**: 0.0
  * **Required**: *yes*

* **Parameter name**: *offset*

  * **Description**: *offset* is a shift of box respectively to top left corner. For example, *offset* equal to `85.0` means that the shift of neighborhood prior boxes centers is 85.0.
  * **Range of values**: a non-negative floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *variance*

  * **Description**: *variance* is the variance of adjusting bounding boxes. The parameter can contain 0, 1 or 4 elements.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *scale_all_sizes*

  * **Description**: *scale_all_sizes* is a flag that specifies the type of inference. For example, *scale_all_sizes* equal to 0 means that the *PriorBox* layer is inferred in MXNet-like manner, which means that the *max_size* parameter is ignored.
  * **Range of values**:
    * 0 - do not use *max_size*
    * 1 - use *max_size*
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *fixed_ratio*

    * **Description**: *fixed_ratio* is an aspect ratio of a box. For example, *fixed_ratio* equal to 2.000000 means that the aspect ratio for the first box aspect ratio is 2.
    * **Range of values**: a list of positive floating-point numbers
    * **Type**: `float[]`
    * **Default value**: None
    * **Required**: *no*

* **Parameter name**: *fixed_size*

    * **Description**: *fixed_size* is an initial box size (in pixels). For example, *fixed_size* equal to 15 means that the initial box size is 15.
    * **Range of values**: a list of positive floating-point numbers
    * **Type**: `float[]`
    * **Default value**: None
    * **Required**: *no*

* **Parameter name**: *density*

    * **Description**: *density* is the square root of the number of boxes of each type. For example, *density* equal to 2 means that the first box generates four boxes of the same size and with the same shifted centers.
    * **Range of values**: a list of positive floating-point numbers
    * **Type**: `float[]`
    * **Default value**: None
    * **Required**: *no*

**Inputs**:

*   **1**: 4D input blob. Used to get height and width only. Required.

*   **2**: 4D input blob. Used to get image height and image width only. Required.

**Mathematical Formulation**:

*PriorBox* computes coordinates of prior boxes as follows:
1.  Calculates *center_x* and *center_y* of prior box:
    \f[
    W \equiv Width \quad Of \quad Image
    \f]
    \f[
    H \equiv Height \quad Of \quad Image
    \f]
    *   If step equals 0:
        \f[
        center_x=(w+0.5)
        \f]
        \f[
        center_y=(h+0.5)
        \f]
    *   else:
        \f[
        center_x=(w+offset)`step
        \f]
        \f[
        center_y=(h+offset)`step
        \f]
        \f[
        w \subset \left( 0, W \right )
        \f]
        \f[
        h \subset \left( 0, H \right )
        \f]
2.  For each \f$ s \subset \left( 0, min_sizes \right ) \f$, calculates coordinates of prior boxes:
    \f[
    xmin = \frac{\frac{center_x - s}{2}}{W}
    \f]
    \f[
    ymin = \frac{\frac{center_y - s}{2}}{H}
    \f]
    \f[
    xmax = \frac{\frac{center_x + s}{2}}{W}
    \f]
    \f[
    ymax = \frac{\frac{center_y + s}{2}}{H}
    \f]

**Example**

```xml
<layer ... type="PriorBox" ... >
    <data step="64.000000" min_size="162.000000" max_size="213.000000" offset="0.500000" flip="1" clip="0" aspect_ratio="2.000000,3.000000" variance="0.100000,0.100000,0.200000,0.200000" />
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## PriorBoxClustered Layer <a name="PriorBoxClustered"></a>
<a href="#toc">Back to top</a>

**Name**: *PriorBoxClustered*

**Category**: *Layer*

**Short description**: *PriorBoxClustered* layer generates prior boxes of specified sizes normalized to the input image size.

**Parameters**: *PriorBoxClustered* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *width*

  * **Description**: *width* specifies desired boxes widths in pixels.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *height*

  * **Description**: *height* specifies desired boxes heights in pixels.
  * **Range of values**: positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *clip*

  * **Description**: *clip* is a flag that specifies if each value in the output blob is clipped within *[0,1]*.
  * **Range of values**:
    * 0 - do not perform clipping
    * 1 - clip each value in the output blob to *[0,1]*
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *step (step_w, step_h)*

  * **Description**: *step (step_w, step_h)* is the distance between box centers. For example, *step* equal to 85.0 means that the distance between neighborhood prior boxes centers is 85.0. If both *step_h* and *step_w* are 0.0, they are updated with value of *step*. If after that they are still 0.0, they are calculated as input image heights/width divided by the first input heights/width.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: 0.0
  * **Required**: *yes*

* **Parameter name**: *offset*

  * **Description**: *offset* is a shift of box respectively to top left corner. For example, *offset* equal to 85.0 means that the shift of neighborhood prior boxes centers is 85.0.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *variance*

  * **Description**: *variance* is the variance of adjusting bounding boxes.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *img_h*

  * **Description**: *img_h* is the height of input image. It is calculated as the second input height unless provided explicitly.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: 1.0
  * **Required**: *yes*

* **Parameter name**: *img_w*

  * **Description**: *img_w* is the width of input image. It is calculated as second input width unless provided explicitly.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: 1.0
  * **Required**: *yes*

**Inputs**:

*   **1**: 4D input blob. Used to get height and width only. Required.

*   **2**: 4D input blob. Used to get image height and image width only. Required.

**Mathematical Formulation**

*PriorBoxClustered* computes coordinates of prior boxes as follows:
1.  Calculates the *center_x* and *center_y* of prior box:
    \f[
    W \equiv Width \quad Of \quad Image
    \f]
    \f[
    H \equiv Height \quad Of \quad Image
    \f]
    \f[
    center_x=(w+offset)`step
    \f]
    \f[
    center_y=(h+offset)`step
    \f]
    \f[
    w \subset \left( 0, W \right )
    \f]
    \f[
    h \subset \left( 0, H \right )
    \f]
2.  For each \f$s \subset \left( 0, W \right )\f$, calculates the prior boxes coordinates:
    \f[
    xmin = \frac{center_x - \frac{width_s}{2}}{W}
    \f]
	\f[
	ymin = \frac{center_y - \frac{height_s}{2}}{H}
	\f]
	\f[
	xmax = \frac{center_x - \frac{width_s}{2}}{W}
	\f]
	\f[
	ymax = \frac{center_y - \frac{height_s}{2}}{H}
	\f]
If *clip* is defined, the coordinates of prior boxes are recalculated with the formula:
\f$coordinate = \min(\max(coordinate,0), 1)\f$

**Example**

```xml
<layer ... type="PriorBoxClustered">
    <data clip="0" flip="0" height="44.0,10.0,30.0,19.0,94.0,32.0,61.0,53.0,17.0" offset="0.5" step="16.0" variance="0.1,0.1,0.2,0.2"
     width="86.0,13.0,57.0,39.0,68.0,34.0,142.0,50.0,23.0"/>
    <input>
        ...
    </input>
    <output>
        ...
    </output>
</layer>
```

* * *

## Proposal Layer <a name="Proposal"></a>
<a href="#toc">Back to top</a>

**Name**: *Proposal*

**Category**: *Layer*

**Short description**: *Proposal* layer filters bounding boxes and outputs only those with the highest prediction confidence.

**Parameters**: *Proposal* layer parameters are specified in the `data` node, which is a child of the `layer` node. The layer has three inputs: a blob with probabilities whether particular bounding box corresponds to background and foreground, a blob with logits for each of the bounding boxes, a blob with input image size in the [`image_height`, `image_width`, `scale_height_and_width`] or [`image_height`, `image_width`, `scale_height`, `scale_width`] format.

* **Parameter name**: *base_size*

  * **Description**: *base_size* is the size of the anchor to which *scale* and *ratio* parameters are applied.
  * **Range of values**: a positive integer number
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *pre_nms_topn*

  * **Description**: *pre_nms_topn* is the number of bounding boxes before the NMS operation. For example, *pre_nms_topn* equal to 15 means that the minimum box size is 15.
  * **Range of values**: a positive integer number
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *post_nms_topn*

  * **Description**: *post_nms_topn* is the number of bounding boxes after the NMS operation. For example, *post_nms_topn* equal to 15 means that the maximum box size is 15.
  * **Range of values**: a positive integer number
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *nms_thresh*

  * **Description**: *nms_thresh* is the minimum value of the proposal to be taken into consideration. For example, *nms_thresh* equal to 0.5 means that all boxes with prediction probability less than 0.5 are filtered out.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *feat_stride*

  * **Description**: *feat_stride* is the step size to slide over boxes (in pixels). For example, *feat_stride* equal to 16 means that all boxes are analyzed with the slide 16.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *min_size*

  * **Description**: *min_size* is the minimum size of box to be taken into consideration. For example, *min_size* equal 35 means that all boxes with box size less than 35 are filtered out.
  * **Range of values**: a positive integer number
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *ratio*

  * **Description**: *ratio* is the ratios for anchor generation.
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *scale*

  * **Description**: *scale* is the scales for anchor generation.
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *yes*

* **Parameter name**: *clip_before_nms*

  * **Description**: *clip_before_nms* flag that specifies whether to perform clip bounding boxes before non-maximum suppression or not.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *clip_after_nms*

  * **Description**: *clip_after_nms* is a flag that specifies whether to perform clip bounding boxes after non-maximum suppression or not.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *normalize*

  * **Description**: *normalize* is a flag that specifies whether to perform normalization of output boxes to *[0,1]* interval or not.
  * **Range of values**: 0 or 1
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *box_size_scale*

  * **Description**: *box_size_scale* specifies the scale factor applied to logits of box sizes before decoding.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: 1.0
  * **Required**: *no*

* **Parameter name**: *box_coordinate_scale*

  * **Description**: *box_coordinate_scale* specifies the scale factor applied to logits of box coordinates before decoding.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: 1.0
  * **Required**: *no*

* **Parameter name**: *framework*

  * **Description**: *framework* specifies how the box coordinates are calculated.
  * **Range of values**:
    * "" (empty string) - calculate box coordinates like in Caffe*
    * *tensorflow* - calculate box coordinates like in the TensorFlow* Object Detection API models
  * **Type**: string
  * **Default value**: "" (empty string)
  * **Required**: *no*

* **Parameter name**: *for_deformable*

  * **Description**: *for_deformable* specifies how the box coordinates are calculated.
  * **Range of values**:  0 or 1
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

**Mathematical Formulation**

*Proposal* layer accepts three inputs with four dimensions. The produced blob has two dimensions: the first one equals `batch_size * post_nms_topn`.
*Proposal* layer does the following with the input blob:
1.  Generates initial anchor boxes. Left top corner of all boxes is at (0, 0). Width and height of boxes are calculated from *base_size* with *scale* and *ratio* parameters.
2.  For each point in the first input blob:
    *   pins anchor boxes to the image according to the second input blob that contains four deltas for each box: for *x* and *y* of center, for *width* and for *height*
    *   finds out score in the first input blob
3.  Filters out boxes with size less than *min_size*
4.  Sorts all proposals (*box*, *score*) by score from highest to lowest
5.  Takes top *pre_nms_topn* proposals
6.  Calculates intersections for boxes and filter out all boxes with \f$intersection/union > nms\_thresh\f$
7.  Takes top *post_nms_topn* proposals
8.  Returns top proposals

**Inputs**:

*   **1**: 4D input blob with class prediction scores. Required.

*   **2**: 4D input blob with box logits. Required.

*   **3**: 1D input blob 3 or 4 elements: [image height, image width, scale for image height/width OR scale for image height and scale for image width]. Required.

**Example**

```xml
<layer ... type="Proposal" ... >
    <data base_size="16" feat_stride="16" min_size="16" nms_thresh="0.6" post_nms_topn="200" pre_nms_topn="6000"
     ratio="2.67" scale="4.0,6.0,9.0,16.0,24.0,32.0"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## PSROIPooling Layer <a name="PSROIPooling"></a>
<a href="#toc">Back to top</a>

**Name**: *PSROIPooling*

**Category**: *Pool*

**Short description**: *PSROIPooling* layer compute position-sensitive pooling on regions of interest specified by input.

**Detailed description**: [Reference](https://arxiv.org/pdf/1703.06211.pdf)

**Parameters**: *PSRoiPooling* layer parameters are specified in the `data` node, which is a child of the `layer` node. *PSROIPooling* layer takes two input blobs: with feature maps and with regions of interests (box coordinates). The latter is specified as five element tuples: *[batch_id, x_1, y_1, x_2, y_2]*. ROIs coordinates are specified in absolute values for the average mode and in normalized values (to *[0,1]* interval) for bilinear interpolation.

* **Parameter name**: *output_dim*

  * **Description**: *output_dim* is a pooled output channel number.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *group_size*

  * **Description**: *group_size* is the number of groups to encode position-sensitive score maps. Use for *average* mode only.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *spatial_scale*

  * **Description**: *spatial_scale* is a multiplicative spatial scale factor to translate ROI coordinates from their input scale to the scale used when pooling.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *mode*
  * **Description**: *mode* specifies mode for pooling.
  * **Range of values**:
    * *average* - perform average pooling
    * *bilinear* - perform pooling with bilinear interpolation
  * **Type**: string
  * **Default value**: *average*
  * **Required**: *yes*

* **Parameter name**: *spatial_bins_x*
  * **Description**: *spatial_bins_x* specifies numbers of bins to divide the input feature maps over width. Used for "bilinear" mode only.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *spatial_bins_y*
  * **Description**: *spatial_bins_y* specifies numbers of bins to divide the input feature maps over height. Used for *bilinear* mode only.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: 4D input blob with feature maps. Required.

*   **2**: 2D input blob describing box consisting of five element tuples: `[batch_id, x_1, y_1, x_2, y_2]`. Required.

**Example**

```xml
<layer ... type="PSROIPooling" ... >
    <data group_size="6" mode="bilinear" output_dim="360" spatial_bins_x="3" spatial_bins_y="3" spatial_scale="1"/>
    <input>                
        <port id="0">
            <dim>1</dim>
            <dim>3240</dim>
            <dim>38</dim>
            <dim>38</dim>
        </port>
        <port id="1">
            <dim>100</dim>
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>100</dim>
            <dim>360</dim>
            <dim>6</dim>
            <dim>6</dim>
        </port>
    </output>
</layer>
```

* * *

## FakeQuantize Layer <a name="FakeQuantize"></a>
<a href="#toc">Back to top</a>

**Name**: *FakeQuantize*

**Category**: *Layer*

**Short description**: *FakeQuantize* layer is element-wise linear quantization of floating-point input values into a discrete set of floating-point values.

**Detailed description**: Input and output ranges as well as the number of levels of quantization are specified by dedicated inputs and attributes. There can be different limits for each element or groups of elements (channels) of the input blobs. Otherwise, one limit applies to all elements. It depends on shape of inputs that specify limits and regular broadcasting rules applied for input blobs. The output of the operator is a floating-point number of the same type as the input blob. In general, there are four values that specify quantization for each element: *input_low*, *input_high*, *output_low*, *output_high*. *input_low* and *input_high* parameters specify the input range of quantization. All input values that are outside this range are clipped to the range before actual quantization. *output_low* and *output_high* specify minimum and maximum quantized values at the output.

**Parameters**: *Quantize* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *levels*

  * **Description**: *levels* is the number of quantization levels.
  * **Range of values**: an integer greater than or equal to 2
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: `X` - multidimensional input blob to quantize. Required.

*   **2**: `input_low` - minimum limit for input value. The shape must be broadcastable to the shape of `X`. Required.

*   **3**: `input_high` - maximum limit for input value. Can be the same as `input_low` for binarization. The shape must be broadcastable to the shape of `X`. Required.

*   **4**: `output_low` - minimum quantized value. The shape must be broadcastable to the shape of `X`. Required.

*   **5**: `output_high` - maximum quantized value. The shape must be broadcastable to the of `X`. Required.

**Mathematical Formulation**

Each element of the output is defined as the result of the following expression:

```python
if x <= input_low:
    output = output_low
elif x > input_high:
    output = output_high
else:
    # input_low < x <= input_high
    output = round((x - input_low) / (input_high - input_low) * (levels-1)) / (levels-1) * (output_high - output_low) + output_low
```

**Example**
```xml
<layer … type="FakeQuantize"…>
    <data levels="2"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>64</dim>
            <dim>56</dim>
            <dim>56</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>64</dim>
            <dim>1</dim>
            <dim>1</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>64</dim>
            <dim>1</dim>
            <dim>1</dim>
        </port>
        <port id="3">
            <dim>1</dim>
            <dim>1</dim>
            <dim>1</dim>
            <dim>1</dim>
        </port>
        <port id="4">
            <dim>1</dim>
            <dim>1</dim>
            <dim>1</dim>
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>1</dim>
            <dim>64</dim>
            <dim>56</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```

* * *

## Range Layer <a name="Range"></a>
<a href="#toc">Back to top</a>

**Name**: *Range*

**Category**: *Layer*

**Short description**: *Range* sequence of numbers according input values.

**Detailed description**: *Range* layers generates a sequence of numbers starting from the value in the first input up to but not including the value in the second input with a step equal to the value in the third input.

**Parameters**: *Range* layer does not have parameters.

**Inputs**:

*   **1**: 0D blob (constant) with the start value of the range. Required.

*   **2**: 0D blob (constant) with the limit value of the range. Required.

*   **3**: 0D blob (constant) with the step value. Required.

**Example**

```xml
<layer ... type="Range">
    <input>
        <port id="0"/> <!-- value 5-->
        <port id="1"/> <!-- value 15-->
        <port id="2"/> <!-- value 1-->
    </input>
    <output>
        <port id="3">
            <dim>10</dim> <!-- output value is: [5,6,7,8,9,10,11,12,13,14]-->
        </port>
    </output>
</layer>
```

* * *

## RegionYolo Layer <a name="RegionYolo"></a>
<a href="#toc">Back to top</a>

**Name**: *RegionYolo*

**Category**: *Layer*

**Short description**: *RegionYolo* computes the coordinates of regions with probability for each class.

**Detailed description**: [Reference](https://arxiv.org/pdf/1612.08242.pdf)

**Parameters**: *RegionYolo* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *coords*

  * **Description**: *coords* is the number of coordinates for each region.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *classes*

  * **Description**: *classes* is the number of classes for each region.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *num*

  * **Description**: *num* is the number of regions.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *do_softmax*

  * **Description**: *do_softmax* is a flag that specifies the inference method and affects how the number of regions is determined.
  * **Range of values**:
    * *0* - do not perform softmax
    * *1* - perform softmax
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

* **Parameter name**: *mask*

  * **Description**: *mask* specifies the number of regions. Use this parameter instead of *num* when *do_softmax* is equal to 0.
  * **Range of values**: a list of integers
  * **Type**: `int[]`
  * **Default value**: `[]`
  * **Required**: *no*

**Inputs**:

*   **1**: 4D input blob. Required.

**Example**

```xml
<layer ... type="RegionYolo" ... >
    <data axis="1" classes="80" coords="4" do_softmax="0" end_axis="3" mask="0,1,2" num="9"/>
    <input> ... </input>
    <output> ... </output>
    <weights .../>
</layer>
```

* * *

## ReLU Layer <a name="ReLU"></a>
<a href="#toc">Back to top</a>

**Name**: *ReLU*

**Category**: *Activation*

**Short description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/relu.html)

**Detailed description**: [Reference](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#rectified-linear-units)

**Parameters**: *ReLU* layer parameters are specified parameters in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *negative_slope*

  * **Description**: *negative_slope* is a multiplier, which is used if the unit is not active (that is, negative). For example, *negative_slope* equal to 0.1 means that an inactive unit value would be multiplied by 0.1 and this is the [Leaky ReLU](https://keras.io/layers/advanced-activations/#leakyrelu). If *negative_slope* is equal to 0.0, this is the usual *ReLU*.
  * **Range of values**: a non-negative floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *no*

**Mathematical Formulation**

\f[
Y_{i}^{( l )} = max(0, Y_{i}^{( l - 1 )})
\f]

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Example**

```xml
<layer ... type="ReLU" ... >
    <data negative_slope="0.100000"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## ReorgYolo Layer <a name="ReorgYolo"></a>
<a href="#toc">Back to top</a>

**Name**: *ReorgYolo*

**Category**: *Layer*

**Short description**: *ReorgYolo* reorganizes input blob taking into account strides.

**Detailed description**: [Reference](https://arxiv.org/pdf/1612.08242.pdf)

**Parameters**: *ReorgYolo* layer parameters are specified parameters in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *stride*

  * **Description**: *stride* is the distance between cut throws in output blobs.
  * **Range of values**: an integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: 4D input blob. Required.

**Example**

```xml
<layer ... type="ReorgYolo" ... >
    <data stride="1"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Resample (Type 1) Layer <a name="Resample_1"></a>
<a href="#toc">Back to top</a>

**Name**: *Resample*

**Category**: *Layer*

**Short description**: *Resample* layer scales the input blob by the specified parameters.

**Parameters**: *Resample* layer parameters are specified in the `data` node, which is a child of the `layer` node. *Resample* **Type 1** layer has one input blob containing image to resample.

* **Parameter name**: *type*

  * **Description**: *type* parameter specifies the type of blob interpolation.
  * **Range of values**:
    * *caffe.ResampleParameter.LINEAR* - linear blob interpolation
    * *caffe.ResampleParameter.NEAREST* - nearest-neighbor blob interpolation
  * **Type**: string
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *antialias*

  * **Description**: *antialias* is a flag that specifies whether to perform anti-aliasing.
  * **Range of values**:
    * 0 - do not perform anti-aliasing
    * 1 - perform anti-aliasing
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *factor*

  * **Description**: *factor* specifies a scale factor for output height and width.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: 4D input blob. Required.

**Example**

```xml
<layer type="Resample">
    <data antialias="0" factor="2" type="caffe.ResampleParameter.LINEAR"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>25</dim>
            <dim>30</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>3</dim>
            <dim>50</dim>
            <dim>60</dim>
        </port>
    </output>
​</layer>
```

* * *

## Resample (Type 2) Layer <a name="Resample_2"></a>
<a href="#toc">Back to top</a>

**Name**: *Resample*

**Category**: *Layer*

**Short description**: *Resample* layer scales the input blob by the specified parameters.

**Parameters**: *Resample* layer parameters are specified in the `data` node, which is a child of the `layer` node. *Resample* **Type 2** layer has two input blobs containing image to resample and output dimensions.

* **Parameter name**: *type*

  * **Description**: *type* parameter specifies the type of blob interpolation.
  * **Range of values**:
    * *caffe.ResampleParameter.LINEAR* - linear blob interpolation
    * *caffe.ResampleParameter.NEAREST* - nearest-neighbor blob interpolation
  * **Type**: string
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *antialias*

  * **Description**: *antialias* is a flag that specifies whether to perform anti-aliasing.
  * **Range of values**:
    * 0 - do not perform anti-aliasing
    * 1 - perform anti-aliasing
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *factor*

  * **Description**: *factor* parameter is ignored in the *Resample* **Type 2**.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: 4D input blob. Required.

*   **2**: 1D blob describing output shape. Required.

**Example**

```xml
<layer type="Resample">
    <data antialias="0" factor="1" type="caffe.ResampleParameter.LINEAR"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>25</dim>
            <dim>30</dim>
        </port>
        <port id="1">
            <dim>4</dim>  <!--The values in this input are [1,3,50,60] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>3</dim>
            <dim>50</dim>
            <dim>60</dim>
        </port>
    </output>
​</layer>
```

* * *

## Reshape Layer <a name="Reshape"></a>
<a href="#toc">Back to top</a>

**Name**: *Reshape*

**Category**: *Layer*

**Short description**: *Reshape* layer changes dimensions of the input blob according to the specified order. Input blob volume is equal to output blob volume, where volume is the product of dimensions.

**Detailed description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/reshape.html)

**Parameters**: *Reshape* layer does not have parameters. *Reshape* layer takes two input blobs: the blob to be resized and the output blob shape. The values in the second blob can be -1, 0 and any positive integer number. The two special values -1 and 0:
   * 0 means copying the respective dimension of the input blob.
   * -1 means that this dimension is calculated to keep the overall elements count the same as in the input blob. No more than one `-1` can be used in a reshape operation.

**Inputs**:

*   **1**: Multidimensional input blob. Required.

*   **2**: 1D blob describing output shape. Required.

**Example**

```xml
<layer ... type="Reshape" ...>
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>5</dim>
            <dim>5</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>3</dim>   <!--The blob contains 3 elements: 0, -1, 4 -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>2</dim>
            <dim>150</dim>
            <dim>4</dim>
        </port>
    </output>
</layer>
```

* * *

## ReverseSequence Layer <a name="ReverseSequence"></a>
<a href="#toc">Back to top</a>

**Name**: *ReverseSequence*

**Category**: *Layer*

**Short description**: *ReverseSequence* reverses variable length slices of data.

**Detailed description**: *ReverseSequence* slices input along the dimension specified in the *batch_axis*, and for each slice *i*, reverses the first *lengths[i]* (the second input) elements along the dimension specified in the *seq_axis*.

**Parameters**: *ReverseSequence* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *batch_axis*

  * **Description**: *batch_axis* is the index of the batch dimension.
  * **Range of values**: an integer. Can be negative.
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *no*

* **Parameter name**: *seq_axis*

  * **Description**: *seq_axis* is the index of the sequence dimension.
  * **Range of values**: an integer. Can be negative.
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: Blob with input data to reverse. Required.

*   **2**: 1D blob with sequence lengths in the first input blob. Required.

**Example**

```xml
<layer ... type="ReverseSequence">
    <data batch_axis="0" seq_axis="1"/>
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
        <port id="1">
            <dim>10</dim>
        </port>
     </input>
    <output>
        <port id="2">
            <dim>3</dim>
            <dim>10</dim>
            <dim>100</dim>
            <dim>200</dim>
        </port>
    </output>
</layer>
```

* * *
## RNNCell Layer <a name="RNNCell"></a>
<a href="#toc">Back to top</a>

**Name**: *RNNCell*

**Category**: *Layer*

**Short description**: *RNNCell* layer computes the output using the formula described in the [article](https://hackernoon.com/understanding-architecture-of-lstm-cell-from-scratch-with-code-8da40f0b71f4).

**Parameters**: *RNNCell* layer parameters should be specified as the `data` node, which is a child of the `layer` node.

* **Parameter name**: *hidden_size*

  * **Description**: *hidden_size* specifies hidden state size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *activations*

  * **Description**: activation functions for gates
  * **Range of values**: any combination of *relu*, *sigmoid*, *tanh*
  * **Type**: a list of strings
  * **Default value**: *sigmoid,tanh*
  * **Required**: *no*

* **Parameter name**: *activations_alpha, activations_beta*

  * **Description**: *activations_alpha, activations_beta* functions parameters
  * **Range of values**: a list of floating-point numbers
  * **Type**: `float[]`
  * **Default value**: None
  * **Required**: *no*

* **Parameter name**: *clip*

  * **Description**: *clip* specifies value for tensor clipping to be in *[-C, C]* before activations
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *no*

**Inputs**

* **1**: `X` - 2D ([batch_size, input_size]) input data. Required.

* **2**: `Hi` - 2D ([batch_size, hidden_size]) input hidden state data. Required.

**Outputs**

* **1**: `Ho` - 2D ([batch_size, hidden_size]) output hidden state.

* * *

## ROIPooling Layer <a name="ROIPooling"></a>
<a href="#toc">Back to top</a>

**Name**: *ROIPooling*

**Category**: *Pool*

**Short description**: *ROIPooling* is a *pooling layer* used over feature maps of non-uniform input sizes and outputs a feature map of a fixed size.

**Detailed description**: [deepsense.io reference](https://blog.deepsense.ai/region-of-interest-pooling-explained/)

**Parameters**: *ROIPooling* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *pooled_h*

  * **Description**: *pooled_h* is the height of the ROI output feature map. For example, *pooled_h* equal to 6 means that the height of the output of *ROIPooling* is 6.
  * **Range of values**: a non-negavive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *pooled_w*

  * **Description**: *pooled_w* is the width of the ROI output feature map. For example, *pooled_w* equal to 6 means that the width of the output of *ROIPooling* is 6.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *spatial_scale*

  * **Description**: *spatial_scale* is the ratio of the input feature map over the input image size.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *method*

  * **Description**: *method* specifies a method to perform pooling. If the method is *bilinear*, the input box coordinates are normalized to the [0,1] interval.
  * **Range of values**: *max* or *bilinear*
  * **Type**: string
  * **Default value**: *max*
  * **Required**: *no*

**Inputs**:

*   **1**: 4D input blob with feature maps. Required.

*   **2**: 2D input blob describing box consisting of 5 element tuples: [batch_id, x_1, y_1, x_2, y_2]. Required.

**Mathematical Formulation**

\f[
output_{j} = MAX\{ x_{0}, ... x_{i}\}
\f]

**Example**

```xml
<layer ... type="ROIPooling" ... >
        <data pooled_h="6" pooled_w="6" spatial_scale="0.062500"/>
        <input> ... </input>
        <output> ... </output>
    </layer>
```

* * *

## ExperimentalDetectronROIFeatureExtractor Layer <a name="ExperimentalDetectronROIFeatureExtractor"></a>
<a href="#toc">Back to top</a>

**Name**: *ExperimentalDetectronROIFeatureExtractor* (aka *ROIAlignPyramid*)

**Category**: *Pool*

**Short description**: *ExperimentalDetectronROIFeatureExtractor* is the *ROIAlign* operation applied over a feature pyramid.

**Detailed description**: *ExperimentalDetectronROIFeatureExtractor* maps input ROIs to the levels of the pyramid depending on the sizes of ROIs and parameters of the operation, and then extracts features via *ROIAlign* from corresponding pyramid levels.
For more details please see the math formulas below and the following sources:

  * [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)
  * [Facebook AI / detectron](https://ai.facebook.com/tools/detectron/)
  * [ONNX / ROI Align](https://github.com/onnx/onnx/blob/rel-1.5.0/docs/Operators.md#RoiAlign)
  * [NNEF / ROI Align](https://www.khronos.org/registry/NNEF/specs/1.0/nnef-1.0.2.html#roi-resize)

**Parameters**: *ExperimentalDetectronROIFeatureExtractor* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *output_size*

  * **Description**: *output_size* is the width and height of the output tensor.
  * **Range of values**: a positive integer number
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *sampling_ratio*

  * **Description**: *sampling_ratio* is the number of sampling points per the output value. If 0, then use adaptive number computed as `ceil(roi_width / output_width)`, and likewise for height.
  * **Range of values**: a non-negative integer number
  * **Type**: `int`
  * **Default value**: 0
  * **Required**: *yes*

* **Parameter name**: *pyramid_scales*

  * **Description**: *pyramid_scales* enlists `image_size / layer_size[l]` ratios for pyramid layers `l=1,...,L`, where `L` is the number of pyramid layers, and `image_size` refers to network's input image. Note that pyramid's largest layer may have smaller size than input image, e.g. `image_size` is 640 in the XML example below.
  * **Range of values**: a list of positive integer numbers
  * **Type**: `int[]`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **0**: 2D input blob describing the rois as 4-tuples: [x<sub>1</sub>, y<sub>1</sub>, x<sub>2</sub>, y<sub>2</sub>]. Batch size is the number of rois. Coordinates *x* and *y* are `float` numbers and refer to the input *image_size*. Required.

*   **1**, ..., **L**: Pyramid of 4D input blobs with feature maps. Batch size must be 1. The number of channels must be the same for all layers of the pyramid. The layer width and height must equal to the `layer_size[l] = image_size / pyramid_scales[l]`. Required.

**Outputs**:

*   **0**: 4D output blob. Batch size equals to number of rois.
Channels number is the same as for all images in the input pyramid.
Data type is `float`. Required.

**Mathematical Formulation**

*ExperimentalDetectronROIFeatureExtractor* applies the *ROIAlign* algorithm to the pyramid layers:

* output[i, :, :, :] = ROIAlign(inputPyramid[j], rois[i])
* j = PyramidLevelMapper(rois[i])

PyramidLevelMapper maps the ROI to the pyramid level using the following formula:

* j = floor(2 + log<sub>2</sub>(sqrt(w * h) / 224)

Here 224 is the "canonical" size, 2 is the pyramid starting level, and w, h are the ROI width and height.

**Example**

```xml
<layer ... type="ExperimentalDetectronROIFeatureExtractor">
	<data output_size="14"
          pyramid_scales="4,8,16,32"
          sampling_ratio="2"/>
	<input>
		<port id="0">
			<dim>100</dim>
			<dim>4</dim>
		</port>
		<port id="1">
			<dim>1</dim>
			<dim>256</dim>
			<dim>160</dim>
			<dim>160</dim>
		</port>
		<port id="2">
			<dim>1</dim>
			<dim>256</dim>
			<dim>80</dim>
			<dim>80</dim>
		</port>
		<port id="3">
			<dim>1</dim>
			<dim>256</dim>
			<dim>40</dim>
			<dim>40</dim>
		</port>
		<port id="4">
			<dim>1</dim>
			<dim>256</dim>
			<dim>20</dim>
			<dim>20</dim>
		</port>
	</input>
	<output>
		<port id="5">
			<dim>100</dim>
			<dim>256</dim>
			<dim>14</dim>
			<dim>14</dim>
		</port>
	</output>
</layer>
```

* * *

## ExperimentalSparseWeightedSum Layer <a name="ExperimentalSparseWeightedSum"></a>
<a href="#toc">Back to top</a>

**Name**: *ExperimentalSparseWeightedSum*

**Category**: *Layer*

**Short description**: *ExperimentalSparseWeightedSum* extracts embedding vectors from the parameters table for each object feature value and sum up these embedding vectors multiplied by weights for each object.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup_sparse). This is similar to *embedding_lookup_sparse* but it accepts objects with empty feature values for which it uses a default value to extract an embedding from the parameters table. In comparison with *embedding_lookup_sparse* it has a limitation to work only with two-dimensional indices tensor.

**Inputs**:

*   **1**: 2-D tensor. Input indices of the sparse tensor. It contains with an integer type. Required.
*   **2**: 1-D tensor. Input values of the sparse tensor. It contains with an integer type. Required.
*   **3**: 1-D tensor. Dense shape of the sparse tensor. It contains with an integer type. Required.
*   **4**: N-D tensor. The parameters table. It contains with a float type. Required.
*   **5**: 0-D tensor. The default value. It contains with an integer type. Required.
*   **6**: 1-D tensor. Input weights. It contains with a float type. Optional.

**Outputs**:

*   **1**: The output tensor of resulted embedding vectors for each object. It is has a shape [batch_size, params_table_shape[1], ..., params_table_shape[-1]] where batch_size is a number of objects or a number of rows in the sparse tensor.

* * *

## ScaleShift Layer <a name="ScaleShift"></a>
<a href="#toc">Back to top</a>

**Name**: *ScaleShift*

**Category**: *Layer*

**Short description**: *ScaleShift* layer performs linear transformation of the input blobs. Weights denote a scaling parameter, biases denote a shift.

**Parameters**: *ScaleShift* layer does not have parameters.

**Inputs**:

*   **1**: 4D input blob. Required.

**Mathematical Formulation**

\f[
o_{i} =\gamma b_{i} + \beta
\f]

**Example**

```
<layer ... type="ScaleShift" ... >
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Select Layer <a name="Select"></a>
<a href="#toc">Back to top</a>
**Name**: *Select*

**Category**: *Layer*

**Short description**: *Select* layer returns a tensor filled with the elements from the second or the third input, depending on the condition (the first input) value.

**Detailed description**: *Select* takes elements from the second (`then`) or the third (`else`) input based on a condition mask
 provided in the first input (`cond`). The `cond` tensor is broadcasted to `then` and `else` tensors. The output tensor shape is equal
 to the broadcasted shape of `cond`, `then`, and `else`. The behavior is similar to [numpy.where](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html) with three parameters.

**Parameters**: *Select* layer does not have parameters.

**Inputs**:
* **1**: `cond` tensor with selection mask (only integer values). The tensor can be 0D.
* **2**: `then` the tensor with elements to take where condition is true.
* **3**: `else` the tensor with elements to take where condition is false.

**Example**

```xml
<layer ... type="Select">
    <input>
        <port id="0">   <!-- cond value is: [[0, 0], [1, 0], [1, 1]]-->
            <dim>3</dim>
            <dim>2</dim>
        </port>
        <port id="1">   <!-- then value is: [[-1, 0], [1, 2], [3, 4]]-->
            <dim>3</dim>
            <dim>2</dim>
        </port>
        <port id="2">    <!-- else value is: [[11, 10], [9, 8], [7, 6]]-->
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </input>
    <output>
        <port id="1">  <!-- output value is: [[11, 10], [1, 8], [3, 4]]-->
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```

* * *

## Shape Layer <a name="Shape"></a>
<a href="#toc">Back to top</a>

**Name**: *Shape*

**Category**: *Layer*

**Short description**: *Shape* produces a blob with the input blob shape.

**Parameters**: *Shape* layer does not have parameters.

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Example**

```xml
<layer ... type="Shape">
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </input>
    <output>
        <port id="1">  <!-- output value is: [2,3,224,224]-->
            <dim>4</dim>
        </port>
    </output>
</layer>
```

* * *

## ShuffleChannels Layer <a name="ShuffleChannels"></a>
<a href="#toc">Back to top</a>

**Name**: *ShuffleChannels*

**Category**: *Layer*

**Short description**: *ShuffleChannels* permutes data in the channel dimension of the input blob.

**Parameters**: *ShuffleChannels* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* specifies the index of a channel dimension.
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *No*

* **Parameter name**: *group*

  * **Description**: *group* specifies the number of groups to split the channel dimension into. This number must evenly divide the channel dimension size.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *No*

**Inputs**:

*   **1**: 4D input blob. Required.

**Mathematical Formulation**

The operation is the equivalent with the following transformation of the input blob *x* of shape *[N, C, H, W]*:

```
x' = reshape(x, [N, group, C / group, H * W])
x'' = transpose(x', [0, 2, 1, 3])
y = reshape(x'', [N, C, H, W])
```

where *group* is the layer parameter described above.

**Example**

```xml
<layer ... type="ShuffleChannels" ...>
    <data group="3" axis="1"/>
    <input>
        <port id="0">
            <dim>3</dim>
            <dim>12</dim>
            <dim>200</dim>
            <dim>400</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>3</dim>
            <dim>12</dim>
            <dim>200</dim>
            <dim>400</dim>
        </port>
    </output>
</layer>
```

* * *

## SimplerNMS Layer <a name="SimplerNMS"></a>
<a href="#toc">Back to top</a>

**Name**: *SimplerNMS*

**Category**: *Layer*

**Short description**: *SimplerNMS* layer filters bounding boxes and outputs only those with the highest confidence of prediction.

**Parameters**: *SimplerNMS* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *pre_nms_topn*

  * **Description**: *pre_nms_topn* is the number of bounding boxes before the NMS operation. For example, *pre_nms_topn* equal to 15 means that the minimum box size is 15.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *post_nms_topn*

  * **Description**: *post_nms_topn* is the quantity of bounding boxes after the NMS operation. For example, *post_nms_topn* equal to 15 means that the maximum box size is 15.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *iou_threshold*

  * **Description**: *iou_threshold* is the minimum ratio of boxes overlapping to be taken into consideration. For example, *iou_threshold* equal to 0.7 means that all boxes with overlapping ratio less than 0.7 are filtered out.
  * **Range of values**: a positive floating-point number
  * **Type**: `float`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *feat_stride*

  * **Description**: *feat_stride* is the step size to slide over boxes (in pixels). For example, *feat_stride* equal to 16 means that all boxes are analyzed with the slide 16.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *min_bbox_size*

  * **Description**: *min_bbox_size* is the minimum size of a box to be taken into consideration.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *scale*

  * **Description**: *scale* is for generating anchor boxes.
  * **Range of values**: a list of positive floating-point numbers
  * **Type**: `float[]`
  * **Default value**: `[]`
  * **Required**: *no*

**Inputs**:

*   **1**: 4D input blob with class prediction scores. Required.

*   **2**: 4D input blob with box logits. Required.

*   **3**: 1D input blob 3 or 4 elements: [image height, image width, scale for image height/width OR scale for image height and scale for image width]. Required.

**Mathematical Formulation**

*SimplerNMS* accepts three inputs with four dimensions. Produced blob has two dimensions, the first one equals *post_nms_topn*.
*SimplerNMS* does the following with the input blob:
1.  Generates initial anchor boxes. Left top corner of all boxes is (0, 0). Width and height of boxes are calculated based on scaled (according to the scale parameter) default widths and heights
2.  For each point in the first input blob:
    *   pins anchor boxes to a picture according to the second input blob, which contains four deltas for each box: for `x` and `y` of the center, for width, and for height
    *   finds out score in the first input blob
3.  Filters out boxes with size less than *min_bbox_size.*
4.  Sorts all proposals (*box, score*) by score from highest to lowest
5.  Takes top *pre_nms_topn* proposals
6.  Calculates intersections for boxes and filters out all with \f$intersection/union > iou\_threshold\f$
7.  Takes top *post_nms_topn* proposals
8.  Returns top proposals

**Example**

```xml
<layer ... type="SimplerNMS" ... >
    <data iou_threshold="0.700000" min_bbox_size="16" feat_stride="16" pre_nms_topn="6000" post_nms_topn="150"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *

## Slice Layer <a name="Slice"></a>
<a href="#toc">Back to top</a>

**Name**: *Slice*

**Category**: *Layer*

**Short description**: *Slice* layer splits the input blob into several pieces over the specified axis.

**Parameters**: *Slice* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* specifies the axis to split the input blob along
  * **Range of values**: a non-negative integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Example**

```xml
<layer ... type="Slice" ...>
    <data axis="1"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>1048</dim>
            <dim>14</dim>
            <dim>14</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>1</dim>
            <dim>1024</dim>
            <dim>14</dim>
            <dim>14</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>24</dim>
            <dim>14</dim>
            <dim>14</dim>
        </port>
    </output>
</layer>
```

* * *

## SoftMax Layer <a name="SoftMax"></a>
<a href="#toc">Back to top</a>

**Name**: *SoftMax*

**Category**: *Activation*

**Short description**: [Reference](https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax)

**Detailed description**: [Reference](http://cs231n.github.io/linear-classify/#softmax)

**Parameters**: *SoftMax* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* is the axis along which the *SoftMax* is calculated.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: 1
  * **Required**: *no*

**Mathematical Formulation**

\f[
y_{c} = \frac{e^{Z_{c}}}{\sum_{d=1}^{C}e^{Z_{d}}}
\f]
where \f$C\f$ is a number of classes

**Example**

```xml
<layer ... type="SoftMax" ... >
    <data axis="1" />
    <input> ... </input>
    <output> ... </output>
</layer>
```

**Inputs**:

*   **1**: Multidimensional input blob. Required.

* * *

## SparseFillEmptyRows Layer <a name="SparseFillEmptyRows"></a>
<a href="#toc">Back to top</a>

**Name**: *SparseFillEmptyRows*

**Category**: *Layer*

**Short description**: *SparseFillEmptyRows* fills empty rows in the input 2-D SparseTensor with a default value.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sparse-fill-empty-rows)

**Inputs**:

*   **1**: 2-D tensor. Input indices of the sparse tensor. Required.
*   **2**: 1-D tensor. Input values of the sparse tensor. Required.
*   **3**: 1-D tensor. Shape of the sparse tensor. Value of this input is required for the Model Optimizer.
*   **4**: 0-D tensor. Default value to insert at rows missing from the input sparse tensor. Required.

**Outputs**:

*   **1**: 2-D tensor. Indices of the filled sparse tensor.
*   **2**: 1-D tensor. Values of the filled sparse tensor.
*   **3**: 1-D tensor. An indicator of whether the dense row was missing in the input sparse tensor.

* * *

## SparseSegmentMean Layer <a name="SparseSegmentMean"></a>
<a href="#toc">Back to top</a>

**Name**: *SparseSegmentMean*

**Category**: *Layer*

**Short description**: *SparseSegmentMean* computes the mean along sparse segments of a tensor.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sparse-segment-mean)

**Parameters**: *SparseSegmentMean* layer does not have parameters.

**Inputs**:

*   **1**: ND tensor. Data tensor from which rows are selected for the mean operation. Required.
*   **2**: 1D tensor. Tensor of rows indices selected from the first input tensor along 0 dimension. Required.
*   **3**: 1D tensor. Tensor of segment IDs that rows selected for the operation belong to. Rows belonging to the same segment are summed up and divided by N, where N is a number of selected rows in a segment. This input has the same size as the second input. Values must be sorted in ascending order and can be repeated. Required.

**Outputs**:

*   **1**: ND tensor. It has the same shape as the data tensor, except for dimension 0, which has a size equal to a size of an indices tensor.

* * *

## SparseSegmentSqrtN Layer <a name="SparseSegmentSqrtN"></a>
<a href="#toc">Back to top</a>

**Name**: *SparseSegmentSqrtN*

**Category**: *Layer*

**Short description**: *SparseSegmentSqrtN* computes the sum along sparse segments of a tensor and divides it by the square root of N,  where N is a number of rows in a segment.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sparse-segment-sqrt-n)

**Parameters**: *SparseSegmentSqrtN* layer does not have parameters.

**Inputs**:

*   **1**: ND tensor. Data tensor from which rows are selected. Required.
*   **2**: 1D tensor. Tensor of rows indices selected from the first input tensor along 0 dimension. Required.
*   **3**: 1D tensor. Tensor of segment IDs that selected rows belong to. Rows belonging to the same segment are summed up and divided by the square root of N, where N is a number of rows in a segment. This input tensor has the same size as the second input. Values must be sorted in ascending order and can be repeated. Required.

**Outputs**:

*   **1**: ND tensor. It has the same shape as the data tensor, except for a dimension 0, which has a size equal to a size of an indices tensor.

* * *

## SparseSegmentSum Layer <a name="SparseSegmentSum"></a>
<a href="#toc">Back to top</a>

**Name**: *SparseSegmentSum*

**Category**: *Layer*

**Short description**: *SparseSegmentSum* computes the sum along sparse segments of a tensor.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/sparse-segment-sum)

**Parameters**: *SparseSegmentSum* layer does not have parameters.

**Inputs**:

*   **1**: ND tensor. Data tensor from which rows are selected. Required.
*   **2**: 1D tensor. Tensor of rows indices selected from the first input tensor along 0 dimension. Required.
*   **3**: 1D tensor. Tensor of segment IDs that selected rows belong to. Rows belonging to the same segment are summed up. This input tensor has the same size as the second input. Values must be sorted in ascending order and can be repeated. Required.

**Outputs**:

*   **1**: ND tensor. It has the same shape as the data tensor, except for a dimension 0, which has a size equal to a size of an indices tensor.

* * *

## SparseToDense Layer <a name="SparseToDense"></a>
<a href="#toc">Back to top</a>

**Name**: *SparseToDense*

**Category**: *Layer*

**Short description**: *SparseToDense* converts a sparse tensor into a dense tensor.

**Detailed description**: [Reference](https://www.tensorflow.org/api_docs/python/tf/sparse/to_dense)

**Inputs**:

*   **1**: 2-D tensor. Input indices of the sparse tensor. It contains with an integer type. Required.
*   **2**: 1-D tensor. Dense shape of the sparse tensor. It contains with an integer type. Required.
*   **3**: 1-D tensor. Input values of the sparse tensor. It contains with integer and float types. Required.
*   **4**: 0-D tensor. Default value to insert at missing positions. The fourth input type must be the same as the third input type. If it is not specified, zero value is used. Optional.

**Outputs**:

*   **1**: The output dense tensor. The output tensor shape is equal to a value of the second input.

* * *

## Split Layer <a name="Split"></a>
<a href="#toc">Back to top</a>

**Name**: *Split*

**Category**: *Layer*

**Short description**: *Split* layer splits the input along the specified axis into several output pieces.

**Detailed description**: [Reference](http://caffe.berkeleyvision.org/tutorial/layers/split.html)

**Parameters**: *Split* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* is the number of the axis to split input blob along.
  * **Range of values**: a non-negative integer less than the number of dimensions in the input
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *num_split*

  * **Description**: *num_split* is the number of pieces to split the input into. The *num_split* must evenly divide the size of the *axis* dimension.
  * **Range of values**: a positive integer less than or equal to the size of the dimension being split over
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Mathematical Formulation**

For example, if the blob is *BxC+CxHxW*, `axis="1"`, and `num_split="2"`, the sizes of output blobs are *BxCxHxW*.

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Example**

```xml
<layer ... type="Split" ... >
    <data axis="0" num_split="2"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *


## Squeeze Layer <a name="Squeeze"></a>

**Name**: *Squeeze*

**Category**: *Layer*

**Short description**: *Squeeze* removes specified dimensions (second input) equal to 1 of the first input tensor. If the second input is omitted then all dimensions equal to 1 are removed. If the specified dimension is not equal to one then error is raised.

**Parameters**: *Squeeze* layer doesn't have parameters.

**Inputs**:

*   **1**: Multidimensional input blob. Required.

*   **2**: `(optional)`: 0D or 1D tensor with dimensions indices to squeeze. Values could be negative. Indices could be integer or float values.

**Example**

*Example 1:*
```xml
<layer type="Squeeze">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>1</dim>
            <dim>2</dim>
        </port>
    </input>
    <input>
        <port id="1">
            <dim>2</dim>  <!-- value [0, 2] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```

*Example 2: squeeze 1D tensor with 1 element to a 0D tensor (constant)* 
```xml
<layer type="Squeeze">
    <input>
        <port id="0">
            <dim>1</dim>
        </port>
    </input>
    <input>
        <port id="1">
            <dim>1</dim>  <!-- value is [0] -->
        </port>
    </input>
    <output>
        <port id="2">
        </port>
    </output>
</layer>
```


* * *

## StridedSlice Layer <a name="StridedSlice"></a>

**Name**: *StridedSlice*

**Short description**: *StridedSlice* layer extracts a strided slice of a blob.
 It is similar to generalized array indexing in Python\*.

**Parameters**: *StridedSlice* layer parameters are specified in the `data` node, which is a child of the `layer` node.

*   **Parameter name**: *begin_mask*

	* **Description**: *begin_mask* is a bit mask. *begin_mask[i]* equal to 0 means that the corresponding dimension of the `begin` input is ignored.
	* **Range of values**: a list of `0`s and `1`s
	* **Type**: `int[]`
	* **Default value**: `[1]`
	* **Required**: *yes*

*   **Parameter name**: *end_mask*

	* **Description**: *end_mask* is a bit mask. If *end_mask[i]* is 0, the corresponding dimension of the `end` input is ignored.
	* **Range of values**: a list of `0`s and `1`s
	* **Type**: `int[]`
	* **Default value**: `[1]`
	* **Required**: *yes*

*   **Parameter name**: *new_axis_mask*

	* **Description**: *new_axis_mask* is a bit mask. If *new_axis_mask[i]* is 1, a length 1 dimension is inserted on the `i`-th position of input blob.
	* **Range of values**: a list of `0`s and `1`s
	* **Type**: `int[]`
	* **Default value**: `[0]`
	* **Required**: *no*


*   **Parameter name**: *shrink_axis_mask*

	* **Description**: *shrink_axis_mask* is a bit mask. If *shrink_axis_mask[i]* is 1, the dimension on the `i`-th position is deleted.
	* **Range of values**: a list of `0`s and `1`s
	* **Type**: `int[]`
	* **Default value**: `[0]`
	* **Required**: *no*

*   **Parameter name**: *ellipsis_mask*

	* **Description**: *ellipsis_mask* is a bit mask. It inserts missing dimensions on a position of a non-zero bit.
	* **Range of values**: a list of `0`s and `1`. Only one non-zero bit is allowed.
	* **Type**: `int[]`
	* **Default value**: `[0]`
	* **Required**: *no*

**Inputs**:

*   **1**: Multidimensional input blob. Required.

*   **2**: `begin` input - 1D input blob with begin indexes for input blob slicing. Required.
           Out-of-bounds values are silently clamped. If `begin_mask[i]` is 0, the value of `begin[i]` is ignored
           and the range of the appropriate dimension starts from 0.
           Negative values mean indexing starts from the end. For example, if `foo=[1,2,3]`, `begin[0]=-1` means `begin[0]=3`.

*   **3**: `end` input - 1D input blob with end indexes for input blob slicing. Required.
           Out-of-bounds values will be silently clamped. If `end_mask[i]` is 0, the value of `end[i]` is ignored
           and the full range of the appropriate dimension is used instead.
           Negative values mean indexing starts from the end. For example, if `foo=[1,2,3]`, `end[0]=-1` means `end[0]=3`.

*   **4**: `stride` input - 1D input blob with strides. Optional.

**Example**
```xml
<layer ... type="StridedSlice" ...>
    <data begin_mask="0,1,0,0,0" ellipsis_mask="0,0,0,0,0" end_mask="0,1,0,0,0" new_axis_mask="0,0,0,0,0" shrink_axis_mask="0,1,0,0,0"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>2</dim>
            <dim>384</dim>
            <dim>640</dim>
            <dim>8</dim>
        </port>
        <port id="1">
            <dim>5</dim>
        </port>
        <port id="2">
            <dim>5</dim>
        </port>
        <port id="3">
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="4">
            <dim>1</dim>
            <dim>384</dim>
            <dim>640</dim>
            <dim>8</dim>
        </port>
    </output>
</layer>
```


* * *


## TensorIterator Layer <a name="TensorIterator"></a>
<a href="#toc">Back to top</a>

**Name**: *TensorIterator*

**Category**: *Layer*

**Short description**: *TensorIterator* (TI) layer performs recurrent sub-graph execution iterating through the data.

**Parameters**: The parameters are specified in the child nodes of the `port_map` and `back_edges` sections, which are child nodes of the layer node. The `port_map` and `back_edges` sections specify data mapping rules.

* **Node**: *port_map* is a set of rules to map input/output data blobs of the `TensorIterator` layer onto `body` data blobs. Port mapping rule is presented as `input`/`output` nodes.

	*   **Parameter name**: *external_port_id*

	    * **Description**: *external_port_id* is a port ID of the `TensorIterator` layer.
	    * **Range of values**: indexes of the *TensorIterator* outputs
	    * **Type**: `int`
	    * **Default value**: None
	    * **Required**: *yes*

	*   **Parameter name**: *internal_layer_id*

	    * **Description**: *internal_layer_id* is a layer ID inside the `body` sub-network to map to.
	    * **Range of values**: IDs of the layers inside in the *TensorIterator* layer
	    * **Type**: `int`
	    * **Default value**: None
	    * **Required**: *yes*

	*   **Parameter name**: *internal_port_id*

	    * **Description**: *internal_port_id* is a port ID of the `body` layer to map to.
	    * **Range of values**: indexes of the `body` layer input
	    * **Type**: `int`
	    * **Default value**: None
	    * **Required**: *yes*

	*   **Parameter name**: *axis*

	    * **Description**: *axis* is an axis to iterate through. `-1` means no iteration is done.
	    * **Range of values**: an integer
	    * **Type**: `int`
	    * **Default value**: -1
	    * **Required**: *no*

	*   **Parameter name**: *start*

	    * **Description**: *start* is an index where the iteration starts from. Negative value means counting indexes from the end.
	    * **Range of values**: an integer
	    * **Type**: `int`
	    * **Default value**: 0
	    * **Required**: *no*

	*   **Parameter name**: *end*

	    * **Description**: *end* is an index where iteration ends. Negative value means counting indexes from the end.
	    * **Range of values**: an integer
	    * **Type**: `int`
	    * **Default value**: -1
	    * **Required**: *no*

	*   **Parameter name**: *stride*

	    * **Description**: *stride* is a step of iteration. Negative value means backward iteration.
	    * **Range of values**: an integer
	    * **Type**: `int`
	    * **Default value**: 1
	    * **Required**: *no*

* **Node**: *back_edges* is a set of rules to transfer data blobs between `body` iteration. Mapping rule is presented as a general `edge` node with port and layer indexes of `body` sub-network.

	*   **Parameter name**: *from-layer*

	    * **Description**: *from-layer* is a layer ID inside the `body` sub-network.
	    * **Range of values**: IDs of the layers inside the *TensorIterator*
	    * **Type**: `int`
	    * **Default value**: None
	    * **Required**: *yes*

	*   **Parameter name**: *from-port*

	    * **Description**: *from-port* is a port ID inside the `body` sub-network to start mapping from.
	    * **Range of values**: the respective layer port index
	    * **Type**: `int`
	    * **Default value**: None
	    * **Required**: *yes*

	*   **Parameter name**: *to-layer*

	    * **Description**: *to-layer* is a layer ID inside the `body` sub-network to end mapping.
	    * **Range of values**: IDs of the layers inside the *TensorIterator*
	    * **Type**: `int`
	    * **Default value**: None
	    * **Required**: *yes*

	*   **Parameter name**: *to-port*

	    * **Description**: *to-port* is a port ID inside the `body` sub-network to end mapping.
	    * **Range of values**: the respective layer port index
	    * **Type**: `int`
	    * **Default value**: None
	    * **Required**: *yes*

* **Node**: *body* is a sub-network that will be recurrently executed.

    * **Parameters**: The *body* node does not have parameters.

**Example**

```xml
<layer ... type="TensorIterator" ... >
    <input> ... </input>
    <output> ... </output>
    <port_map>
        <input external_port_id="0" internal_layer_id="0" internal_port_id="0" axis="1" start="-1" end="0" stride="-1"/>
        <input external_port_id="1" internal_layer_id="1" internal_port_id="1"/>
        ...
        <output external_port_id="3" internal_layer_id="2" internal_port_id="1" axis="1" start="-1" end="0" stride="-1"/>
        ...
    </port_map>
    <back_edges>
        <edge from-layer="1" from-port="1" to-layer="1" to-port="1"/>
        ...
    </back_edges>
    <body>
        <layers> ... </layers>
        <edges> ... </edges>
    </body>
</layer>
```

* * *

## Tile Layer <a name="Tile"></a>
<a href="#toc">Back to top</a>

**Name**: *Tile*

**Category**: *Layer*

**Short description**: *Tile* layer extends input blob with copies of data along a specified axis.

**Detailed description**: [Reference](http://caffe.help/manual/layers/tile.html)

**Parameters**: *Tile* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: *axis* is the index of an axis to tile. For example, *axis* equal to 3 means that the fourth axis is used for tiling.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *tiles*

  * **Description**: *tiles* is the size of the specified axis in the output blob. For example, *tiles* equal to 88 means that the output blob gets 88 copies of data from the specified axis.
  * **Range of values**: a positive integer
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

**Mathematical Formulation**

*Tile* extends input blobs and filling in output blobs by the following rules:
\f[
out_i=input_i[inner\_dim*t]
\f]
\f[
t \in \left ( 0, \quad tiles \right )
\f]

**Inputs**:

*   **1**: Multidimensional input blob. Required.

**Example**

```xml
<layer ... type="Tile" ... >
    <data axis="3" tiles="88"/>
    <input> ... </input>
    <output> ... </output>
</layer>
```

* * *


## TopK Layer<a name="TopK"></a>
<a href="#toc">Back to top</a>

**Name**: *TopK*

**Category**: *Layer*

**Short description**: *TopK* layer computes indices and values of the *k* maximum/minimum values for each slice along the axis specified.

**Parameters**: *TopK* layer parameters are specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *axis*

  * **Description**: Specifies the axis along which to search for k maximum/minimum values.
  * **Range of values**: an integer. Negative value means counting dimension from the end.
  * **Type**: `int`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *mode*

  * **Description**: *mode* specifies an operation to use for selecting the largest element of two.
  * **Range of values**: `min`, `max`
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*

* **Parameter name**: *sort*

  * **Description**: *sort* specifies an order of output elements and/or indices.
  * **Range of values**: `value`, `index`, `none`
  * **Type**: `string`
  * **Default value**: None
  * **Required**: *yes*


**Inputs**:

*   **1**: Arbitrary tensor. Required.

*   **2**: *k* - scalar specifies how many maximum/minimum elements should be computed

**Outputs**:

*   **1**: Output tensor with top *k* values from input tensor along specified dimension *axis*. The shape of the tensor is `[input1.shape[0], ..., input1.shape[axis-1], k, input1.shape[axis+1], ...]`.

*   **2**: Output tensor with top *k* indices for each slice along *axis* dimension.
 The shape of the tensor is the same as for the 1st output, that is `[input1.shape[0], ..., input1.shape[axis-1], k, input1.shape[axis+1], ...]`

**Mathematical Formulation**

Output tensor is populated by values computes in the following way:

\f[
output[i1, ..., i(axis-1), j, i(axis+1) ..., iN] = top_k(input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]), k, sort, mode)
\f]

So for each slice `input[i1, ...., i(axis-1), :, i(axis+1), ..., iN]` which represents 1D array, top_k value is computed individually. Sorting and minimum/maximum are controlled by `sort` and `mode` attributes.

**Example**

```xml
<layer ... type="TopK" ... >
    <data axis="1" mode="max" sort="value"/>
	<input>
		<port id="0">
			<dim>6</dim>
			<dim>12</dim>
			<dim>10</dim>
			<dim>24</dim>
		</port>
		<port id="1">   <!-- value k = 3 -->
		</port>
	<output>
		<port id="2">
			<dim>6</dim>
			<dim>3</dim>
			<dim>10</dim>
			<dim>24</dim>
		</port>
	</output>
</layer>
```

* * *

## Unique Layer <a name="Unique"></a>
<a href="#toc">Back to top</a>

**Name**: *Unique*

**Category**: *Layer*

**Short description**: *Unique* finds unique elements in 1-D tensor.

**Detailed description**: [Reference](https://pytorch.org/docs/stable/torch.html?highlight=unique#torch.unique)

**Parameters**: *Unique* layer parameters should be specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *sorted*

  * **Description**: If *sorted* is equal to *true*, the unique elements in the output are sorted in ascending order. Otherwise, all of the unique elements are sorted in the same order as they occur in the input.
  * **Range of values**: *true* or *false*
  * **Type**: `string`
  * **Required**: *yes*

* **Parameter name**: *return_inverse*

  * **Description**: If *return_inverse* is equal to *true*, the layer outputs the indices. Otherwise, it does not.
  * **Range of values**: *true* or *false*
  * **Type**: `string`
  * **Required**: *yes*

* **Parameter name**: *return_counts*

  * **Description**: If *return_counts* is equal to *true*, the layer outputs the counts for each unique element. Otherwise, it does not.
  * **Range of values**: *true* or *false*
  * **Type**: `string`
  * **Required**: *yes*

**Input**:

*   **1**: 1-D tensor. Input tensor. Required.

**Outputs**:

*   **1**: 1-D tensor. Tensor of all unique elements from the input tensor. As a number of unique elements can be less than a size of the input, the end of this tensor is marked with the latest unique element. Required.
*   **2**: 1-D tensor. Tensor of indices of unique elements of the first output that can be used to reconstruct the input. The size of this tensor is equal to the input size. It outputs in the second output port. Optional.
*   **3**: 1-D tensor. Tensor of counts of occurrences of each unique element in the input. It has the same size as the output with unique elements. The end of this tensor is marked with zero. It outputs in the second output port if return_inverse is *false*, otherwise, it outputs in the third output port. Optional.

**Example**
```xml
<layer ... type="Unique" ...>
    <data sorted="false" return_inverse="true" return_counts="false"/>
    <input>
        <port id="0">
            <dim>20</dim>
        </port>
    </input>
    <output>
        <port id="0">
            <dim>20</dim>
        </port>
        <port id="1">
            <dim>20</dim>
        </port>
    </output>
</layer>
```

* * *



## Unsqueeze Layer <a name="Unsqueeze"></a>

**Name**: *Unsqueeze*

**Category**: *Layer*

**Short description**: *Unsqueeze* adds dimensions of size 1 to the first input tensor. The second input value specifies a list of dimensions that will be inserted. Indices specify dimensions in the output tensor.

**Parameters**: *Unsqueeze* layer doesn't have parameters.

**Inputs**:

*   **1**: Multidimensional input blob. Required.

*   **2**: OD or 1D tensor with dimensions indices to be set to 1. Values could be negative. Indices could be integer or float values.

**Example**

*Example 1:*
```xml
<layer type="Unsqueeze">
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>3</dim>
        </port>
    </input>
    <input>
        <port id="1">
            <dim>2</dim>  <!-- value is [0, 3] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>2</dim>
            <dim>3</dim>
            <dim>1</dim>
        </port>
    </output>
</layer>
```

*Example 2: (unsqueeze 0D tensor (constant) to 1D tensor)*
```xml
<layer type="Unsqueeze">
    <input>
        <port id="0">
        </port>
    </input>
    <input>
        <port id="1">
            <dim>1</dim>  <!-- value is [0] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
        </port>
    </output>
</layer>
```

* * *

## Unique Layer <a name="Unique"></a>
<a href="#toc">Back to top</a>

**Name**: *Unique*

**Category**: *Layer*

**Short description**: *Unique* finds unique elements in 1-D tensor.

**Detailed description**: [Reference](https://pytorch.org/docs/stable/torch.html?highlight=unique#torch.unique)

**Parameters**: *Unique* layer parameters should be specified in the `data` node, which is a child of the `layer` node.

* **Parameter name**: *sorted*

  * **Description**: If *sorted* is equal to *true*, the unique elements in the output are sorted in ascending order. Otherwise, all of the unique elements are sorted in the same order as they occur in the input.
  * **Range of values**: *true* or *false*
  * **Type**: `string`
  * **Required**: *yes*

* **Parameter name**: *return_inverse*

  * **Description**: If *return_inverse* is equal to *true*, the layer outputs the indices. Otherwise, it does not.
  * **Range of values**: *true* or *false*
  * **Type**: `string`
  * **Required**: *yes*

* **Parameter name**: *return_counts*

  * **Description**: If *return_counts* is equal to *true*, the layer outputs the counts for each unique element. Otherwise, it does not.
  * **Range of values**: *true* or *false*
  * **Type**: `string`
  * **Required**: *yes*

**Input**:

*   **1**: 1-D tensor. Input tensor. Required.

**Outputs**:

*   **1**: 1-D tensor. Tensor of all unique elements from the input tensor. As a number of unique elements can be less than a size of the input, the end of this tensor is marked with the latest unique element. Required.
*   **2**: 1-D tensor. Tensor of indices of unique elements of the first output that can be used to reconstruct the input. The size of this tensor is equal to the input size. It outputs in the second output port. Optional.
*   **3**: 1-D tensor. Tensor of counts of occurrences of each unique element in the input. It has the same size as the output with unique elements. The end of this tensor is marked with zero. It outputs in the second output port if return_inverse is *false*, otherwise, it outputs in the third output port. Optional.

**Example**
```xml
<layer ... type="Unique" ...>
    <data sorted="false" return_inverse="true" return_counts="false"/>
    <input>
        <port id="0">
            <dim>20</dim>
        </port>
    </input>
    <output>
        <port id="0">
            <dim>20</dim>
        </port>
        <port id="1">
            <dim>20</dim>
        </port>
    </output>
</layer>
```

* * *