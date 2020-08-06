## Reshape <a name="Reshape"></a> {#openvino_docs_ops_shape_Reshape_1}

**Versioned name**: *Reshape-1*

**Category**: Shape manipulation

**Short description**: *Reshape* operation changes dimensions of the input tensor according to the specified order. Input tensor volume is equal to output tensor volume, where volume is the product of dimensions.

**Detailed description**:

*Reshape* layer takes two input tensors: the tensor to be resized and the output tensor shape. The values in the second tensor could be -1, 0 and any positive integer number. The two special values -1 and 0:
   * `0` means "copy the respective dimension of the input tensor" if `special_zero` is set to `true`; otherwise it is a normal dimension and is applicable to empty tensors.
   * `-1` means that this dimension is calculated to keep the overall elements count the same as in the input tensor. Not more than one `-1` can be used in a reshape operation.

**Attributes**:

* *special_zero*

  * **Description**: *special_zero* controls how zero values in `shape` are interpreted. If *special_zero* is `false`, then 0 is interpreted as-is which means that output shape will contain a zero dimension at the specified location. Input and output tensors are empty in this case. If *special_zero* is `true`, then all zeros in `shape` implies the copying of corresponding dimensions from `data.shape` into the output shape.
  * **Range of values**: `false` or `true`
  * **Type**: boolean
  * **Default value**: None
  * **Required**: *yes*

**Inputs**:

*   **1**: `data` -- multidimensional input tensor. Required.

*   **2**: `shape` -- 1D tensor describing output shape. Required.

**Outputs**:

*   **1**: Output tensor with the same content as a tensor at input `data` but with shape defined by input `shape`.

**Examples**

```xml
<layer ... type="Reshape" ...>
    <data special_zero="false"/>
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>5</dim>
            <dim>5</dim>
            <dim>0</dim>
        </port>
        <port id="1">
            <dim>2</dim>   <!--The tensor contains 3 elements: 0, 4 -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>0</dim>
            <dim>4</dim>
        </port>
    </output>
</layer>
```

```xml
<layer ... type="Reshape" ...>
    <data special_zero="true"/>
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>5</dim>
            <dim>5</dim>
            <dim>24</dim>
        </port>
        <port id="1">
            <dim>3</dim>   <!--The tensor contains 3 elements: 0, -1, 4 -->
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

