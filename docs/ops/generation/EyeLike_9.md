## EyeLike <a name="EyeLike"></a> {#openvino_docs_ops_generation_EyeLike_9}

**Versioned name**: *EyeLike-8*

**Category**: *Generation*

**Short description**: *EyeLike* operation generates identity matrices.

**Detailed description**:

*EyeLike* operation generates an identity matrix or a batch matrices with ones on the diagonal and zeros everywhere else. Index of the diagonal to be populated with ones is given by `diagonal_index`: `output[i, i + diagonal_index] = 1`.


Example 1. *EyeLike* output with `output_type` = f32:

``` 
num_rows = 3

output  = [[1. 0. 0.]
           [0. 1. 0.]
           [0. 0. 1.]]
```

Example 2. *EyeLike* output with `output_type` = i32, `diagonal_index` = 2:

``` 
num_rows = 3

num_columns = 4

output  = [[0 0 1 0]
           [0 0 0 1]
           [0 0 0 0]]
```

Example 3. *EyeLike* output with `output_type` = f16, `diagonal_index` = 5:

``` 
num_rows = 2

output  = [[0. 0.]
           [0. 0.]]
```

**Attributes**:

* *output_type*

    * **Description**: the type of the output
    * **Range of values**: any numeric type
    * **Type**: string
    * **Required**: *Yes*

* *diagonal_index*

    * **Description**: index of the diagonal to be populated. A positive value refers to an upper diagonal and a negative value refers to a lower diagonal. Value `0` populates the main diagonal.
    * **Range of values**: any integer value
    * **Type**: int
    * **Default value**: *0*
    * **Required**: *No*

**Inputs**:

*   **1**: `num_rows` - scalar or 1D tensor with 1 non-negative element of type *T* describing the number of rows in each matrix. **Required.**

*   **2**: `num_columns` - scalar or 1D tensor with 1 non-negative element of type *T* describing the number of rows in each matrix. Optionally, with default value equals to `num_rows`.

*   **3**: `batch_shape` - 1D tensor with non-negative values of type *T* defines leading batch dimensions of output shape. 
If `batch_shape` is an empty list, *EyeLike* operation generates a 2D tensor (matrix). Optionally, default is an empty tensor.


**Outputs**:

* **1**: A tensor with type specified by the attribute *output_type*. The shape is `batch_shape + [num_rows, num_columns]`

**Types**

* *T*: `int32` or `int64`.

**Examples**

*Example 1*

```xml
<layer ... name="EyeLike" type="EyeLike">
    <data output_type="i32"/>
    <input>
        <port id="0" precision="I32"/>  <!-- num rows -->
    </input>
    <output>
        <port id="3" precision="I32" names="EyeLike:0">
            <dim>-1</dim>
            <dim>-1</dim>
        </port>
    </output>
</layer>
```

*Example 2*

```xml
<layer ... name="EyeLike" type="EyeLike">
    <data output_type="f32"/>
    <input>
        <port id="0" precision="I32"/>  <!-- num rows -->
        <port id="2" precision="I32">   <!-- batch shape value -->
            <dim>2</dim>
        </port>
    </input>
    <output>
        <port id="3" precision="FP32" names="EyeLike:0">
            <dim>-1</dim>
            <dim>-1</dim>
            <dim>-1</dim>
            <dim>-1</dim>
        </port>
    </output>
</layer>
```

*Example 3*

```xml
<layer ... name="EyeLike" type="EyeLike">
    <data output_type="i64" diagonal_index="2"/>
    <input>
        <port id="0" precision="I32"/>  <!-- num rows -->
        <port id="1" precision="I32"/>  <!-- num columns -->
        <port id="2" precision="I32">   <!-- batch shape value -->
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="3" precision="I64" names="EyeLike:0">
            <dim>-1</dim>
            <dim>-1</dim>
            <dim>-1</dim>
        </port>
    </output>
</layer>
```
