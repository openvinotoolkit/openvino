## Gather <a name="Gather"></a> {#openvino_docs_ops_movement_Gather_7}

**Versioned name**: *Gather-7*

**Category**: Data movement operations

**Short description**: *Gather* operation takes slices of data of the first input tensor according to the indices
 specified with the second input tensor and axis from the third input. Semantics of this operation is identical to 
TensorFlow\* [Gather](https://www.tensorflow.org/api_docs/python/tf/gather) operation.

**Detailed description**

    output[p_0, p_1, ..., p_{axis-1}, p_axis, ..., p_{axis + k}, ...] = 
       data[p_0, p_1, ..., p_{axis-1}, indices[p_0, p_1, ..., p_{b-1}, p_b, ..., p_{axis}, j], ...]

Where `data`, `indices` and `axis` are tensors from first, second and third inputs correspondingly, and `b` is 
the number of batch dimensions.

**Attributes**:
* *batch_dims*
  * **Description**: *batch_dims* (also denoted as `b`) is a leading number of dimensions of `data` tensor and `indices` 
  representing the batches, and *Gather* starts to gather from the `b` dimension. It requires the first `b` 
  dimensions in `data` and `indices` tensors to be equal.
  * **Range of values**: `[0; min(data.rank, indices.rank))` and `batch_dims <= axis`
  * **Type**: *T_AXIS*
  * **Default value**: 0
  * **Required**: *no*

Example 1 with default *batch_dims* value:
```
batch_dims = 0
axis = 0

indices = [0, 0, 4] 
data    = [1, 2, 3, 4, 5]
output  = [1, 1, 5]
```

Example 2 with non-default *batch_dims* value:
```
batch_dims = 1
axis = 1

indices = [[0, 0, 4], <-- this is applied to the first batch 
           [4, 0, 0]]  <-- this is applied to the second batch
indices_shape = (2, 3)

data    = [[1, 2, 3, 4, 5],  <-- the first batch
           [6, 7, 8, 9, 10]]  <-- the second batch 
data_shape = (2, 5)

output  = [[ 1, 1, 5],
           [10, 6, 6]]
output_shape = (2, 3)
```

Example 3 with non-default *batch_dims* value:
```
batch_dims = 2
axis = 2

indices = [[[0, 0, 4],  <-- this is applied to the first batch, index = (0, 0)
            [4, 0, 0]],  <-- this is applied to the second batch, index = (0, 1)
          
           [[1, 2, 4],  <-- this is applied to the third batch, index = (1, 0)
            [4, 3, 2]]]  <-- this is applied to the fourth batch, index = (1, 1) 
indices_shape = (2, 2, 3)

data    = [[[1, 2, 3, 4, 5],  <-- the first batch, index = (0, 0)
            [6, 7, 8, 9, 10]],  <-- the second batch, index = (0, 1)
          
           [[11, 12, 13, 14, 15],  <-- the third batch, index = (1, 0)
            [16, 17, 18, 19, 20]]]  <-- the fourth batch, index = (1, 1)
data_shape = (2, 2, 5)

output  = [[[ 1, 1, 5],
            [10, 6, 6]],

           [[12, 13, 15],
            [20, 19, 18]]] 
output_shape = (2, 2, 3)
```
Example 4 with *axis* > *batch_dims*:
```
batch_dims = 1
axis = 2

indices = [[1, 2, 4],  <-- this is applied to the first batch 
           [4, 3, 2]]  <-- this is applied to the second batch
indices_shape = (2, 3)

data = [[[[ 1,  2,  3,  4], <-- first batch
          [ 5,  6,  7,  8],
          [ 9, 10, 11, 12],
          [13, 14, 15, 16],
          [17, 18, 19, 20]]],
  
        [[[21, 22, 23, 24], <-- second batch
          [25, 26, 27, 28],
          [29, 30, 31, 32],
          [33, 34, 35, 36],
          [37, 38, 39, 40]]]]
data_shape = (2, 1, 5, 4)

output = [[[[ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [17, 18, 19, 20]]],

          [[[37, 38, 39, 40],
            [33, 34, 35, 36],
            [29, 30, 31, 32]]]]
output_shape = (2, 1, 3, 4)
```

**Inputs**

* **1**:  `data` tensor of type *T* with arbitrary data. **Required**.

* **2**:  `indices` tensor of type *T_IND* with indices to gather. The values for indices are in the range `[0, data[axis] - 1]`. 
**Required**.

* **3**:  Scalar or 1D tensor `axis` of *T_AXIS* type is a dimension index to gather data from. For example, 
*axis* equal to 1 means that gathering is performed over the first dimension. Negative value means reverse indexing. 
Allowed values are from `[-len(data.shape), len(indices.shape) - 1]` and `axis >= batch_dims`. 
**Required**.

**Outputs**

* **1**: The resulting tensor of type *T* that consists of elements from `data` tensor gathered by `indices`. The shape 
of the output tensor is `data.shape[:axis] + indices.shape[batch_dims:] + data.shape[axis + 1:]`

**Types**

* *T*: any supported type.

* *T_IND*: `int32` or `int64`.

* *T_AXIS*: `int32` or `int64`.

**Example**

```xml
<layer ... type="Gather" version="opset7">
    <data batch_dims="1" />
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>64</dim>
            <dim>128</dim>
        </port>
        <port id="1">
            <dim>2</dim>
            <dim>32</dim>
            <dim>21</dim>
        </port>
        <port id="2"/>   <!--  axis = 1  -->
    </input>
    <output>
        <port id="2">
            <dim>2</dim>
            <dim>32</dim>
            <dim>21</dim>
            <dim>128</dim>
        </port>
    </output>
</layer>
```
