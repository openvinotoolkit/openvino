## Gather <a name="Gather"></a> {#openvino_docs_ops_movement_Gather_7}

**Versioned name**: *Gather-7*

**Category**: Data movement operations

**Short description**: *Gather* operation takes slices of data in the first input tensor according to the indices
 specified in the second input tensor and axis from the third input.

**Detailed description**

    output[p_0, p_1, ..., p_{axis-1}, i, ..., j, ...] = 
    data[p_0, p_1, ..., p_{axis-1}, indices[p_0, p_1, ..., p_{b-1}, i, ..., j], ...]

Where `data`, `indices` and `axis` are tensors from first, second and third inputs correspondingly, and `b` is 
batch dimension.

**Attributes**:
* *batch_dims*
  * **Description**: *batch_dims* (denoted as `b`) is a leading number of dimensions of `data` tensor and `indices` 
  representing the batches, and *Gather* starts to gather from the `b` dimension. It requires the first `b` 
  dimensions in `data` and `indices` tensors to be equal. In case non default value for *batch_dims* the output shape 
  is calculated as `output.shape = params.shape[:axis] + indices.shape[batch_dims:] + data.shape[axis + 1:]`.
  * **Range of values**: integer number and belongs to `[0; min(data.rank, indices.rank))`
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*

Example 1 shows how *Gather* operates when *batch_dims* is specified (equals to 0):
```
indices = [0, 0, 4] 
data    = [1,  2,  3,  4,  5]
output  = [ 1,  1,  5]
```

Example 2 shows how *Gather* operates with non-default *batch_dims* value:
```
batch_dims = 1
indices = [[0, 0, 4], <-- this is applied to the first batch 
           [4, 0, 0]]  <-- this is applied to the second batch
indices_shape = (2, 3)

data    = [[ 1,  2,  3,  4,  5],  <-- the first batch
           [ 6,  7,  8,  9, 10]]  <-- the second batch 
data_shape = (2, 5)

output  = [[ 1,  1,  5],
           [10,  6,  6]]
output_shape = (2, 3)
```

Example 3 shows how *Gather* operates with non-default *batch_dims* value:
```
batch_dims = 2
indices = [[[0, 0, 4],  <-- this is applied to the first batch, index = (0, 0)
            [4, 0, 0]],  <-- this is applied to the second batch, index = (0, 1)
          
           [[1, 2, 4],  <-- this is applied to the third batch, index = (1, 0)
            [4, 3, 2]]]  <-- this is applied to the fourth batch, index = (1, 1) 
indices_shape = (2, 2, 3)

data    = [[[1,  2,  3,  4,  5],  <-- the first batch, index = (0, 0)
            [6,  7,  8,  9, 10]],  <-- the second batch, index = (0, 1)
          
           [[11,  12,  13,  14,  15],  <-- the third batch, index = (1, 0)
            [16,  17,  18,  19, 20]]]  <-- the fourth batch, index = (1, 1)
data_shape = (2, 2, 5)

output  = [[[ 1,  1,  5],
            [10,  6,  6]],

           [[12, 13, 15],
            [20, 19, 18]]] 
output_shape = (2, 2, 3)
```

**Inputs**

* **1**:  Tensor with arbitrary data. Required.

* **2**:  Tensor with indices to gather. The values for indices are in the range `[0, data[axis] - 1]`. Required.

* **3**:  Scalar or 1D tensor *axis* is a dimension index to gather data from. For example, *axis* equal to 1 means 
that gathering is performed over the first dimension. Negative value means reverse indexing. Allowed values are from 
`[-len(data.shape), len(indices.shape) - 1]`. Required.

**Outputs**

* **1**: The resulting tensor that consists of elements from the first input tensor gathered by indices from the
 second input tensor. Shape of the tensor is `data.shape[:axis] + indices.shape[batch_dims:] + data.shape[axis + 1:]`

**Example**

```xml
<layer id="1" type="Gather">
    <data batch_dims="2" />
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>2</dim>
            <dim>5</dim>
        </port>
        <port id="1">
            <dim>2</dim>
            <dim>2</dim>
            <dim>3</dim>
        </port>
        <port id="2"/>   <!--  axis = 0  -->
    </input>
    <output>
        <port id="2">
            <dim>2</dim>
            <dim>2</dim>
            <dim>3</dim>
        </port>
    </output>
</layer>
```