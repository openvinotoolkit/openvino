## BatchToSpace <a name="BatchToSpace"></a> {#openvino_docs_ops_movement_BatchToSpace_2}

**Versioned name**: *BatchToSpace-2*

**Category**: *Data movement*

**Short description**: The *BatchToSpace* operation reshapes the "batch" dimension 0 into N - 1 dimensions of shape `block_shape` + [batch] and interleaves these blocks back into the grid defined by the spatial dimensions `[1, ..., N - 1]` to obtain a result with the same rank as `data` input. The spatial dimensions of this intermediate result are then optionally cropped according to `crops_begin` and `crops_end` to produce the output. This is the reverse of the *SpaceToBatch* operation.

**Detailed description**:

The *BatchToSpace* operation is similar to the TensorFlow* operation [BatchToSpaceND](https://www.tensorflow.org/api_docs/python/tf/batch_to_space_nd)

The operation is equivalent to the following transformation of the input tensors `data` with shape `[batch, D_1, D_2 ... D_{N-1}]` and `block_shape`, `crops_begin`, `crops_end` of shape `[N]` to *Y* output tensor.

    note: B_0 is expected to be 1.
    x' = reshape(`data`, [B_1, ..., B_{N - 1}, batch / (B_1 * ... B_{N - 1}), D_1, D_2, ..., D_{N - 1}]), where B_i = block_shape[i]

    x'' = transpose(x', [N, N + 1, 0, N + 2, 1, ..., N + N - 1, N - 1])

    x''' = reshape(x'', [batch / (B_1 * ... * B_{N - 1}), D_1 * B_1, D_2 * B_2, ... , D_{N - 1} * B_{N - 1}])

   Crop the start and end of dimensions according to `crops_begin`, `crops_end` to produce the output of shape:
   note: `crops_begin[0], crops_end[0]` are expected to be 0.
    `y = [batch / (B_1 * ... * B_{N - 1}), crop(D_1 * B_1, crops_begin[1], crops_end[1]), crop(D_2 * B_2, crops_begin[2], crops_end[2]), ... , crop(D_{N - 1} * B_{N - 1}, crops_begin[N - 1], crops_end[N - 1])]`

**Attributes**

    No attributes available.

**Inputs**

*   **1**: `data` - input N-D tensor `[batch, D_1, D_2 ... D_{N-1}]` of *T1* type with rank >= 2. **Required.**
*   **2**: `block_shape` - input 1-D tensor of *T2* type with shape `[N]` that is equal to the size of `data` input shape. All values must be >= 1.`block_shape[0]` is expected to be 1. **Required.** 
*   **3**: `crops_begin` - input 1-D tensor of *T2* type with shape `[N]` that is equal to the size of `data` input shape. All values must be non-negative. crops_begin specifies the amount to crop from the beginning along each axis of `data` input . It is required that `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i]`. `crops_begin[0]` is expected to be 0. **Required.**
*   **4**: `crops_end` - input 1-D tensor of *T2* type with shape `[N]` that is equal to the size of `data` input shape. All values must be non-negative. crops_end specifies the amount to crop from the ending along each axis of `data` input. It is required that `crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i]`. `crops_end[0]` is expected to be 0. **Required.**

**Outputs**

*   **1**: N-D tensor with shape `[batch / (block_shape[0] * block_shape[1] * ... * block_shape[N - 1]), D_1 * block_shape[1] - crops_begin[1] - crops_end[1], D_2 * block_shape[2] - crops_begin[2] - crops_end[2], ..., D_{N - 1} * block_shape[N - 1] - crops_begin[N - 1] - crops_end[N - 1]` of the same type as `data` input. 

**Types**

* *T1*: any supported type.
* *T2*: any supported integer type.

**Example**

```xml
<layer type="BatchToSpace" ...>
    <input>
        <port id="0">       <!-- data -->
            <dim>48</dim>   <!-- batch -->
            <dim>3</dim>    <!-- spatial dimension 1 -->   
            <dim>3</dim>    <!-- spatial dimension 2 -->
            <dim>1</dim>    <!-- spatial dimension 3 -->
            <dim>3</dim>    <!-- spatial dimension 4 -->
        </port>
        <port id="1">       <!-- block_shape value: [1, 2, 4, 3, 1] -->
            <dim>5</dim>  
        </port>
        <port id="2">       <!-- crops_begin value: [0, 0, 1, 0, 0] -->
            <dim>5</dim>
        </port>
        <port id="3">       <!-- crops_end value: [0, 0, 1, 0, 0] -->
            <dim>5</dim>
        </port>
    </input>
    <output>
        <port id="3">      
            <dim>2</dim>    <!-- data.shape[0] / (block_shape.shape[0] * block_shape.shape[1] * ... * block_shape.shape[4]) -->
            <dim>6</dim>    <!-- data.shape[1] * block_shape.shape[1] - crops_begin[1] - crops_end[1]-->
            <dim>10</dim>   <!-- data.shape[2] * block_shape.shape[2] - crops_begin[2] - crops_end[2] -->
            <dim>3</dim>    <!-- data.shape[3] * block_shape.shape[3] - crops_begin[3] - crops_end[3] -->
            <dim>3</dim>    <!-- data.shape[4] * block_shape.shape[4] - crops_begin[4] - crops_end[4] -->
        </port>
    </output>
</layer>
```
