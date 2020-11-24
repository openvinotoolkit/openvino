## GatherElements <a name="GatherElements"></a> {#openvino_docs_ops_movement_GatherElements_6}

**Versioned name**: *GatherElements-6*

**Category**: Data movement operations

**Short description**: *GatherElements* takes elements from input `data` tensor at positions specified in `indices`.

**Detailed description** *GatherElements* takes elements from `data` tensor at positions specified in `indices`. `data` 
and `indices` tensors have the same rank `r >= 1`. Optional attribute `axis` (by default `axis` is 0) determines 
along which axis elements with indices specified in `indices` are taken. `indices` tensor has the same shape as `data` 
except for `axis` dimension. Output consists of values (gathered from the `data`) for each element in `indices` 
and has the same shape as `indices`. Operation can be expressed through GatherND-5 from opset but in this case execution 
will be slower. (in this case additional memory will be used and additional memory access operations also will slow 
down execution).

For instance, in the 3D case (`r = 3`), the output is determined by the following equations:
```
  out[i][j][k] = data[indices[i][j][k]][j][k] if axis = 0
  out[i][j][k] = data[i][indices[i][j][k]][k] if axis = 1
  out[i][j][k] = data[i][j][indices[i][j][k]] if axis = 2
```
Example 1:
```
  data = [
      [1, 2],
      [3, 4],
  ]
  indices = [
      [0, 1],
      [0, 0],
  ]
  axis = 0
  output = [
      [
        [1, 4],
        [1, 2],
      ],
  ]
```
Example 2 with axis = 1 and `indices` having greater (than `data`) shape:
```
data = [
      [1, 7],
      [4, 3],
  ]
  indices = [
      [1, 1, 0],
      [1, 0, 1],
  ]
  axis = 1
  output = [
      [
        [7, 7, 1],
        [3, 4, 3],
      ],
  ]
```

Example 3 `indices` has lesser (than `data`) shape:
```
data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
  ]
  indices = [
      [1, 0, 1],
      [1, 2, 0],
  ]
  axis = 0
  output = [
      [
        [4, 2, 6],
        [4, 8, 3],
      ],
  ]
```
**Attributes**:
* *axis* 
  * **Description**: Which axis to gather on. Negative value means counting dimensions from the back. 
  * **Range of values**: `[-r, r-1]` where `r = rank(data)`.
  * **Type**: int
  * **Default value**: 0
  * **Required**: *no*


**Inputs**:

* **1**:  `data` tensor of type *T*. This is a tensor of a `rank >= 1`. Required.

* **2**:  `indices` tensor of type *T_IND* with the same rank as the input. All index values are expected to be within
 bounds `[0, s-1]`, where `s` is size along `axis` dimension of `data` tensor. It is an error if any of the index 
 values are out of bounds. Required.

**Outputs**:

*   **1**: Tensor with gathered values of type *T*.

**Types**
      
* *T*: any supported type.

* *T_IND*: `int32` or `int64`.

**Example**

```xml
<... type="GatherElements" ...>
    <data axis=1 />
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>2</dim>
        </port>
        <port id="1">
            <dim>2</dim>
            <dim>3</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>2</dim>
            <dim>3</dim>
        </port>
    </output>
</layer>
```
