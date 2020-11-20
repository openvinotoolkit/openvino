## GatherElements <a name="GatherElements"></a> {#openvino_docs_ops_movement_GatherElements_6} 

**Versioned name**: *GatherElements-6*

**Category**: Data movement operations

**Short description**: *GatherElements* is an indexing operation that produces its output by indexing the input data 
tensor at index positions determined by elements of the indices tensor.

**Detailed description**
*GatherElements* takes two inputs `data` and `indices` of the same rank `r >= 1` and an optional attribute `axis` that 
identifies an axis of data (by default, the outer-most axis, that is axis 0). Its output shape is the same as the 
shape of `indices` and consists of one value (gathered from the data) for each element in indices.

For instance, in the 3D case (`r = 3`), the output is determined by the following equations:
```
  out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0
  out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1
  out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2
```

This implies that only one axis of indices tensor (specified in `axis` attribute) can have greater size than corresponding axis of a data tensor.

Example 1 with concrete values:
```
  data = [
      [1, 2],
      [3, 4],
  ]
  indices = [
      [0, 0],
      [1, 0],
  ]
  axis = 1
  output = [
      [
        [1, 1],
        [4, 3],
      ],
  ]
```
Example 2 with concrete values:
```
data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
  ]
  indices = [
      [1, 2, 0],
      [2, 0, 0],
  ]
  axis = 0
  output = [
      [
        [4, 8, 3],
        [7, 2, 3],
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
 bounds `[-s, s-1]`, where s is size along `axis` dimension. It is an error if any of the index values are out of bounds.
Required.

**Outputs**:

*   **1**: Tensor with gathered values of type *T*.

**Types**
      
* *T*: any supported type.

* *T_IND*: any supported integer types.

**Example**

```xml
<layer id="1" type="GatherElement">
    <data axis=0 />
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>10</dim>
        </port>
        <port id="1">
            <dim>10</dim>
            <dim>10</dim>
        </port>
    </input>
    <output>
        <port id="3">
            <dim>10</dim>
            <dim>10</dim>
        </port>
    </output>
</layer>
```
