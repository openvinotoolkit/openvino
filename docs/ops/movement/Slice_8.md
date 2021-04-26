## Slice <a name="Slice"></a> {#openvino_docs_ops_movement_Slice_8}

**Versioned name**: *Slice-8*

**Category**: *Data movement operations*

**Short description**: *Slice* operation extracts a slice of the input tensor.

**Detailed Description**

*Slice* operation selects a region of values from the `data` tensor. 
Selected values start at indexes  provided in the `start` and end 
at indexes provides in `stop` (exclusively).

Optional `step` input allows subsampling of `data`, selecting evey *n*-th element, 
where `n=step`.

Optional `axes` input allows specifying slice indexes only on selected axes. 
Other axes will not be affected and will be output in full.

**Inputs**

* **1**: `data`. A tensor of type `T` and arbitrary shape. **Required.**
  
* **2**: `start`. 1D tensor (type `T_INT`) - indices corresponding to axes in `data`. 
  Defines the starting coordinate of the slice in the `data` tensor.
  A negative index value represents counting elements from the end of that dimension. 
  A value larger than the size of a dimension is silently clamped. 
  **Required.**

* **3**: `stop`. 1D, type `T_INT`, similar to `start`.
  Defines the coordinate of the opposite vertex of the slice, or where the slice ends.
  Stop indexes are exclusive, which means values lying on the ending edge are
  not included in the output slice.
  In order to slice to the end of a dimension of unknown size `INT_MAX`
  may be used (or `INT_MIN` if slicing backwards).
  **Required.**

* **4**: `step`. 1D tensor of type `T_INT` and the same shape as `start` and `stop`.
  Integer value which specifies the increment between each index used in slicing.
  Value cannot be `0`, negative value indicates slicing backwards.
  Optional. Default value: [1, 1...]

* **5**: `axes`. 1D tensor of type `T_INT`.
  Optional 1D tensor indicating to which dimensions, the values in `start` and `stop` apply.
  A negative value means counting dimensions from the highest rank.
  Optional. Default value: [0, 1, 2...]


**Outputs**:

*   **1**: Tensor  of type *T* with values of the selected slice.

**Types**

* *T*: any arbitrary supported type.
* *T_INT*: any supported integer type.

**Example**

```xml
<layer id="1" type="Slice" ...>
    <input>
        <port id="0">       <!-- data -->
            <dim>20</dim>
            <dim>10</dim>
            <dim>5</dim>
        </port>
        <port id="1">       <!-- start: 0, 0 -->
          <dim>2</dim>
        </port>
        <port id="2">       <!-- stop: 4, 10 -->
          <dim>2</dim>
        </port>
        <port id="3">       <!-- step: 1, 1 -->
          <dim>2</dim>
        </port>
        <port id="4">       <!-- axes: 0, 1 -->
          <dim>2</dim>
        </port>
    </input>
    <output>
        <port id="5">
            <dim>4</dim>
            <dim>10</dim>
            <dim>5</dim>
        </port>
    </output>
</layer>
```