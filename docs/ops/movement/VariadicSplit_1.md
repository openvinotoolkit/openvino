## VariadicSplit <a name="VariadicSplit"></a> {#openvino_docs_ops_movement_VariadicSplit_1}

**Versioned name**: *VariadicSplit-1*

**Category**: *Data movement operations*

**Short description**: *VariadicSplit* operation splits an input tensor into pieces along some axis. The pieces may have variadic lengths depending on *"split_lengths*" attribute.

**Attributes**

No attributes available.

**Inputs**

* **1**: `data` - A tensor of type T1. **Required.**

* **2**: `axis` - An axis along `data` to split. A scalar of type T2 with value from range `-rank(data) .. rank(data)-1`. Negative values address dimensions from the end. 
**Required.**

* **3**: `split_lengths` - A list containing the sizes of each output tensor along the split `axis`. Size of `split_lengths` should be equal to the number of outputs. The sum of sizes must match `data.shape[axis]`. A 1-D Tensor of type T2. `split_lenghts` can contain a single `-1` element, that means all remaining items along specified `axis` that are not consumed by other parts. **Required.**

**Outputs**

* **Multiple outputs**: Tensors of the same type as the `data` tensor. The shape of the i-th output has the same shape as the `data` except along dimension `axis` where the size is `split_lengths[i]` if `split_lengths[i] != -1`. `-1` item, if exists, is processed as described in the `split_lengths` input description.

**Detailed Description**

*VariadicSplit* operation splits the `data` input tensor into pieces along `axis`. The i-th shape of output tensor will be equal to the `data` shape except along dimension `axis` where the size will be `split_lengths[i]`. The sum of elements of split_lengths must match `data.shape[axis]`.

Shape of output tensor will be:
\f[
shape_output_tensor = shape_input_tensor[shape_input_tensor[0], shape_input_tensor[1], ..., split_lengths[axis], ..., shape_input_tensor[D-1]], where D rank of input tensor.
\f]

**Types**

* *T1*: arbitrary supported type.
* *T2*: any integer type.

**Examples**

```xml
<layer id="1" type="VariadicSplit" ...>
    <input>
        <port id="0">            <!-- some data -->
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">            <!-- axis: 0 -->
        </port>
        <port id="2">
            <dim>3</dim>         <!-- split_lengths: [1, 2, 3] -->
        </port>
    </input>
    <output>
        <port id="3">
            <dim>1</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="4">
            <dim>2</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="5">
            <dim>3</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```

```xml
<layer id="1" type="VariadicSplit" ...>
    <input>
        <port id="0">            <!-- some data -->
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">            <!-- axis: 0 -->
        </port>
        <port id="2">
            <dim>2</dim>         <!-- split_lengths: [-1, 2] -->
        </port>
    </input>
    <output>
        <port id="3">
            <dim>4</dim>         <!--  4 = 6 - 2  -->
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="4">
            <dim>2</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```