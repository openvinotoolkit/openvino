## Split <a name="Split"></a>

**Versioned name**: *Split-1*

**Category**: *Data movement operations*

**Short description**: *Split* operation splits an input tensor into pieces of the same length along some axis.

**Attributes**

* *num_splits*

  * **Description**: it specifies the number of outputs into which the initial "*data*" tensor will be split along *"axis"*
  * **Range of values**: a positive integer less than or equal to the size of the dimension being split over
  * **Type**: any integer type
  * **Default value**: None
  * **Required**: *Yes*

**Inputs**

* **1**: "data" - A tensor of type T1. **Required.**

* **2**: "axis" - axis along *"data"* to split. A scalar of type T2 with value from range `-rank(data) .. rank(data)-1`. Negative values address dimensions from the end. **Required.**

**Outputs**

* **Multiple outputs**: Tensors of the same type as the 1st input tensor. The shape of the i-th output has the same shape as the *"data"* except along dimension *"axis"* where the size is `data.shape[i]/num_splits`.

**Detailed Description**

*Split* operation splits the *"data"* input tensor into pieces of the same length along *"axis"*. The i-th shape of output tensor will be equal to the *"data"* shape except along dimension *"axis"* where the shape will be `data.shape[i]/num_splits`. The sum of elements of split_lengths must match `data.shape[axis]`.

Shape of output tensor will be:
\f[
shape_output_tensor = shape_input_tensor[shape_input_tensor[0], shape_input_tensor[1], ... ,split_lengths[axis], ... shape_input_tensor[D-1]], where D rank of input tensor.
\f]


**Types**

* *T1*: arbitrary supported type.
* *T2*: any integer type.

**Example**

```xml
<layer id="1" type="Split" ...>
    <data num_splits="3" />
    <input>
        <port id="0">       <!-- some data -->
            <dim>6</dim>
            <dim>12</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="1">       <!-- axis: 1 -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>6</dim>
            <dim>4</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="3">
            <dim>6</dim>
            <dim>4</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
        <port id="4">
            <dim>6</dim>
            <dim>4</dim>
            <dim>10</dim>
            <dim>24</dim>
        </port>
    </output>
</layer>
```