## Transpose<a name="Transpose"></a>

**Versioned name**: *Transpose-1*

**Category**: Data movement

**Short description**: *Transpose* operation reorders the input tensor dimensions.

**Attributes**:

No attributes available.

**Inputs**:

* **1**: "arg" - the tensor to be transposed. A tensor of type T1. **Required.**
* **2**: "input_order" - the permutation to apply to the axes of the input shape. Must be a vector of element T2 type, with shape [n], where n is the rank of "arg". The tensor's value must contain every integer in the range [0,n-1]. If an empty list is specified [] then the axes will be inverted. A tensor of type T2. **Required.**

**Outputs**:

*   **1**: A tensor with shape and type matching 1st tensor.

**Types**

* *T1*: arbitrary supported type.
* *T2*: any integer type.

**Detailed description**:

*Transpose* operation reorders the input tensor dimensions. Source indexes and destination indexes are bound by the formula: 
\f[
    output[i(order[0]), i(order[1]), ..., i(order[N-1])] = input[i(0), i(1), ..., i(N-1)], where i(j) in range 0..(input.shape[j]-1).
\f]

**Examples**

*Example 1*

```xml
<layer ... type="Transpose">
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>3</dim>
            <dim>4</dim>
        </port>
        <port id="1">
            <dim>3</dim>  <!-- [2, 0, 1] -->
        </port>
    </input>
    <output>
        <port id="2">
            <dim>4</dim>
            <dim>2</dim>
            <dim>3</dim>
        </port>
    </output>
</layer>
```

*Example 2: input_order in not specified*

```xml
<layer ... type="Transpose">
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>3</dim>
            <dim>4</dim>
        </port>
    </input>
    <output>         <!-- input_order = [2, 1, 0] if input_order is not set -->
        <port id="1">
            <dim>4</dim>
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```

*Example 3: input_order = empty_list []*

```xml
<layer ... type="Transpose">
    <input>
        <port id="0">
            <dim>2</dim>
            <dim>3</dim>
            <dim>4</dim>
        </port>
        <port id="1">
            <dim>0</dim> <!-- input_order = [2, 1, 0] if input_order is empty list -->
        </port>
    </input>
    <output>         
        <port id="2">
            <dim>4</dim>
            <dim>3</dim>
            <dim>2</dim>
        </port>
    </output>
</layer>
```