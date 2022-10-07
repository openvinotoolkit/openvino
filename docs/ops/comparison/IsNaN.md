# IsNaN {#openvino_docs_ops_comparison_IsNaN_10}

**Versioned name**: *IsNaN-10*

**Category**: *Comparison*

**Short description**: *IsNaN* returns the boolean mask of a given tensor which maps `NaN` to `True`.

**Detailed description**: *IsNaN* returns the bolean mask of the input tensor in which `True` corresponds to `NaN` and `False` to other values.
* The output tensor has the same shape as input tensor.
* The `i`'th element of the output tensor is `True` if  `i`'th element of the input tensor is `NaN`. Otherwise it is `False`.
* For example, for given input tensor `[NaN, 2.1, 3.7, NaN]` the output tensor is `[True, False, False, True]`.

**Inputs**:

*   **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

*   **1**: A boolean tensor of the same shape as the input tensor.

**Types**

* *T*: `bfloat16`, `double`, `float`, `float16`.

**Example**

```xml
<layer ... type="IsNaN">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
     </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
