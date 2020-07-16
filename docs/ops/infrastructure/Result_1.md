## Result <a name="Result"></a> {#openvino_docs_ops_infrastructure_Result_1}

**Versioned name**: *Result-1*

**Category**: *Infrastructure*

**Short description**: *Result* layer specifies output of the model.

**Attributes**: 

    No attributes available.

**Inputs**

* **1**: A tensor of type T. **Required.**

**Types**

* *T*: arbitrary supported type.

**Example**

```xml
<layer ... type="Result" ...>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>3</dim>
            <dim>224</dim>
            <dim>224</dim>
        </port>
    </input>
</layer>
```