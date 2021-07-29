## Assign <a name="Assign"></a> {#openvino_docs_ops_infrastructure_Assign_3}

**Versioned name**: *Assign-3*

**Category**: *Infrastructure*

**Short description**: *Assign* sets an input value to the `variable_id` variable.

**Detailed description**:

*Assign* operation sets an input value to the `variable_id` variable. This value will be returned by *ReadValue* operation on next infer if variable was not reset.
The operation checks that the type and shape of the input are the same as
declared in `variable_id` and returns an error otherwise.

**Attributes**:

* *variable_id*

  * **Description**: identificator of the variable to be updated
  * **Range of values**: any non-empty string
  * **Type**: string
  * **Required**: *yes*

**Inputs**

*   **1**: `new_value` - input tensor of any supported type. **Required.**

**Outputs**

*   **1**: tensor with the same shape and type as `new_value`.

**Example**

```xml
<layer ... type="Assign" ...>
    <data variable_id="lstm_state_1"/>
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
