## If <a name="If"></a> {#openvino_docs_ops_infrastructure_If_8}

**Versioned name**: *If-8*

**Category**: Infrastructure

**Short description**: *If* operation, depending on `cond` value, performs the internal network, which is described in the `then_body` (if `cond` is `True`) or `else_body` (if `cond` is `False`). 

**If attributes**:

**Detailed description**:

* **Internal networks**:

    `then_body`/`else_body` is a network that will be executed depending on the `cond` value. The network is described operation by operation as a typical IR network. The internal networks have parameters (`Parameter` operations) and results (`Result` operations).
    
    * **Internal networks parameters** - inputs() to the internal network which associated with *If* inputs via *portmap*. The number of parameters for the internal network can be any (even zero).
    
    * **Internal networks results** - outputs from the internal network which associated with *If* outputs via *portmap*. The internal network must contain at least one result. Each *If* output is associated with one result from the internal network. It follows that number of `then_body` results is the equal to the number of outputs from the *If* and the number of `else_body` results. Type of the internal network result and type of the associated output from *If* must be equal.
    
    * **Internal networks attributes**:

            No attributes available.

* **Port maps**:
    
    *port_map* is a set of rules to map input or output data tensors of *If* operation onto the internal network data tensors. The `port_map` entries can be `input` and `output`. Each entry describes a corresponding mapping rule. *If* has two *port_maps* - `then_port_map` for `then_body` and `else_port_map` for `else_body`.

    * **Port map attributes**:

        * *external_port_id*
            * **Description**: *external_port_id* is a port ID of *If* operation.
            * **Range of values**: IDs of the *If* inputs and outputs
            * **Type**: `unsigned int`
            * **Default value**: None
            * **Required**: *yes*

        * *internal_layer_id*

            * **Description**: *internal_layer_id* is a `Parameter` or `Result` operation ID inside the internal network to map to.
            * **Range of values**: IDs of the `Parameter` or `Result` operations inside in the internal network 
            * **Type**: `unsigned int`
            * **Default value**: None
            * **Required**: *yes*

**If Inputs**


* **cond**: A scalar or 1D tensor with 1 element of `boolean` type specifying which an internal network  to execute. `True` value means to execute the `then_body`, `False` - `else_body`. *Required*.

* **Multiple other inputs**: tensors of different types and shapes. *Optional*.

**If Outputs**

* **Multiple outputs**: Results of execution of one of internal networks. Tensors of any type and shape. *Required*.


**Body Inputs**

* **Multiple inputs**: tensors of different types and shapes. *Optional*.


**Body Outputs**

* **Multiple outputs**: Results of execution of the internal network. Tensors of any type and shape.  *Required*


**Examples**

*Example 1: a typical If structure*
```xml
<layer type="If" version="opset8" ...>
    <input> ... </input>
    <output> ... </output>
    <then_port_map>
        <input external_port_id="1" internal_layer_id="0"/>
        <input external_port_id="2" internal_layer_id="1"/>
        <input external_port_id="3" internal_layer_id="2"/>
        ...
        <output external_port_id="0" internal_layer_id="55"/>
        <output external_port_id="1" internal_layer_id="56"/>
        <output external_port_id="2" internal_layer_id="57"/>
        ...
    </then_port_map>
    <else_port_map>
        <output external_port_id="0" internal_layer_id="123"/>
        <output external_port_id="1" internal_layer_id="59"/>
        <output external_port_id="2" internal_layer_id="73"/>
        ...
    </else_port_map>
    <then_body>
        <layers> ... </layers>
        <edges> ... </edges>
    </then_body>
    <else_body>
        <layers> ... </layers>
        <edges> ... </edges>
    </else_body>
</layer>
```
