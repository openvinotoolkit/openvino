## If <a name="If"></a> {#openvino_docs_ops_infrastructure_If_8}

**Versioned name**: *If-8*

**Category**: Infrastructure

**Short description**: *If* operation contains two internal networks(`then_body` and `else_body`) and performs one of them depending on `cond` value. If `cond` is  `True` `then_body` will be executed. `else_body` will be executed if `cond` value is `False`. 

**Detailed description**

*If* must not contain empty internal networks. Each of them must have at least one operation `Result`. Also the number outputs from *If* always must be greater than zero and equal to the number outputs from each internal networks. There are examples *If* showing its features below:

```
    //---------First example with one input---------------------------
    input(cond) // if with one input
        if(cond)
        {
            const Tensor<double> var1 = Tensor(shape=[2,3]);
            const Tensor<double>    var2 = Tensor(shape=[3]);
            result(var1, var2); // body result function
        }
        else
        {
            const Tensor<double> var3 = Tensor(shape=[2,3]);
            const Tensor<double>    var4 = Tensor(shape=[2,3]);
            result(var3 + var4, var4);
        }
        
    //---------Second example with multiple inputs--------------------
    input(cond, input1, input2) 
    // input1 - Tensor<double> with shape [2,3] if
    // input2 - Tensor<int> with shape [1]
        if(cond)
        {
            const Tensor<double> var1 = Tensor(shape=[2,3]);
            const Tensor<int>    var2 = Tensor(shape=[3]);
            result(var1 + input1, var2);
        }
        else
        {
            const Tensor<double> var3 = Tensor(shape=[1]);
            result(Stack(var3, input1), var3*input2);
        }
    //---------Third example ------------------------------------------
    input(cond, input1)
    // input1 - Tensor<int> with shape [120]
        if(cond)
        {
            return(input1)
        }
        else
        {
            const Tensor<int>  var1 = Tensor(shape=[120]);
            result(var1);
        }
    //---------Fourth example with incorrect outputs--------------------
    input(cond, input1)
    // input1 - Tensor<int> with shape [120]
        if(cond)
        {
            result(input1)
        }
        else
        {
            const Tensor<int>  var1 = Tensor(shape=[120]);
            const Tensor<int>  var2 = Tensor(shape=[150]);
            result(var1, var2);
        }
    //---------Fifth example with incorrect outputs--------------------
    input(cond)
        if(cond)
        {
            const Tensor<int>  var1 = Tensor(shape=[120]);
            return(var1);
        }
        else
        {
            const Tensor<double>  var2 = Tensor(shape=[120]);
            result(var2);
        }
     
     //Function `result` connect internal networks results with *If* outputs, for example result(var1, var2). First argument `var1` is connected(associated) with first output from *If* and second argument `var2` with second output. 
```
1. First, second and third examples show that all inputs(except the first) to *If* are optional. The internal networks can be independent of external inputs.
2. The number of outputs from *If* is undefine in fourth example, because `then_body` has only one result and `else_body` have two results. This example is incorrect and should not work.
3. The type of output from *If* is undefine in fifth example. This example is incorrect and should not work because result from `then_body` has `int` type and `else_body` has `double` type.
Note: The shape of output from *If* can be undefine(first-fourth examples).

**If attributes**:

* **Internal networks**:

    `then_body`/`else_body` is a network that will be executed depending on the `cond` value. The network is described operation by operation as a typical IR network. The internal networks have parameters (`Parameter` operations) and results (`Result` operations).
    
    * **Internal networks parameters** - inputs to the internal network which associated with *If* inputs via *portmap*. The number of parameters for the internal network can be any (even zero).
    
    * **Internal networks results** - outputs from the internal network which associated with *If* outputs via *portmap*. The internal network must contain at least one result. Each *If* output is associated with one result from the internal network. It follows that number of `then_body` results is the equal to the number of outputs from the *If* and the number of `else_body` results. Type of the internal network result and type of the associated output from *If* must be equal.
    

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
	<layer id="6" name="PartitionedCall/model/if/cond" type="If" version="opset7">
        <input>
            <port id="0"/>
            <port id="1">
                <dim>2</dim>
                <dim>4</dim>
            </port>
            <port id="2">
                <dim>2</dim>
                <dim>4</dim>
            </port>
            <port id="3">
                <dim>2</dim>
                <dim>4</dim>
            </port>
        </input>
        <output>
            <port id="4" names="PartitionedCall/model/if/cond/Identity:0,PartitionedCall/model/if/cond:0" precision="FP32">
                <dim>2</dim>
                <dim>4</dim>
            </port>
        </output>
        <then_port_map>
            <input external_port_id="1" internal_layer_id="0"/>
            <input external_port_id="2" internal_layer_id="1"/>
            <output external_port_id="0" internal_layer_id="3"/>
        </then_port_map>
        <else_port_map>
            <input external_port_id="1" internal_layer_id="0"/>
            <input external_port_id="3" internal_layer_id="1"/>
            <output external_port_id="0" internal_layer_id="3"/>
        </else_port_map>
        <then_body>
            <layers>
                <layer id="0" name="add_x" type="Parameter" version="opset1">
                    <data element_type="f32" shape="2,4"/>
                    <output>
                        <port id="0" names="add_x:0" precision="FP32">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer id="1" name="add_z" type="Parameter" version="opset1">
                    <data element_type="f32" shape="2,4"/>
                    <output>
                        <port id="0" names="add_z:0" precision="FP32">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="Add" type="Add" version="opset1">
                    <data auto_broadcast="numpy"/>
                    <input>
                        <port id="0">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                        <port id="1">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" names="Add:0" precision="FP32">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" name="Identity/sink_port_0" type="Result" version="opset1">
                    <input>
                        <port id="0">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </input>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
                <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
            </edges>
        </then_body>
        <else_body>
            <layers>
                <layer id="0" name="add_x" type="Parameter" version="opset1">
                    <data element_type="f32" shape="2,4"/>
                    <output>
                        <port id="0" names="add_x:0" precision="FP32">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer id="1" name="add_w" type="Parameter" version="opset1">
                    <data element_type="f32" shape="2,4"/>
                    <output>
                        <port id="0" names="add_w:0" precision="FP32">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="Add" type="Add" version="opset1">
                    <data auto_broadcast="numpy"/>
                    <input>
                        <port id="0">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                        <port id="1">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" names="Add:0" precision="FP32">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" name="Identity/sink_port_0" type="Result" version="opset1">
                    <input>
                        <port id="0">
                            <dim>2</dim>
                            <dim>4</dim>
                        </port>
                    </input>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
                <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
            </edges>
        </else_body>
    </layer>
```
