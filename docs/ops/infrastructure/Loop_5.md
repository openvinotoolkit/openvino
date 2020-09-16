## Loop <a name="Loop"></a> {#openvino_docs_ops_infrastructure_Loop_5}

**Versioned name**: *Loop-5*

**Category**: Infrastructure

**Short description**: *Loop* operation performs recurrent execution of the network, which is described in the `body`, iterating through the data. 
The operation has similar semantic to the ONNX* Loop [operation](https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Loop-13).

**Detailed description**

The body of the Loop can be executed 0 or more times depending on the values passed to the Loop operation inputs called "trip count" and "termination condition".

1. Trip count input is an integer scalar input specifying maximum number of iterations. //Default value is "-1" meaning infinite number of iterations.
2. Loop termination condition input is a boolean scalar input specifying whether to run the current loop iteration or not.
Note, that the body of the Loop must yield the termination condition value whether the input is provided or not. 

Both inputs are optional.
//Default value is a "True" constant meaning need to perform first iteration.

There are several combinations of these two inputs which are described in the following code snippets (empty string `""` means no input provided):

```
  input ("", ""):
      for (int i = 0; ; ++i) 
      {
          cond = ...; // the "cond" value must be yield by the body, but is ignored in the loop condition check
      }

  input ("", cond) // while loop
      bool cond = ...;
      for (int i = 0; cond; ++i) 
      {
          cond = ...;
      }

  input ("", 1) // do-while loop
      bool cond = true;
      for (int i = 0; cond; ++i) 
      {
          cond = ...;
      }

  input (trip_count, "") // for loop
      int trip_count = ...;
      for (int i = 0; i < trip_count; ++i) 
      {
          cond = ...; // the "cond" value must be yield by the body, but is ignored in the loop condition check
      }

  input (trip_count, cond)
      int trip_count = ...;
      bool cond = ...;
      for (int i = 0; i < trip_count && cond; ++i) 
      {
          cond = ...;
      }
```

The body graph has two inputs: 
1. The "current iteration" number.
2. The "termination condition". This value is provided from the corresponding input of the Loop operation for the first iteration and calculated in the body graph for the consequent iterations.

Loop operation description in the IR has regular sections: `input` and `output`. They connect Loop body to the outer graph and specify termination condition(s).
Loop operation description in the IR also has several special sections: `body`, `port_map` and `back_edges` similar to the ones from the TensorIterator operation but having some important features described below.

1. The body operation getting an input from the main graph should have an entry in the `port_map` section of the Loop operation. These edges connect input ports of the Loop with the body `Parameter`s.
1. The body operation producing tensor to be used in the subsequent iterations (like in RNN models) should have a back edge described in the `back_edges` section of the operation. The back edge connects the respective body `Parameter` and `Result` operations. For such a case the Loop operation node provides input for the first iteration, while corresponding Loop operation output produces the tensor computed during the last iteration.
1. Output tensors produced by a particular body operation across all iterations can be concatenated and returned as a Loop operation output. The corresponding `output` entry in the `port_map` should have `axis` attribute specifying the axis to concatenate. Therefore, outputs from operations corresponding to `output` entries in the `port_map` without `axis` attribute are returned "as is" (without concatenation).
1. There is one body `Parameter` operation not connected through the `port_map`. This is a "current iteration" input. The Loop operation is responsible for providing the appropriate value for each iteration.
1. Connection of nodes inside the Loop with the main graph should be done through `Parameter` and `Result` body operations. No other ways to connect graphs are allowed.

**Loop attributes**:

* **Body**:

    `body` is a network that will be recurrently executed. The network is described operation by operation as a typical IR network.

    * **Body attributes**:

            No attributes available.

* **Port map**:

    *port_map* is a set of rules to map input or output data tensors of  `Loop` operation onto `body` data tensors. The `port_map` entries can be` input` and `output`. Each entry describes a corresponding mapping rule.

    * **Port map attributes**:

        * *external_port_id*
            * **Description**: *external_port_id* is a port ID of the `Loop` operation.
            * **Range of values**: indexes of the *Loop* outputs
            * **Type**: `int`
            * **Default value**: None
            * **Required**: *yes*

        * *internal_operation_id*

            * **Description**: *internal_operation_id* is a `Parameter` or `Result` operation ID inside the `body` network to map to.
            * **Range of values**: IDs of the `Parameter` operations inside in the *Loop* operation
            * **Type**: `int`
            * **Default value**: None
            * **Required**: *yes*

        * *axis*

            * **Description**: *axis* is an axis to concatenate the body `Result` output across all iterations. Can be specified for `output` edges only.
            * **Range of values**: an integer
            * **Type**: `int`
            * **Default value**: None
            * **Required**: *no*

* **Back edges**:

    *back_edges* is a set of rules to transfer tensor values from `body` outputs at one iteration to `body` parameters at the next iteration. Back edge connects some `Result` operation in the `body` to `Parameter` operation in the same `body`.

    * **Back edge attributes**:

        * *from-operation*

            * **Description**: *from-operation* is a `Result` operation ID inside the `body` network.
            * **Range of values**: IDs of the `Result` operations inside the *Loop*
            * **Type**: `int`
            * **Default value**: None
            * **Required**: *yes*

        * *to-operation*

            * **Description**: *to-operation* is a `Parameter` operation ID inside the `body` network to end mapping.
            * **Range of values**: IDs of the `Parameter` operations inside the *Loop*
            * **Type**: `int`
            * **Default value**: None
            * **Required**: *yes*

**Loop Inputs**

* **Trip count**: A scalar tensor of `int64` type specifying maximum number of iterations. *Optional*.

* **Termination condition**: A scalar tensor of `boolean` type specifying whether to execute the first iteration or not. *Optional*.

* **Multiple other inputs**: tensors of different types and shapes. *Optional*.

**Loop Outputs**

* **Multiple outputs**: Results of execution of the `body`. Tensors of any type and shape.


**Body Inputs**

* **Current iteration**: A scalar tensor of `int64` type specifying the current iteration number. *Optional*.

* **Termination condition**: A scalar tensor of `boolean` type specifying whether to execute the current iteration or not. *Optional*.

* **Multiple other inputs**: tensors of different types and shapes. *Optional*.

**Body Outputs**

* **Termination condition**: A scalar tensor of `boolean` type specifying whether to execute the next iteration or not.

* **Multiple outputs**: Results of execution of the `body`. Tensors of any type and shape.


**Examples**

*Example 1: a typical Loop structure*
```xml
<operation type="Loop" ... >
    <input> ... </input>
    <output> ... </output>
    <port_map>
        <input external_port_id="0" internal_operation_id="0" axis="1" start="-1" end="0" stride="-1"/>
        <input external_port_id="1" internal_operation_id="1"/>
        ...
        <output external_port_id="3" internal_operation_id="2" axis="1" start="-1" end="0" stride="-1"/>
        ...
    </port_map>
    <back_edges>
        <edge from-operation="1" to-operation="2"/>
        ...
    </back_edges>
    <body>
        <operations> ... </operations>
        <edges> ... </edges>
    </body>
</operation>
```

*Example 2: a full Loop operation*

```xml
<operation type="Loop" ...>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>25</dim>
            <dim>512</dim>
        </port>
        <port id="1">
            <dim>1</dim>
            <dim>256</dim>
        </port>
        <port id="2">
            <dim>1</dim>
            <dim>256</dim>
        </port>
    </input>
    <output>
        <port id="3" precision="FP32">
            <dim>1</dim>
            <dim>25</dim>
            <dim>256</dim>
        </port>
    </output>
    <port_map>
        <input axis="1" external_port_id="0" internal_operation_id="0" start="0"/>
        <input external_port_id="1" internal_operation_id="3"/>
        <input external_port_id="2" internal_operation_id="4"/>
        <output axis="1" external_port_id="3" internal_operation_id="12"/>
    </port_map>
    <back_edges>
        <edge from-operation="8" to-operation="4"/>
        <edge from-operation="9" to-operation="3"/>
    </back_edges>
    <body>
        <operations>
            <operation id="0" type="Parameter" ...>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </operation>
            <operation id="1" type="Const" ...>
                <data offset="0" size="16"/>
                <output>
                    <port id="1" precision="I64">
                        <dim>2</dim>
                    </port>
                </output>
            </operation>
            <operation id="2" type="Reshape" ...>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>2</dim>
                    </port>
                </input>
                <output>
                    <port id="2" precision="FP32">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                </output>
            </operation>
            <operation id="3" type="Parameter" ...>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </operation>
            <operation id="4" type="Parameter" ...>
                <output>
                    <port id="0" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </operation>
            <operation id="5" type="Const" ...>
                <data offset="16" size="3145728"/>
                <output>
                    <port id="1" precision="FP32">
                        <dim>1024</dim>
                        <dim>768</dim>
                    </port>
                </output>
            </operation>
            <operation id="6" type="Const" ...>
                <data offset="3145744" size="4096"/>
                <output>
                    <port id="1" precision="FP32">
                        <dim>1024</dim>
                    </port>
                </output>
            </operation>
            <operation id="7" type="LSTMCell" ...>
                <data hidden_size="256"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>512</dim>
                    </port>
                    <port id="1">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="2">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="3">
                        <dim>1024</dim>
                        <dim>768</dim>
                    </port>
                    <port id="4">
                        <dim>1024</dim>
                    </port>
                </input>
                <output>
                    <port id="5" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="6" precision="FP32">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </operation>
            <operation id="8" type="Result" ...>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </operation>
            <operation id="9" type="Result" ...>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </operation>
            <operation id="10" type="Const" ...>
                <data offset="3149840" size="24"/>
                <output>
                    <port id="1" precision="I64">
                        <dim>3</dim>
                    </port>
                </output>
            </operation>
            <operation id="11" type="Reshape" ...>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                    <port id="1">
                        <dim>3</dim>
                    </port>
                </input>
                <output>
                    <port id="2" precision="FP32">
                        <dim>1</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </output>
            </operation>
            <operation id="12" type="Result" ...>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>1</dim>
                        <dim>256</dim>
                    </port>
                </input>
            </operation>
        </operations>
        <edges>
            <edge from-operation="0" from-port="0" to-operation="2" to-port="0"/>
            <edge from-operation="1" from-port="1" to-operation="2" to-port="1"/>
            <edge from-operation="2" from-port="2" to-operation="7" to-port="0"/>
            <edge from-operation="3" from-port="0" to-operation="7" to-port="1"/>
            <edge from-operation="4" from-port="0" to-operation="7" to-port="2"/>
            <edge from-operation="5" from-port="1" to-operation="7" to-port="3"/>
            <edge from-operation="6" from-port="1" to-operation="7" to-port="4"/>
            <edge from-operation="7" from-port="6" to-operation="8" to-port="0"/>
            <edge from-operation="7" from-port="5" to-operation="9" to-port="0"/>
            <edge from-operation="7" from-port="5" to-operation="11" to-port="0"/>
            <edge from-operation="10" from-port="1" to-operation="11" to-port="1"/>
            <edge from-operation="11" from-port="2" to-operation="12" to-port="0"/>
        </edges>
    </body>
</operation>
```