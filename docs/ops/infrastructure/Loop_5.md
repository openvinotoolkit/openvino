## Loop <a name="Loop"></a> {#openvino_docs_ops_infrastructure_Loop_5}

**Versioned name**: *Loop-5*

**Category**: Infrastructure

**Short description**: *Loop* operation performs recurrent execution of the network, which is described in the `body`, iterating through the data. 
The operation has similar semantic to the ONNX* Loop [operation](https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Loop-13).

**Detailed description**

The body of the Loop can be executed 0 or more times depending on the values passed to the Loop operation inputs called "trip count" and "termination condition".

1. Trip count input is an integer scalar input specifying maximum number of iterations. When this input is not connected perform infinite number of iterations.
2. Loop termination condition input is a boolean scalar input specifying whether to run the current loop iteration or not. When this input is not connected perform first iteration.
Note, that the body of the Loop must yield the termination condition value whether the input is provided or not. 

There are several combinations of these two inputs which are described in the following code snippet (empty string `""` means no input provided):

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

The body graph has at least two inputs: 
1. The "current iteration" number which is an integer scalar number.
2. The "termination condition" which is a boolean scalar value. This value is provided from the corresponding input of the Loop operation for the first iteration and calculated in the body graph for the consequent iterations.

Loop operation description in the IR has regular sections: `input` and `output`. They connect Loop body to the outer graph and specify termination condition(s).
Loop operation description in the IR also has several special sections: `body`, `port_map` and `back_edges` similar to the ones from the TensorIterator operation but having some important features described below.

1. The body operation getting an input from the main graph should have an entry in the `port_map` section of the Loop operation. These edges connect input ports of the Loop with the body `Parameter`s.
1. The body operation producing tensor to be used in the subsequent iterations (like in RNN models) should have a back edge described in the `back_edges` section of the operation. The back edge connects the respective body `Parameter` and `Result` operations. For such a case the Loop operation node provides input for the first iteration, while corresponding Loop operation output produces the tensor computed during the last iteration.
1. Output tensors produced by a particular body operation across all iterations can be concatenated and returned as a Loop operation output (this is a "scan output" according to the ONNX* Loop operation [specification](https://github.com/onnx/onnx/blob/master/docs/Changelog.md#Loop-13)). The corresponding `output` entry in the `port_map` should have `axis` attribute specifying the axis to concatenate. Therefore, outputs from operations corresponding to `output` entries in the `port_map` without `axis` attribute are returned "as is" (without concatenation).
1. There is one body `Parameter` operation not connected through the `port_map`. This is a "current iteration" input. The Loop operation is responsible for providing the appropriate value for each iteration.
1. The body `Parameter` operation corresponding to termination condition must have attribute `"condition=True"` in the corresponding `port_map` entry. If termination condition input is not connected then no entries should have this attribute set.
1. Connection of nodes inside the Loop body with the main graph should be done through `Parameter` and `Result` body operations. No other ways to connect graphs are allowed.

**Loop attributes**:

* **Body**:

    `body` is a network that will be recurrently executed. The network is described operation by operation as a typical IR network.

    * **Body attributes**:

            No attributes available.

* **Port map**:

    *port_map* is a set of rules to map input or output data tensors of `Loop` operation onto `body` data tensors. The `port_map` entries can be` input` and `output`. Each entry describes a corresponding mapping rule.

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

        * *condition*

            * **Description**: *condition* is a boolean flag to mark input as a termination condition input. Can be specified as `True` for only one `input` edge.
            * **Range of values**: True or False
            * **Type**: `boolean`
            * **Default value**: False
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

* **Current iteration**: A scalar tensor of `int64` type specifying the current iteration number. *Required*.

* **Termination condition**: A scalar tensor of `boolean` type specifying whether to execute the current iteration or not. *Required*.

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
        <input external_port_id="0" internal_operation_id="0" condition="True"/>
        <input external_port_id="1" internal_operation_id="1"/>
        ...
        <output external_port_id="3" internal_operation_id="2"/>
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
