Loop
====


.. meta::
  :description: Learn about Loop-5 - an infrastructure operation, which
                can be performed on two required and one optional input tensor.

**Versioned name**: *Loop-5*

**Category**: *Infrastructure*

**Short description**: *Loop* operation performs recurrent execution of the network, which is described in the ``body``, iterating through the data.
The operation has similar semantic to the ONNX Loop `operation <https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Loop-13>`__.

**Detailed description**

The body of the Loop can be executed 0 or more times depending on the values passed to the Loop operation inputs called "trip count", "execution condition" and input of the Loop body called "current iteration".

These Loop operation inputs have the following meaning:

1. Trip count is an integer scalar or 1D tensor with 1 element input specifying maximum number of iterations. To simulate infinite loop Constant ``-1`` can be provided as input.
2. Loop execution condition input is a boolean scalar or 1D tensor with 1 element input specifying whether to run the first loop iteration or not. Note, that the body of the Loop must yield the condition value for the consecutive iterations.

There are several combinations of these two inputs ``(trip_count, execution condition)`` which are described in the following code snippet:

.. code-block:: sh

     input (-1, true) // infinite loop
         bool cond = true;
         for (int i = 0; cond; ++i)
         {
             cond = true; // sub-graph calculating condition must always return "true"!
         }

     input (-1, cond) // while loop
         bool cond = ...;
         for (int i = 0; cond; ++i)
         {
             cond = ...;
         }

     input (-1, true) // do-while loop
         bool cond = true;
         for (int i = 0; cond; ++i)
         {
             cond = ...;
         }

     input (trip_count, true) // for loop
         int trip_count = ...;
         bool cond = true;
         for (int i = 0; i < trip_count; ++i)
         {
             cond = true; // sub-graph calculating condition must always return "true"!
         }

     input (trip_count, cond) // for with condition
         int trip_count = ...;
         bool cond = ...;
         for (int i = 0; i < trip_count && cond; ++i)
         {
             cond = ...;
         }


1. One of the body graph inputs called "current iteration" is an integer scalar or 1D integer tensor with 1 number specifying current iteration number. The iteration number starts from 0 and incremented by one for each iteration. This input is optional and may not exist if the iteration number value is not used in the body.
2. One of the body graph outputs is called "condition" is a boolean scalar or 1D tensor with 1 element. This value is used to decide whenever to perform the next iteration or not.

Loop operation description in the IR has regular sections: ``input`` and ``output``. They connect Loop body to the outer graph and specify condition(s).
Loop operation description in the IR also has several special sections: ``body``, ``port_map`` and ``back_edges`` similar to the ones from the TensorIterator operation but having some important features described below.

1. The body operation getting an input from the main graph should have an entry in the ``port_map`` section of the Loop operation. These edges connect input ports of the Loop with the body ``Parameter``\ s.
2. Input tensors to the Loop can be sliced along a specified axis, the Loop can iterates over all sliced parts. The corresponding ``input`` entry in the ``port_map`` should have ``axis`` attribute specifying the axis to slice. Therefore, inputs to the Loop operation corresponding to ``input`` entries in the ``port_map`` without ``axis`` attribute are used "as is" (without slicing).
3. The body operation producing tensor to be used in the subsequent iterations (like in RNN models) should have a back edge described in the ``back_edges`` section of the operation. The back edge connects the respective body ``Parameter`` and ``Result`` operations. For such a case the Loop operation node provides input for the first iteration, while corresponding Loop operation output produces the tensor computed during the last iteration.
4. Output tensors produced by a particular body operation across all iterations can be concatenated and returned as a Loop operation output (this is a "scan output" according to the ONNX* Loop operation `specification <https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Loop-13>`__ ). The corresponding ``output`` entry in the ``port_map`` should have ``axis`` attribute specifying the axis to concatenate. Therefore, outputs from operations corresponding to ``output`` entries in the ``port_map`` without ``axis`` attribute are returned "as is" (without concatenation).
5. There is one body ``Parameter`` operation not connected through the ``port_map``. This is a "current iteration" input. The Loop operation is responsible for providing the appropriate value for each iteration.
6. Connection of nodes inside the Loop body with the main graph should be done through ``Parameter`` and ``Result`` body operations. No other ways to connect graphs are allowed.

**Loop attributes**:

* **Body**:

  ``body`` is a network that will be recurrently executed. The network is described operation by operation as a typical IR network.

  * **Body attributes**:

    No attributes available.

* **Port map**:

  *port_map* is a set of rules to map input or output data tensors of ``Loop`` operation onto ``body`` data tensors. The ``port_map`` entries can be`` input`` and ``output``. Each entry describes a corresponding mapping rule.

  * **Port map attributes**:

    * *external_port_id*

      * **Description**: *external_port_id* is a port ID of the ``Loop`` operation. The value ``-1`` means that the body node is not connected to the ``Loop`` operation.
      * **Range of values**: IDs of the *Loop* outputs
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

    * *internal_layer_id*

      * **Description**: *internal_layer_id* is a ``Parameter`` or ``Result`` operation ID inside the ``body`` network to map to.
      * **Range of values**: IDs of the ``Parameter`` operations inside in the *Loop* operation
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

    * *axis*

      * **Description**: if *axis* is specified for ``output`` entry, then it is an axis to concatenate the body ``Result`` output across all iterations.
        If *axis* is specified for ``input`` entry, then it is an axis to iterate through, it triggers the slicing of the input tensor.

      * **Range of values**: an integer. Negative value means counting dimension from the end.
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *no*

* **Back edges**:

  *back_edges* is a set of rules to transfer tensor values from ``body`` outputs at one iteration to ``body`` parameters at the next iteration. Back edge connects some ``Result`` operation in the ``body`` to ``Parameter`` operation in the same ``body``.

  * **Back edge attributes**:

    * *from-layer*

      * **Description**: *from-layer* is a ``Result`` operation ID inside the ``body`` network.
      * **Range of values**: IDs of the ``Result`` operations inside the *Loop*
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

    * *to-layer*

      * **Description**: *to-layer* is a ``Parameter`` operation ID inside the ``body`` network to end mapping.
      * **Range of values**: IDs of the ``Parameter`` operations inside the *Loop*
      * **Type**: ``int``
      * **Default value**: None
      * **Required**: *yes*

**Loop Inputs**

* **Trip count**: A scalar or 1D tensor with 1 element of ``int64`` or ``int32`` type specifying maximum number of iterations. **Required.**

* **ExecutionCondition**: A scalar or 1D tensor with 1 element of ``boolean`` type specifying whether to execute the first iteration or not. ``True`` value means to execute the 1st iteration. **Required.**

* **Multiple other inputs**: tensors of different types and shapes. **Optional.**

**Loop Outputs**

* **Multiple outputs**: Results of execution of the ``body``. Tensors of any type and shape.


**Body Inputs**

* **Multiple inputs**: tensors of different types and shapes except the one corresponding to the current iteration number. This input is marked in the port_map with attribute ``purpose = "current_iteration"`` and produces a scalar or 1D tensor with 1 element of ``int64`` or ``int32`` type. **Optional.**


**Body Outputs**

* **Multiple outputs**: Results of execution of the ``body``. Tensors of any type and shape except the one corresponding to the output with execution condition. This output is marked in the port_map with attribute ``purpose = "execution_condition"`` and is mandatory and produces a scalar or 1D tensor with 1 element of ``boolean`` type. Other outputs are optional.

**Examples**

*Example 1: a typical Loop structure*

.. code-block:: xml
   :force:

   <layer type="Loop" ... >
       <input> ... </input>
       <output> ... </output>
       <port_map>
           <input external_port_id="0" internal_layer_id="0"/>
           <input external_port_id="1" internal_layer_id="1"/>
           <input external_port_id="-1" internal_layer_id="2" purpose="current_iteration"/>
           ...
           <output external_port_id="3" internal_layer_id="4"/>
           <output external_port_id="4" internal_layer_id="10" axis="1"/>
           <output external_port_id="-1" internal_layer_id="22" purpose="execution_condition"/>
           ...
       </port_map>
       <back_edges>
           <edge from-layer="1" to-layer="5"/>
           ...
       </back_edges>
       <body>
           <layers> ... </layers>
           <edges> ... </edges>
       </body>
   </layer>



