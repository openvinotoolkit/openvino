If
==


.. meta::
  :description: Learn about If-8 - an element-wise, condition operation, which
                can be performed on multiple tensors in OpenVINO.

**Versioned name**: *If-8*

**Category**: *Condition*

**Short description**: *If* operation contains two internal networks(subgraphs) such as ``then_body`` and ``else_body``,
and performs one of them depending on ``cond`` value. If ``cond`` is  ``True``, ``then_body`` is executed. If ``cond`` is  ``False``,
the operation executes the ``else_body`` subgraph.

**Detailed description**

*If* must not contain empty subgraphs. Each of them must have at least one operation ``Result``.
Also the number of outputs from *If* always must be greater than zero and equal to the number of outputs from each subgraph.

**If attributes**:

* **Subgraphs**:

  ``then_body``/``else_body`` are subgraphs that are executed depending on the ``cond`` value.
  The subgraph is described operation by operation as a typical IR network.
  The subgraph has inputs (``Parameter`` operations) and outputs (``Result`` operations).

  * **Subgraph's inputs** - inputs to the subgraph which associated with *If* inputs via *port_map*.

    The subgraph can have any number of inputs (even zero).

  * **Subgraph's outputs** - outputs from the subgraph which associated with *If* outputs via *port_map*.

    The subgraph must contain at least one output. Each *If* output is associated with one output from the subgraph.
    Therefore the number of ``then_body`` outputs is equal to the number of outputs from *If* and
    the number of ``else_body`` outputs.
    The type of the subgraph output and the type of the associated output from *If* must be equal.


* **Port maps**:

  *port_map* is a set of rules to map input or output data tensors of *If* operation onto the subgraph data tensors.
  The ``port_map`` entries can be ``input`` and ``output``. Each entry describes a corresponding mapping rule.
  *If* has two *port_maps*: ``then_port_map`` for ``then_body`` and ``else_port_map`` for ``else_body``.

  * **Port map attributes**:

    * *external_port_id*

      * **Description**: *external_port_id* is a port ID of *If* operation.
      * **Range of values**: IDs of the *If* inputs and outputs
      * **Type**: ``unsigned int``
      * **Default value**: None
      * **Required**: *yes*

    * *internal_layer_id*

      * **Description**: *internal_layer_id* is a ``Parameter`` or ``Result`` operation ID inside
        the subgraph to map to.
      * **Range of values**: IDs of the ``Parameter`` or ``Result`` operations in the subgraph
      * **Type**: ``unsigned int``
      * **Default value**: None
      * **Required**: *yes*

**If Inputs**


* **cond**: A scalar or 1D tensor with 1 element of ``boolean`` type specifying which subgraph to execute.
  ``True`` value means to execute the ``then_body``, ``False`` - ``else_body``. *Required*.

* **Multiple other inputs**: Tensors of different types and shapes. *Optional*.

**If Outputs**

* **Multiple outputs**: Results of execution of one of the subgraph. Tensors of any type and shape.


**Body Inputs**

* **Multiple inputs**: Tensors of different types and shapes. *Optional*.


**Body Outputs**

* **Multiple outputs**: Results of execution of the subgraph. Tensors of any type and shape.


**Examples**

*Example 1: a typical If structure*


.. code-block:: xml
   :force:

   <layer id="6" name="if/cond" type="If" version="opset8">
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
           <port id="4" names="if/cond/Identity:0,if/cond:0" precision="FP32">
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



