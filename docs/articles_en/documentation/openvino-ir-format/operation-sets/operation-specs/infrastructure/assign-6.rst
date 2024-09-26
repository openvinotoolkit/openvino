Assign
======


.. meta::
  :description: Learn about Assign-6 - an infrastructure operation, which
                can be performed on a single input tensor to set a value to variable_id.

**Versioned name**: *Assign-6*

**Category**: *Infrastructure*

**Short description**: *Assign* sets an input value to the ``variable_id`` variable.

**Detailed description**:

ReadValue, Assign, and Variable define a coherent mechanism for reading, writing and
storing a memory buffer between inference calls. More details can be found on the
:doc:`State API <../../../../../../openvino-workflow/running-inference/stateful-models>` documentation page.

*Assign* sets an input value to the ``variable_id`` variable. This value will be read
by the *ReadValue* operation on the next inference call if it has not been reset.
The operation checks if the shape and type specified in ``variable_id`` extend (relax)
the shape and type inferred from the 1st input. If not, it returns an error. For example,
if the type in the variable is specified as dynamic, it means that any type for 1st
input is allowed but if it is specified as f32, only f32 type is allowed.

Only one pair of ReadValue and Assign operations is expected for each Variable in the model.

**Attributes**:

* *variable_id*

  * **Description**: identifier of the variable to be updated
  * **Range of values**: any non-empty string
  * **Type**: string
  * **Required**: *yes*

**Inputs**

* **1**: ``new_value`` - input tensor of any supported type. **Required.**

**Outputs**

* **1**: tensor with the same shape and type as ``new_value``.

**Example**

.. code-block:: xml
   :force:

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


