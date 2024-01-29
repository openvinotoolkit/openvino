.. {#openvino_docs_ops_infrastructure_Assign_6}

Assign
======


.. meta::
  :description: Learn about Assign-6 - an infrastructure operation, which 
                can be performed on a single input tensor to set a value to variable_id.

**Versioned name**: *Assign-6*

**Category**: *Infrastructure*

**Short description**: *Assign* sets an input value to the ``variable_id`` variable.

**Detailed description**:

ReadValue, Assign and Variable define a coherent mechanism for reading, writing and storing a memory buffer between inference calls.
More details can be found on :doc:`StateAPI<openvino_docs_OV_UG_stateful_models_intro>` documentation page.

*Assign* operation sets an input value to the ``variable_id`` variable. This value will be read by *ReadValue* operation on next inference call if variable was not reset.
The operation checks that the shape and type specified in ``variable_id`` variable extend (relax)
the shape and the type inferred from the 1st input and returns an error otherwise, e.g. if the type in the variable is specified
as dynamic, it means that any type for 1st input is allowed but if it is specified as f32, only f32 type is allowed.

It is expected only one pair of ReadValue, Assign operations for each Variable in the model.

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


