ReadValue
=========


.. meta::
  :description: Learn about ReadValue-6 - an infrastructure operation, which
                can be performed on a single input tensor or without input tensors
                to return the value of variable_id.

**Versioned name**: *ReadValue-6*

**Category**: *Infrastructure*

**Short description**: *ReadValue* returns value of the ``variable_id`` variable.

**Detailed description**:

*ReadValue*, *Assign*, and *Variable* define a coherent mechanism for reading, writing,
and storing some memory buffer between inference calls. More details can be found on the
:doc:`StateAPI <../../../../../../openvino-workflow/running-inference/stateful-models>` documentation page.

If the 1st input is provided and this is the first inference or reset has been called,
*ReadValue* returns the value from the 1st input.

If the 1st input is not provided and this is the first inference or reset has been called,
*ReadValue* returns the tensor with the ``variable_shape`` and ``variable_type`` and zero values.

In all other cases *ReadValue* returns the value from the corresponding ``variable_id`` variable.

If the 1st input has been provided, the operation checks if ``variable_shape`` and ``variable_type``
extend (relax) the shape and type inferred from the 1st input. If not, it returns an error.
For example, if ``variable_type`` is specified as dynamic, it means that any type for 1st input
is allowed but if it is specified as f32, only f32 type is allowed.

Only one pair of ReadValue and Assign operations is expected for each Variable in the model.


**Attributes**:

* *variable_id*

  * **Description**: identifier of the variable to be read.
  * **Range of values**: any non-empty string
  * **Type**: string
  * **Required**: *yes*

* *variable_type*

  * **Description**: the type of the variable
  * **Range of values**: : u1, u4, u8, u16, u32, u64, i4, i8, i16, i32, i64, f16, f32, boolean, bf16
  * **Type**: ``string``
  * **Required**: *yes*

* *variable_shape*

  * **Description**: the shape of the variable
  * **Range of values**: list of integers, empty list is allowed, which means 0D or scalar tensor
  * **Type**: ``int[]``
  * **Required**: *yes*

**Inputs**

*   **1**: ``init_value`` - input tensor whose values are used in the first inference or after a reset call. **Optional.**

**Outputs**

*   **1**: tensor with the same shape and type as specified in *variable_type*, *variable_shape*.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="ReadValue" ...>
        <data variable_id="lstm_state_1" variable_type="f32" variable_shape="1,3,224,224"/>
        <input>
            <port id="0">
                <dim>1</dim>
                <dim>3</dim>
                <dim>224</dim>
                <dim>224</dim>
            </port>
        </input>
        <output>
            <port id="1">
                <dim>1</dim>
                <dim>3</dim>
                <dim>224</dim>
                <dim>224</dim>
            </port>
        </output>
    </layer>


