Reverse
=======


.. meta::
  :description: Learn about Reverse-1 - a data movement operation,
                which can be performed on one required and one optional input tensor.

**Versioned name**: *Reverse-1*

**Category**: *Data movement*

**Short description**: *Reverse* operations reverse specified axis in an input tensor.

**Detailed description**: *Reverse* produces a tensor with the same shape as the first input tensor and with elements reversed along dimensions specified in the second input tensor. The axes can be represented either by dimension indices or as a mask. The interpretation of the second input is determined by *mode* attribute.

If ``index`` mode is used, the second tensor should contain indices of axes that should be reversed. The length of the second tensor should be in a range from 0 to rank of the 1st input tensor.

In case if ``mask`` mode is used, then the second input tensor length should be equal to the rank of the 1st input. And each value has boolean value ``true`` or ``false``. ``true`` means the corresponding axes should be reverted, ``false`` means it should be untouched.

If no axis specified, that means either the second input is empty if ``index`` mode is used or second input has only ``false`` elements if ``mask`` mode is used, then *Reverse* just passes the source tensor through output not doing any data movements.

**Attributes**

* *mode*

  * **Description**: specifies how the second input tensor should be interpreted: as a set of indices or a mask
  * **Range of values**: ``index``, ``mask``
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**:

*   **1**: ``data`` the tensor of type *T1* with input data to reverse. **Required.**

*   **2**: ``axis`` 1D tensor of type *T2* populated with indices of reversed axes if *mode* attribute is set to ``index``, otherwise 1D tensor of type *T3* and with a length equal to the rank of ``data`` input that specifies a mask for reversed axes.

**Outputs**:

*   **1**: output reversed tensor with shape and type equal to ``data`` tensor.

**Types**

* *T1*: any supported type.
* *T2*: any supported integer type.
* *T3*: boolean type.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="Reverse">
        <data mode="index"/>
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
            <port id="1">
                <dim>1</dim>   <!-- reverting along single axis -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>3</dim>
                <dim>10</dim>
                <dim>100</dim>
                <dim>200</dim>
            </port>
        </output>
    </layer>



