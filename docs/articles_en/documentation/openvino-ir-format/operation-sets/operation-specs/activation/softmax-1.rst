SoftMax
=======


.. meta::
  :description: Learn about SoftMax-1 - an element-wise, activation operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *SoftMax-1*

**Category**: *Activation function*

**Short description**: `Reference <https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions#softmax>`__

**Detailed description**: `Reference <http://cs231n.github.io/linear-classify/#softmax>`__

**Attributes**

* *axis*

  * **Description**: *axis* represents the axis of which the *SoftMax* is calculated. *axis* equal 1 is a default value.
  * **Range of values**: positive integer value
  * **Type**: int
  * **Default value**: 1
  * **Required**: *no*

**Mathematical Formulation**

.. math::

   y_{c} = \frac{e^{Z_{c}}}{\sum_{d=1}^{C}e^{Z_{d}}}

where :math:`C` is a size of tensor along *axis* dimension.

**Inputs**:

*   **1**: Input tensor with enough number of dimension to be compatible with *axis* attribute. **Required.**

**Outputs**:

*   **1**: The resulting tensor of the same shape and type as input tensor.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="SoftMax" ... >
       <data axis="1" />
       <input> ... </input>
       <output> ... </output>
   </layer>

