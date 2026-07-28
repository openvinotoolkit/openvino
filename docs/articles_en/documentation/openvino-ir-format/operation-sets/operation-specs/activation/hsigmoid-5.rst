HSigmoid
========


.. meta::
  :description: Learn about HSigmoid-5 - an element-wise, activation operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *HSigmoid-5*

**Category**: *Activation function*

**Short description**: HSigmoid takes one input tensor and produces output tensor where the hard version of sigmoid function is applied to the tensor elementwise.

**Detailed description**: For each element from the input tensor calculates corresponding
element in the output tensor with the following formula:

.. math::

   HSigmoid(x) = \frac{min(max(x + 3,\ 0),\ 6)}{6}


The HSigmoid operation is introduced in the following `article <https://arxiv.org/pdf/1905.02244.pdf>`__.

**Attributes**: operations has no attributes.

**Inputs**:

* **1**: A tensor of type *T*. **Required.**

**Outputs**:

* **1**: The resulting tensor of the same shape and type as input tensor.

**Types**

* *T*: any floating-point type.

**Examples**

.. code-block:: xml
   :force:

   <layer ... type="HSigmoid">
       <input>
           <port id="0">
               <dim>256</dim>
           </port>
       </input>
       <output>
           <port id="1">
               <dim>256</dim>
           </port>
       </output>
   </layer>


