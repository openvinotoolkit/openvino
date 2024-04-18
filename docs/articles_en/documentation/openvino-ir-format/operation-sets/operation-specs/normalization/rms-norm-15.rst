.. {#openvino_docs_ops_normalization_RMS_15}

RMS
===


.. meta::
  :description: Learn about RMS-15 - a normalization operation.

**Versioned name**: *RMS-15*

**Category**: *Normalization*

**Short description**: Calculates Root Mean Square (RMS) normalization of the input tensor.

**Detailed description**

*RMSNorm* operation performs Root Mean Square (RMS) normalization on a given input ``data`` along dimensions specified by ``axes`` input.
`Reference <https://arxiv.org/abs/1910.07467>`__.

.. code-block:: py

    (x / Sqrt(ReduceMean(x^2, axes) + eps))


 -   If the optional ``scale`` input is provided:

.. code-block:: py

    (x / Sqrt(ReduceMean(x^2, axes) + eps)) * scale


**Attributes**

* *epsilon*

  * **Description**: A very small value added to the variance for numerical stability. Ensures that division by zero does not occur for any normalized element.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *compute_type*

  * **Description**: The precision for internal computation, before scaling.
  * **Range of values**: Supported floating point type: "f32", "f16", ...
  * **Type**: ``string``
  * **Default value**: "undefined" (the same as the input type)
  * **Required**: *no*


**Inputs**

* **1**: ``data`` - Input data to be normalized. A tensor of type *T* and arbitrary shape. **Required.**

* **2**: ``axes`` - 1D or scalar tensor which specifies indices of dimensions in ``data`` that define normalization slices. Allowed range of axes is ``[-r; r-1]`` where ``r = rank(data)``, the order can be not sorted. Negative value means counting dimensions from the back. Type *T_AXES*. **Required.**

* **3**: ``scale`` - A tensor of type *T* containing the scale values for . The shape should be broadcastable to the shape of ``data`` tensor. **Optional.**


**Outputs**

* **1**: Output tensor of the same shape and type as the ``data`` input tensor.

**Types**

* *T*: any floating point type.
* *T_AXES*: ``int64`` or ``int32``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="RMS">
       <data eps="1e-6"/>
       <input>
           <port id="0">
               <dim>6</dim>
               <dim>12</dim>
               <dim>10</dim>
               <dim>24</dim>
           </port>
           <port id="1">
               <dim>1</dim> <!-- value of [-1] means normalization over the last dimension -->
           </port>
       </input>
       <output>
           <port id="2">
               <dim>6</dim>
               <dim>12</dim>
               <dim>10</dim>
               <dim>24</dim>
           </port>
       </output>
   </layer>
