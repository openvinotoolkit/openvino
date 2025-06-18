.. {#openvino_docs_ops_internal_RMS}

RMS
===


.. meta::
  :description: Learn about RMS a normalization operation.

**Versioned name**: *RMS*

**Category**: *Normalization*

**Short description**: Calculates Root Mean Square (RMS) normalization of the input tensor.

**Detailed description**

*RMS* operation performs Root Mean Square (RMS) normalization on a given input ``data`` along the last dimension of the input.
`Reference <https://arxiv.org/abs/1910.07467>`__.


.. code-block:: py

    (x / Sqrt(ReduceMean(x^2, -1) + eps)) * scale


**Attributes**

* *epsilon*

  * **Description**: A very small value added to the variance for numerical stability. Ensures that division by zero does not occur for any normalized element.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *output_type*

  * **Description**: The precision for output type conversion, after scaling. It's used for output type compression to f16.
  * **Range of values**: Supported floating point type: "f16", "undefined"
  * **Type**: ``string``
  * **Default value**: "undefined" (means that output type is set to the same as the input type)
  * **Required**: *no*


**Inputs**

* **1**: ``data`` - Input data to be normalized. A tensor of type *T* and arbitrary shape. **Required.**

* **2**: ``scale`` - A tensor of type *T* containing the scale values for . The shape should be broadcastable to the shape of ``data`` tensor. **Required.**


**Outputs**

* **1**: Output tensor of the same shape as the ``data`` input tensor and type specified by *output_type* attribute.

**Types**

* *T*: any floating point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="RMS"> <!-- normalization always over the last dimension [-1] -->
       <data eps="1e-6"/>
       <input>
           <port id="0">
               <dim>12</dim>
               <dim>25</dim>
               <dim>512</dim>
           </port>
           <port id="1">
               <dim>512</dim>
           </port>
       </input>
       <output>
           <port id="2">
               <dim>12</dim>
               <dim>25</dim>
               <dim>512</dim>
           </port>
       </output>
   </layer>
