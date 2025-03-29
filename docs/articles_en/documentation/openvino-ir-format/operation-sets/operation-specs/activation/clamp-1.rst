Clamp
=====


.. meta::
  :description: Learn about Clamp-1 - an element-wise, activation operation, which
                can be performed on a single tensor in OpenVINO.

**Versioned name**: *Clamp-1*

**Category**: *Activation function*

**Short description**: *Clamp* operation represents clipping activation function.

**Detailed description**:

*Clamp* performs clipping operation over the input tensor element-wise. Element values of the output are within the range ``[min, max]``.

* Input values that are smaller than *min* are replaced with *min* value.
* Input values that are greater than *max* are replaced with *max* value.
* Input values within the range ``[min, max]`` remain unchanged.

Let *min_value* and *max_value* be *min* and *max*, respectively. The mathematical formula of *Clamp* is as follows:

.. math::

   clamp( x_{i} )=\min\big( \max\left( x_{i},\ min\_value \right),\ max\_value \big)

**Attributes**:

* *min*

  * **Description**: *min* is the lower bound of values in the output.
  * **Range of values**: arbitrary floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output.
  * **Range of values**: arbitrary floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

**Inputs**:

*   **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**:

*   **1**: A tensor of type *T* with same shape as input tensor.

**Types**

* *T*: any numeric type.
*   **Note**: In case of integral numeric type, ceil is used to convert *min* from ``float`` to *T* and floor is used to convert *max* from ``float`` to *T*.

**Example**

.. code-block:: xml
   :force:

   <layer id="1" name="clamp_node" type="Clamp">
       <data min="10" max="50" />
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

