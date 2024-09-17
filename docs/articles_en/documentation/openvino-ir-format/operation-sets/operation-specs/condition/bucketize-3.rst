Bucketize
=========


.. meta::
  :description: Learn about Bucketize-3 - an element-wise, condition operation, which
                can be performed on two given tensors in OpenVINO.

**Versioned name**: *Bucketize-3*

**Category**: *Condition*

**Short description**: *Bucketize* bucketizes the input based on boundaries. This is similar to `Reference <https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/bucketize>`__ .

**Detailed description**: *Bucketize* computes a bucket index for each element from the first input and outputs a tensor of the first input shape. Buckets are defined with boundaries from the second input.

For example, if the first input tensor is ``[[3, 50], [10, -1]]`` and the second input is ``[0, 5, 10]`` with included right bound, the output will be ``[[1, 3], [2, 0]]``.

**Attributes**

* *output_type*

  * **Description**: the output tensor type
  * **Range of values**: "i64" or "i32"
  * **Type**: ``string``
  * **Default value**: "i64"
  * **Required**: *no*

* *with_right_bound*

  * **Description**: indicates whether bucket includes the right or the left edge of interval.
  * **Range of values**:

    * true - bucket includes the right interval edge
    * false - bucket includes the left interval edge
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

**Inputs**:

* **1**: N-D tensor of *T* type with elements for the bucketization. **Required.**
* **2**: 1-D tensor of *T_BOUNDARIES* type with sorted unique boundaries for buckets. **Required.**

**Outputs**:

* **1**: Output tensor with bucket indices of *T_IND* type. If the second input is empty, the bucket index for all elements is equal to 0. The output tensor shape is the same as the first input tensor shape.

**Types**

* *T*: any numeric type.

* *T_BOUNDARIES*: any numeric type.

* *T_IND*: ``int32`` or ``int64``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Bucketize">
       <input>
           <port id="0">
               <dim>49</dim>
               <dim>11</dim>
           </port>
           <port id="1">
               <dim>5</dim>
           </port>
        </input>
       <output>
           <port id="1">
               <dim>49</dim>
               <dim>11</dim>
           </port>
       </output>
   </layer>


