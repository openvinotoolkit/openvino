Unique
======


.. meta::
  :description: Learn about Unique-10 - a data movement operation, which can be
                performed on one required and one optional input tensor.

**Versioned name**: *Unique-10*

**Category**: *Data movement*

**Short description**: *Unique* finds unique elements of a given input tensor. Depending on the operator's attributes it either looks for unique values globally in the whole (flattened) tensor or performs the search along the specified axis.

**Detailed description**
The operator can either work in elementwise mode searching for unique values in the whole tensor or it can consider slices of the input tensor along the specified axis. This way the op is able to find unique subtensors in the input data. Except for the unique elements the operator also produces the indices of the unique elements in the input tensor and the number of occurrences of unique elements in the input data.

**Attributes**:

* *sorted*

  * **Description**: controls whether the unique elements in the output tensor are sorted in ascending order.
  * **Range of values**:

    * false - output tensor's elements are not sorted
    * true - output tensor's elements are sorted
  * **Type**: boolean
  * **Default value**: true
  * **Required**: *no*

* *index_element_type*

  * **Description**: controls the data type of the output tensors containing indices.
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *no*

* *count_element_type*

  * **Description**: controls the data type of the last output tensor.
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i64"
  * **Required**: *no*

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**
* **2**: A tensor of type *T_AXIS*. The allowed tensor shape is 1D with a single element or a scalar. If provided, this input has to be connected to a Constant. **Optional**
  When provided this input is used to "divide" the input tensor into slices along the specified axis before unique elements processing starts. When this input is not provided the operator works on a flattened version of the input tensor (elementwise processing). The range of allowed values is ``[-r; r-1]`` where ``r`` is the rank of the input tensor.

**Outputs**

* **1**: The output tensor containing unique elements (individual values or subtensors). This tensor's type matches the type of the first input tensor: *T*. The values in this tensor are either sorted ascendingly or maintain the same order as in the input tensor. The shape of this output depends on the values of the input tensor and will very often be dynamic. Please refer to the article describing how :doc:`Dynamic Shapes <../../../../../openvino-workflow/running-inference/dynamic-shapes>` are handled in OpenVINO.
* **2**: The output tensor containing indices of the locations of unique elements. The indices map the elements in the first output tensor to their locations in the input tensor. The index always points to the first occurrence of a given unique output element in the input tensor. This is a 1D tensor with type controlled by the ``index_element_type`` attribute.
* **3**: The output tensor containing indices of the locations of elements of the input tensor in the first output tensor. This means that for each element of the input tensor this output will point to the unique value in the first output tensor of this operator. This is a 1D tensor with type controlled by the ``index_element_type`` attribute.
* **4**: The output tensor containing the number of occurrences of each unique value produced by this operator in the first output tensor. This is a 1D tensor with type controlled by the ``count_element_type`` attribute.

**Types**

* *T*: any supported data type.
* *T_AXIS*: ``int64`` or ``int32``.

**Examples**

*Example 1: axis input connected to a constant containing a 'zero'*

.. code-block:: xml
   :force:

    <layer ... type="Unique" ... >
        <data sorted="false" index_element_type="i32"/>
        <input>
            <port id="0" precision="FP32">
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </input>
        <input>
            <port id="1" precision="I64">
                <dim>1</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="FP32">
                <dim>-1</dim>
                <dim>3</xdim>
            </port>
            <port id="3" precision="I32">
                <dim>-1</dim>
            </port>
            <port id="4" precision="I32">
                <dim>3</dim>
            </port>
            <port id="5" precision="I64">
                <dim>-1</dim>
            </port>
        </output>
    </layer>


*Example 2: no axis provided*

.. code-block:: xml
   :force:

    <layer ... type="Unique" ... >
        <input>
            <port id="0" precision="FP32">
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="FP32">
                <dim>-1</dim>
            </port>
            <port id="2" precision="I64">
                <dim>-1</dim>
            </port>
            <port id="3" precision="I64">
                <dim>9</dim>
            </port>
            <port id="4" precision="I64">
                <dim>-1</dim>
            </port>
        </output>
    </layer>

*Example 3: no axis provided, non-default outputs precision*

.. code-block:: xml
   :force:

    <layer ... type="Unique" ... >
        <data sorted="false" index_element_type="i32" count_element_type="i32"/>
        <input>
            <port id="0" precision="FP32">
                <dim>3</dim>
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="1" precision="FP32">
                <dim>-1</dim>
            </port>
            <port id="2" precision="I32">
                <dim>-1</dim>
            </port>
            <port id="3" precision="I32">
                <dim>9</dim>
            </port>
            <port id="4" precision="I32">
                <dim>-1</dim>
            </port>
        </output>
    </layer>



