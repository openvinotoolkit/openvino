Broadcast
=========


.. meta::
  :description: Learn about Broadcast-1 - a data movement operation,
                which can be performed on two required and one optional input tensor.

**Versioned name**: *Broadcast-1*

**Category**: *Data movement*

**Short description**: *Broadcast* replicates data on the first input to fit a given shape on the second input.

**Detailed description**:

*Broadcast* takes the first tensor ``data`` and, following broadcasting rules that are specified by ``mode`` attribute and the 3rd input ``axes_mapping``, builds a new tensor with shape matching the 2nd input tensor ``target_shape``. ``target_shape`` input is a 1D integer tensor that represents required shape of the output.

Attribute ``mode`` and the 3rd input ``axes_mapping`` are relevant for cases when rank of the input ``data`` tensor doesn't match the size of the ``target_shape`` input. They both define how axes from ``data`` shape are mapped to the output axes. If ``mode`` is set to ``numpy``, it means that the standard one-directional numpy broadcasting rules are applied. These rules are described in :doc:`Broadcast Rules For Elementwise Operations <../../broadcast-rules>`, when only one-directional broadcasting is applied: input tensor ``data`` is broadcasted to ``target_shape`` but not vice-versa.

In case if ``mode`` is set to ``explicit``, then 3rd input ``axes_mapping`` comes to play. It contains a list of axis indices, each index maps an axis from the 1st input tensor ``data`` to axis in the output. The size of ``axis_mapping`` should match the rank of input ``data`` tensor, so all axes from ``data`` tensor should be mapped to axes of the output.

For example, ``axes_mapping = [1]`` enables broadcasting of a tensor with shape ``[C]`` to shape ``[N,C,H,W]`` by replication of initial tensor along dimensions 0, 2 and 3. Another example is broadcasting of tensor with shape ``[H,W]`` to shape ``[N,H,W,C]`` with ``axes_mapping = [1, 2]``. Both examples requires ``mode`` set to ``explicit`` and providing mentioned ``axes_mapping`` input, because such operations cannot be expressed with ``axes_mapping`` set to ``numpy``.


**Attributes**:

* *mode*

  * **Description**: specifies rules used for mapping of ``input`` tensor axes to output shape axes.
  * **Range of values**:

    * *numpy* - numpy broadcasting rules, aligned with ONNX Broadcasting. Description is available in `ONNX docs <https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md>`__.; only one-directional broadcasting is applied from ``data`` to ``target_shape``. If this attribute value is used, then the 3rd input for the operation shouldn't be provided.
    * *explicit* - mapping of the input ``data`` shape axes to output shape is provided as an explicit 3rd input.
  * **Type**: string
  * **Default value**: "numpy"
  * **Required**: *no*


**Inputs**:

* **1**: ``data`` - source tensor of any type and shape that is being broadcasted. **Required.**
* **2**: ``taget_shape`` - 1D integer tensor describing output shape. **Required.**
* **3**: ``axes_mapping`` - 1D integer tensor describing a list of axis indices, each index maps an axis from the 1st input tensor ``data`` to axis in the output. The index values in this tensor should be sorted, that disables on-the-fly transpositions of input ``data`` tensor while the broadcasting. ``axes_mapping`` input is optional depending on ``mode`` value.

**Outputs**:

* **1**: Output tensor with replicated content from the 1st tensor ``data`` and with shape matched ``target_shape``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="Broadcast" ...>
       <data mode="numpy"/>
       <input>
           <port id="0">
               <dim>16</dim>
               <dim>1</dim>
               <dim>1</dim>
          </port>
           <port id="1">
               <dim>4</dim>   <!--The tensor contains 4 elements: [1, 16, 50, 50] -->
           </port>
           <!-- the 3rd input shouldn't be provided with mode="numpy" -->
       </input>
       <output>
           <port id="2">
               <dim>1</dim>
               <dim>16</dim>
               <dim>50</dim>
               <dim>50</dim>
           </port>
       </output>
   </layer>

   <layer ... type="Broadcast" ...>
       <data mode="explicit"/>
       <input>
           <port id="0">
               <dim>16</dim>
          </port>
           <port id="1">
               <dim>4</dim>   <!--The tensor contains 4 elements: [1, 16, 50, 50] -->
           </port>
           <port id="1">
               <dim>1</dim>   <!--The tensor contains 1 elements: [1] -->
           </port>
       </input>
       <output>
           <port id="2">
               <dim>1</dim>
               <dim>16</dim>
               <dim>50</dim>
               <dim>50</dim>
           </port>
       </output>
   </layer>

   <layer ... type="Broadcast" ...>
       <data mode="explicit"/>
       <input>
           <port id="0">
               <dim>50</dim>
               <dim>50</dim>
          </port>
           <port id="1">
               <dim>4</dim>   <!--The tensor contains 4 elements: [1, 50, 50, 16] -->
           </port>
           <port id="1">
               <dim>2</dim>   <!--The tensor contains 2 elements: [1, 2] -->
           </port>
       </input>
       <output>
           <port id="2">
               <dim>1</dim>
               <dim>50</dim>
               <dim>50</dim>
               <dim>16</dim>
           </port>
       </output>
   </layer>


