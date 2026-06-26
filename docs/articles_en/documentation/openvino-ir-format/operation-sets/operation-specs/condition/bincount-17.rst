Bincount
========


.. meta::
  :description: Learn about Bincount-17 - a condition operation that counts the
                number of occurrences of each value in a 1-D integer tensor in OpenVINO.

**Versioned name**: *Bincount-17*

**Category**: *Condition*

**Short description**: *Bincount* counts the number of occurrences of each non-negative integer value in a 1-D input tensor, producing a histogram-style output.

**Detailed description**

*Bincount* is equivalent to ``torch.bincount``. It counts the occurrences of each value in the 1-D ``data`` tensor and returns a 1-D output tensor where the element at index ``i`` equals the number of times the value ``i`` appears in ``data`` (or the sum of the corresponding ``weights`` when the optional ``weights`` input is provided).

The output length is ``max(max(data) + 1, minlength)``. When ``data`` is empty, the output length equals ``minlength``.

Because the output size depends on the maximum value in ``data``, the output shape is **data-dependent** and is not known statically in general. The static lower bound is ``minlength``.

**Attributes**

* *minlength*

  * **Description**: minimum length of the output tensor.
  * **Range of values**: a non-negative integer.
  * **Type**: ``int64``
  * **Default value**: 0
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - A 1-D tensor of type *T_DATA* containing non-negative integer values. **Required.**
* **2**: ``weights`` - A 1-D tensor of type *T_WEIGHTS* with the same length as ``data``. When provided, each occurrence of ``data[i]`` contributes ``weights[i]`` to the output instead of 1. **Optional.**

**Outputs**

* **1**: A 1-D tensor of type *T_OUT*. When ``weights`` is absent the type is ``i64``; otherwise the type matches *T_WEIGHTS*. The shape is ``[max(max(data) + 1, minlength)]``.

**Types**

* *T_DATA*: ``i32``, ``i64``, ``u8``, ``u16``, ``u32``, or ``u64``.
* *T_WEIGHTS*: ``f32``, ``f64``, ``i32``, or ``i64``.
* *T_OUT*: ``i64`` when ``weights`` is absent; *T_WEIGHTS* otherwise.

**Example 1: unweighted**

.. code-block:: xml
   :force:

   <layer ... type="Bincount" version="opset17">
       <data minlength="0"/>
       <input>
           <port id="0">
               <dim>10</dim>
           </port>
       </input>
       <output>
           <port id="1" precision="I64">
               <dim>-1</dim>
           </port>
       </output>
   </layer>

**Example 2: weighted with minlength**

.. code-block:: xml
   :force:

   <layer ... type="Bincount" version="opset17">
       <data minlength="5"/>
       <input>
           <port id="0">
               <dim>10</dim>
           </port>
           <port id="1" precision="F32">
               <dim>10</dim>
           </port>
       </input>
       <output>
           <port id="2" precision="F32">
               <dim>-1</dim>
           </port>
       </output>
   </layer>
