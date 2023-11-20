# FakeConvert {#openvino_docs_ops_quantization_FakeConvert_13}

**Note**: FakeConvert is an experimental operation and subject to change.

@sphinxdirective

.. meta::
  :description: Learn about FakeConvert-13 - a quantization operation.

**Versioned name**: *FakeConvert-13*

**Category**: *Quantization*

**Short description**: *FakeConvert* is element-wise emulation of float8 type on the original type of the data input.

**Detailed description**: *FakeConvert* operation emulates 8 bit floating-point type defined by the ``destination_type`` attribute, on the original type of the ``data`` input.
Possible destination types are: "f8e4m3", "f8e5m2". The "f8e4m3" is an fp8 type, where 1 bit for the sign, 4 bits for the exponents and 3 bits for the mantissa. In the "f8e5m2" format of fp8 type, there is 1 bit for the sign, 5 bits for the exponents and 2 for the mantissa.
The types were introduced in the following paper: `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`__ .

*Fake* in *FakeConvert* means the output tensor is of the same floating point type as an input tensor, not float8 type.


**Attributes**

* *destination_type*

  * **Description**: *destination_type* is the emulated type
  * **Range of values**: "f8e4m3", "f8e5m2"
  * **Type**: `string`
  * **Required**: *yes*


**Inputs**:

* **1**: `data` - tensor of type *T_F* and arbitrary shape. **Required.**
* **2**: `scale` - tensor of type *T_F* with a scale factor for the *data* input value. The shape must be broadcastable to the shape of *data*. **Required.**
* **3**: `shift` - tensor of type *T_F* with value to subtract before and add after conversion of the *data* input value. The shape must be broadcastable to the shape of *data*. **Optional.**


**Outputs**:

* **1**: Output tensor of type *T_F* with shape and type matching the 1st input tensor *data*.

**Types**

* *T_F*: supported floating-point type (`FP16` or `FP32`).

**Example**

.. code-block:: xml
   :force:

   <layer … type="FakeConvert"…>
       <data destination_type="f8e4m3"/>
       <input>
           <port id="0">
               <dim>1</dim>
               <dim>64</dim>
               <dim>56</dim>
               <dim>56</dim>
           </port>
           <port id="1">
               <dim>1</dim>
               <dim>64</dim>
               <dim>1</dim>
               <dim>1</dim>
           </port>
           <port id="2">
               <dim>1</dim>
               <dim>64</dim>
               <dim>1</dim>
               <dim>1</dim>
           </port>
       </input>
       <output>
           <port id="3">
               <dim>1</dim>
               <dim>64</dim>
               <dim>56</dim>
               <dim>56</dim>
           </port>
       </output>
   </layer>


@endsphinxdirective
