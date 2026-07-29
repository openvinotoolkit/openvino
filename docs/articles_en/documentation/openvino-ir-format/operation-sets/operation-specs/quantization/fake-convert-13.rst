FakeConvert
===========


**Note**: FakeConvert is an experimental operation and subject to change.

.. meta::
  :description: Learn about FakeConvert-13 - a quantization operation.

**Versioned name**: *FakeConvert-13*

**Category**: *Quantization*

**Short description**: *FakeConvert* is element-wise quantization of floating-point input values into a set of values corresponding to a target low-precision floating-point type.

**Detailed description**: *FakeConvert* operation converts the input tensor to a specified target low-precision floating-point type and performs backward conversion to the source precision. It also applies affine transformation defined by ``scale`` and ``shift`` parameters before the conversion step and its reverse form after the backward conversion.

It emulates types defined by the ``destination_type`` attribute, on the original type of the ``data`` input.

Possible destination types are: "f8e4m3", "f8e5m2". The "f8e4m3" is an 8-bit floating-point format, where 1 bit for the sign, 4 bits for the exponents and 3 bits for the mantissa. The "f8e5m2" is an 8-bit floating-point format, where 1 bit is for the sign, 5 bits for the exponents and 2 for the mantissa.
The FP8 types were introduced in the following paper: `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`__ .

*Fake* in *FakeConvert* means that the output tensor preserve the same element type as an original type of the input tensor, not the ``destination_type``.

Each element of the output is defined as the result of the following expression:

.. code-block:: py
   :force:

    data = data * scale - shift
    ConvertLike(Convert(data, destination_type), data)
    data = (data + shift) / scale


**Attributes**

* *destination_type*

  * **Description**: *destination_type* is the emulated type
  * **Range of values**: "f8e4m3", "f8e5m2"
  * **Type**: `string`
  * **Required**: *yes*


**Inputs**:

* **1**: `data` - tensor of type *T_F* and arbitrary shape. **Required.**
* **2**: `scale` - tensor of type *T_F* with a scale factor for the *data* input value. The shape must be numpy-broadcastable to the shape of *data*. **Required.**
* **3**: `shift` - tensor of type *T_F* with value to subtract before and add after conversion of the *data* input value. The shape must be numpy-broadcastable to the shape of *data*, and match the shape of the *scale* input. **Optional.**


**Outputs**:

* **1**: Output tensor of type *T_F* with shape and type matching the 1st input tensor *data*.

**Types**

* *T_F*: supported floating-point type (`FP16`, `BF16`, `FP32`).

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
