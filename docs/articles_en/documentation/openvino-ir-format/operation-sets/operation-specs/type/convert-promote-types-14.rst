ConvertPromoteTypes
===================


.. meta::
  :description: Learn about ConvertPromoteTypes-14 - type conversion that promotes pair of input tensors to common datatype.

**Versioned name**: *ConvertPromoteTypes-14*

**Category**: *Type conversion*

**Short description**: *ConvertPromoteTypes* operation performs promotion and conversion of ``input_0`` and ``input_1`` to common datatype based on promotion rules.

**Detailed description**
Operation performs datatype promotion for a pair of inputs, returning pair of outputs that represent input tensors converted to common type.

Promotion rules were designed to follow behavior of PyTorch and TensorFlow with experimental NumPy behavior enabled.

If inputs have different type of data (for example, ``floating-point`` and ``integer``), resulting datatype is taken from input with higher type priority,
where ``floating-point`` types have higher priority than ``integer``, and ``integer`` have higher priority than ``boolean``.

    .. note::
        If *promote_unsafe* is set to ``false``, to mitigate possible issue with loss of precision or undefined behaviors caused by difference in maximum/minimum values supported by given data types,
        in conversions from ``integer`` to ``floating-point``, conversion will fail if ``floating-point`` bit-width would be less than double of ``integer``.

If both inputs have same type of data (for example, both are ``integer`` with any bit-width and sign), resulting datatype is chosen to be of same type of data with bit-width
and sign to hold all possible values supported by input data types, except when used with *pytorch_scalar_promotion*.

* In case where *pytorch_scalar_promotion* is set to ``true``, one of inputs is scalar-tensor (rank 0) and second input is dimensioned tensor (any rank other than 0), datatype of the dimensioned tensor would be selected as a result common type, which may result in undefined behaviors if type of scalar input would support greater range of values than tensor input.

* In case of ``floating-point`` types, resulting type is type with lowest bit-width of mantissa and exponential to fit mantissa and exponential of input types. This may result in unexpected bit widening in conversions like ``(bf16, f16) -> f32``. Conversion of ``(f8e4m3, f8e5m2) -> f16`` is a special case where conversion result was manually set to ``f16``, however it could be promoted to either bf16 and f16 based on mantissa and exponential based on rules.

* In case of ``integer`` types, resulting type is an ``integer``, signed when any of inputs is signed and with minimal bit-width to hold all possible values supported by input data types.  In case of promotion of signed and unsigned ``integers``, resulting datatype would be a signed ``integer`` with bit-width of at least double than unsigned input to be able to store possible maximum and minimum values of unsigned one. Exception is for u64 and any signed ``integers`` promotion - since it would result in unsupported by OpenVINO type ``i128``, outcome of this promotion can be set by *u64_integer_promotion_target* attribute, by default set to ``f32``.

    .. note::
        If *promote_unsafe* is set to ``false``, promotions that will introduce bit widening,
        promotions of u64 with any signed ``integer``, or promotion causing conversion to type with lower range of values, exceptions will be raised.

.. note::
    Promotion rules does not depend on order of inputs or values contained within tensors. Shape of tensors may affect type only when *pytorch_scalar_promotion* is set to ``true`` and both inputs have same type priority.

Examples (notation: ``ConvertPromoteTypes(lhs_type, rhs_type) -> promoted_common_type``):

* Regular promotions with attributes set to default values:

    * ``ConvertPromoteTypes(i8, f32) -> f32`` - floating has higher priority than integer, bit-width of 32 is more than double of 8, minimizing impact of precision loss.
    * ``ConvertPromoteTypes(i32, u8) -> i32`` - both types of the same priority, signed integer has enough bit-width to represent all data of unsigned one.

* Promotions that will cause exceptions when *promote_unsafe* will be set to ``false``:

    * ``ConvertPromoteTypes(f16, i64) -> f16`` - Floating-point type has higher priority, however, i64 can support values outside of range of f16, possibly resulting in undefined behaviors in conversion.
    * ``ConvertPromoteTypes(f64, u64) -> f64`` - While f64 supports much bigger max values than u64, precision loss might be significant.
    * ``ConvertPromoteTypes(i8, u8) -> i16`` - Both inputs have integer data type, however, to support ranges from both inputs, bit-widening was necessary.
    * ``ConvertPromoteTypes(f16, bf16) -> f32`` - Both inputs have same data type, however, due to difference in mantissa and exponential, bit-widening to f32 is necessary to represent whole range and precision. This is in accordance of IEE 754.
    * ``ConvertPromoteTypes(f8m4e3, f8m5e3) -> f16`` - Both inputs have f8 data type, however, due to difference in mantissa and exponential, bit widening to either f16 or bf16 is necessary, where f16 was selected as result of this promotion.
    * ``ConvertPromoteTypes(u64, i8) -> f32`` - promotion of u64 and any signed integer would result in i128, which is not supported. Common type is set according to *u64_integer_promotion_target*, default f32.

* Promotions for PyTorch-like mode with *pytorch_scalar_promotion* set to ``true``. Notation is extended by ``S(type)`` marking 0-dimensioned (scalar) tensor, and ``D(type)`` marking dimensioned tensor.

    * ``ConvertPromoteTypes(S(i64), D(u8)) -> u8`` - Inputs have same data type, promote to type of dimensioned input. Rules of safe promotion (controlled by *promote_unsafe*) apply in Pytorch-like conversion.
    * ``ConvertPromoteTypes(S(f16), D(i8)) -> f16`` - Inputs have mixed data types - follow general conversion rules, dimensions of inputs does not affect common type.

**Attributes**:

* *promote_unsafe*

  * **Description**: allow for promotions that might result in bit-widening, significant precision loss and undefined behaviors. When false, exceptions will be raised.
  * **Range of values**: true or false
  * **Type**: ``bool``
  * **Default value**: false
  * **Required**: *no*

* *pytorch_scalar_promotion*

  * **Description**: if true, when scalar and dimensioned tensor with the same type priority (both either floating-point or integers) are provided as inputs, align datatype to dimensioned one.
  * **Range of values**: true or false
  * **Type**: ``bool``
  * **Default value**: false
  * **Required**: *no*

* *u64_integer_promotion_target*

  * **Description**: promotion target for promotion of u64 and any signed integer inputs.
  * **Range of values**: any element type supported by Convert operator.
  * **Type**: ``element::Type``
  * **Default value**: element::f32
  * **Required**: *no*

**Inputs**

* **1**: ``input_0`` - A tensor of type *T1* and arbitrary shape. **Required.**
* **2**: ``input_1`` - A tensor of type *T2* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of *ConvertPromoteTypes* operation applied to input tensor ``input_0``. A tensor of type *T_OUT* and the same shape as ``input_0`` input tensor.
* **2**: The result of *ConvertPromoteTypes* operation applied to input tensor ``input_1``. A tensor of type *T_OUT* and the same shape as ``input_1`` input tensor.

**Types**

* *T1*: any supported type.
* *T2*: any supported type.
* *T_OUT*: Result of type promotion for given input.

**Example 1: Promote floats**

.. code-block:: xml
   :force:

    <layer ... type="ConvertPromoteTypes">
        <data promote_unsafe="false" pytorch_scalar_promotion="false" u64_integer_promotion_target="f32"/>
        <input>
            <port id="0" precision="FP16">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="1" precision="FP32">
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="FP32", names="ConvertPromoteTypes:0">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="3" precision="FP32", names="ConvertPromoteTypes:1">
                <dim>3</dim>
            </port>
        </output>
    </layer>

**Example 2: Promote integers unsafe**

.. code-block:: xml
   :force:

    <layer ... type="ConvertPromoteTypes">
        <data promote_unsafe="true" pytorch_scalar_promotion="false" u64_integer_promotion_target="f32"/>
        <input>
            <port id="0" precision="I16">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="1" precision="U32">
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="I64", names="ConvertPromoteTypes:0">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="3" precision="I64", names="ConvertPromoteTypes:1">
                <dim>3</dim>
            </port>
        </output>
    </layer>

**Example 3: Promote u64 and signed integer unsafe**

.. code-block:: xml
   :force:

    <layer ... type="ConvertPromoteTypes">
        <data promote_unsafe="true" pytorch_scalar_promotion="false" u64_integer_promotion_target="f32"/>
        <input>
            <port id="0" precision="I16">
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="1" precision="U64">
                <dim>3</dim>
            </port>
        </input>
        <output>
            <port id="2" precision="FP32", names="ConvertPromoteTypes:0">  < !-- type provided by u64_integer_promotion_target -->
                <dim>256</dim>
                <dim>56</dim>
            </port>
            <port id="3" precision="FP32", names="ConvertPromoteTypes:1">  < !-- type provided by u64_integer_promotion_target -->
                <dim>3</dim>
            </port>
        </output>
    </layer>
