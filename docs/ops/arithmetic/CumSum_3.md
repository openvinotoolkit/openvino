## CumSum <a name="CumSum"></a> {#openvino_docs_ops_arithmetic_CumSum_3}

**Versioned name**: *CumSum-3*

**Category**: Arithmetic unary operation 

**Short description**: *CumSum* performs cumulative summation of the input elements along the given axis.
 
**Detailed description**: By default, it will do the sum inclusively meaning the first element is copied as is. Through an "exclusive" attribute, this behavior can change to exclude the first element. It can also perform summation in the opposite direction of the axis. For that, set reverse attribute to `true`.

**Attributes**:

* *exclusive*

  * **Description**: If the attribute is set to `true` then an exclusive sum in which the top element is not included is returned. In other terms, if set to `true`, the `j-th` output element would be the sum of the first `(j-1)` elements. Otherwise, it would be the sum of the first `j` elements.
  * **Range of values**:
    * `false` - include the top element
    * `true` - do not include the top element
  * **Type**: `boolean`
  * **Default value**: `false`
  * **Required**: *no*

* *reverse*

  * **Description**: If set to `true` will perform the sums in reverse direction.
  * **Range of values**:
    * `false` - do not perform sums in reverse direction
    * `true` - perform sums in reverse direction
  * **Type**: `boolean`
  * **Default value**: `false`
  * **Required**: *no*

**Inputs**

* **1**: An tensor of type T. **Required.**

* **2**: Scalar axis of type T_AXIS. Negative value means counting dimensions from the back. Default value is 0. **Optional.**

**Outputs**

* **1**: Output tensor with cumulative sums of the input's elements. A tensor of type T of the same shape as 1st input.

**Types**

* *T*: any numeric type.

* *T_AXIS*: any integer number.

**Examples**

*Example 1*

```xml
<layer ... type="CumSum" exclusive="0" reverse="0">
    <input>
        <port id="0">     <!-- input value is: [1., 2., 3., 4., 5.] -->
            <dim>5</dim>
        </port>
        <port id="1"/>     <!-- axis value is: 0 -->
    </input>
    <output>
        <port id="2">     <!-- output value is: [1., 3., 6., 10., 15.] -->
            <dim>5</dim>
        </port>
    </output>
</layer>
```

*Example 2*

```xml
<layer ... type="CumSum" exclusive="1" reverse="0">
    <input>
        <port id="0">     <!-- input value is: [1., 2., 3., 4., 5.] -->
            <dim>5</dim>
        </port>
        <port id="1"/>     <!-- axis value is: 0 -->
    </input>
    <output>
        <port id="2">     <!-- output value is: [0., 1., 3., 6., 10.] -->
            <dim>5</dim>
        </port>
    </output>
</layer>
```

*Example 3*

```xml
<layer ... type="CumSum" exclusive="0" reverse="1">
    <input>
        <port id="0">     <!-- input value is: [1., 2., 3., 4., 5.] -->
            <dim>5</dim>
        </port>
        <port id="1"/>     <!-- axis value is: 0 -->
    </input>
    <output>
        <port id="2">     <!-- output value is: [15., 14., 12., 9., 5.] -->
            <dim>5</dim>
        </port>
    </output>
</layer>
```

*Example 4*

```xml
<layer ... type="CumSum" exclusive="1" reverse="1">
    <input>
        <port id="0">     <!-- input value is: [1., 2., 3., 4., 5.] -->
            <dim>5</dim>
        </port>
        <port id="1"/>     <!-- axis value is: 0 -->
    </input>
    <output>
        <port id="2">     <!-- output value is: [14., 12., 9., 5., 0.] -->
            <dim>5</dim>
        </port>
    </output>
</layer>
```
