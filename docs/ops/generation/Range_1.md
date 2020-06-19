## Range<a name="Range"></a>

**Versioned name**: *Range-1*

**Category**: Generation

**Short description**: *Range* operation generates a sequence of numbers according input values [start, stop) with a step.

**Attributes**:

No attributes available.

**Inputs**:

* **1**: "start" - If a value is not given then *start* = 0. A scalar of type T. **Required.**
* **2**: "stop" - A scalar of type T. **Required.**
* **3**: "step" - If a value is not given then *step* = 1. A scalar of type T. **Required.**

**Outputs**:

* **1**: A tensor with type matching 2nd tensor.

**Types**

* *T*: any numeric type.

**Detailed description**:

*Range* operation generates a sequence of numbers starting from the value in the first input (start) up to but not including the value in the second input (stop) with a step equal to the value in the third input, according to the following formula:

\f[
[start, start + step, start + 2 * step, ..., start + K * step], where K is the maximal integer value that satisfies condition start + K*step < stop, then step is positive value and start + K*step > stop, then step is negative value.
\f]

**Examples**

*Example 1: positive step*

```xml
<layer ... type="Range">
    <input>
        <port id="0">  <!-- start value: 2 -->
        </port>
        <port id="1">  <!-- stop value: 23 -->
        </port>
        <port id="2">  <!-- step value: 3 -->
        </port>
    </input>
    <output>
        <port id="3">
            <dim>7</dim> <!-- [ 2,  5,  8, 11, 14, 17, 20] -->
        </port>
    </output>
</layer>
```

*Example 2: negative step*

```xml
<layer ... type="Range">
    <input>
        <port id="0">  <!-- start value: 23 -->
        </port>
        <port id="1">  <!-- stop value: 2 -->
        </port>
        <port id="2">  <!-- step value: -3 -->
        </port>
    </input>
    <output>
        <port id="3">
            <dim>7</dim> <!-- [23, 20, 17, 14, 11, 8, 5] -->
        </port>
    </output>
</layer>
```

