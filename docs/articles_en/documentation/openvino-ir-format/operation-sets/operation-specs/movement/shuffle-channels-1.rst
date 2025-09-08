ShuffleChannels
===============


.. meta::
  :description: Learn about ShuffleChannels-1 - a data movement operation,
                which can be performed on a single input tensor.

**Versioned name**: *ShuffleChannels-1*

**Name**: *ShuffleChannels*

**Category**: *Data movement*

**Short description**: *ShuffleChannels* permutes data in the channel dimension of the input tensor.

**Detailed description**:

Input tensor of ``data_shape`` is always interpreted as 4D tensor with the following shape:

.. code-block:: cpp

    dim 0: data_shape[0] * data_shape[1] * ... * data_shape[axis-1]
             (or 1 if axis == 0)
    dim 1: group
    dim 2: data_shape[axis] / group
    dim 3: data_shape[axis+1] * data_shape[axis+2] * ... * data_shape[data_shape.size()-1]
            (or 1 if axis points to last dimension)


Trailing and leading to ``axis`` dimensions are flattened and reshaped back to the original shape after channels shuffling.


The operation is equivalent to the following transformation of the input tensor ``x`` of shape ``[N, C, H, W]`` and ``axis = 1``:

.. math::

    x' = reshape(x, [N, group, C / group, H * W])\\
    x'' = transpose(x', [0, 2, 1, 3])\\
    y = reshape(x'', [N, C, H, W])\\


where ``group`` is the layer attribute described below.

**Attributes**:

* *axis*

  * **Description**: *axis* specifies the index of a channel dimension.
  * **Range of values**: an integer number in the range ``[-rank(data_shape), rank(data_shape) - 1]``
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

* *group*

  * **Description**: *group* specifies the number of groups to split the channel dimension into. This number must evenly divide the channel dimension size.
  * **Range of values**: a positive integer in the range ``[1, data_shape[axis]]``
  * **Type**: ``int``
  * **Default value**: 1
  * **Required**: *no*

**Inputs**:

*   **1**: ``data`` input tensor of type *T* and rank greater or equal to 1. **Required.**

**Outputs**:

*   **1**: Output tensor with element type *T* and same shape as the input tensor.

**Types**

* *T*: any supported numeric type.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="ShuffleChannels" ...>
        <data group="3" axis="1"/>
        <input>
            <port id="0">
                <dim>5</dim>
                <dim>12</dim>
                <dim>200</dim>
                <dim>400</dim>
            </port>
        </input>
        <output>
            <port id="1">
                <dim>5</dim>
                <dim>12</dim>
                <dim>200</dim>
                <dim>400</dim>
            </port>
        </output>
    </layer>


