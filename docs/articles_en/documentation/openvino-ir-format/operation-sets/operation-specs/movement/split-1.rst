Split
=====


.. meta::
  :description: Learn about Split-1 - a data movement operation,
                which can be performed on two required input tensors.

**Versioned name**: *Split-1*

**Category**: *Data movement*

**Short description**: *Split* operation splits an input tensor into pieces of the same length along some axis.

**Detailed Description**

*Split* operation splits a given input tensor ``data`` into chunks of the same length along a scalar ``axis``. It produces multiple output tensors based on *num_splits* attribute.
The i-th output tensor shape is equal to the input tensor ``data`` shape, except for dimension along ``axis`` which is ``data.shape[axis]/num_splits``.

.. math::

   shape\_output\_tensor = [data.shape[0], data.shape[1], \dotsc , data.shape[axis]/num\_splits, \dotsc data.shape[D-1]]


Where D is the rank of input tensor ``data``. The axis being split must be evenly divided by *num_splits* attribute.

**Attributes**

* *num_splits*

  * **Description**: number of outputs into which the input tensor ``data`` will be split along ``axis`` dimension. The dimension of ``data`` shape along ``axis`` must be evenly divisible by *num_splits*
  * **Range of values**: an integer within the range ``[1, data.shape[axis]]``
  * **Type**: ``int``
  * **Required**: *yes*

**Inputs**

* **1**: ``data``. A tensor of type *T* and arbitrary shape. **Required.**
* **2**: ``axis``. Axis along ``data`` to split. A scalar of type *T_AXIS* within the range ``[-rank(data), rank(data) - 1]``. Negative values address dimensions from the end. **Required.**
* **Note**: The dimension of input tensor ``data`` shape along ``axis`` must be evenly divisible by *num_splits* attribute.

**Outputs**

* **Multiple outputs**: Tensors of type *T*. The i-th output has the same shape as ``data`` input tensor except for dimension along ``axis`` which is ``data.shape[axis]/num_splits``.

**Types**

* *T*: any arbitrary supported type.
* *T_AXIS*: any integer type.

**Example**

.. code-block:: xml
   :force:

    <layer id="1" type="Split" ...>
        <data num_splits="3" />
        <input>
            <port id="0">       <!-- some data -->
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">       <!-- axis: 1 -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>6</dim>
                <dim>4</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="3">
                <dim>6</dim>
                <dim>4</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="4">
                <dim>6</dim>
                <dim>4</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
        </output>
    </layer>

