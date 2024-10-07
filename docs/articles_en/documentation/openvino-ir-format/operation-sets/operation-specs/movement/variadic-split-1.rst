VariadicSplit
=============


.. meta::
  :description: Learn about VariadicSplit-1 - a data movement operation, which can be
                performed on three required input tensors.

**Versioned name**: *VariadicSplit-1*

**Category**: *Data movement*

**Short description**: *VariadicSplit* operation splits an input tensor into chunks along some axis. The chunks may have variadic lengths depending on ``split_lengths`` input tensor.

**Detailed Description**

*VariadicSplit* operation splits a given input tensor `data` into chunks along a scalar or tensor with shape ``[1]`` ``axis``. It produces multiple output tensors based on additional input tensor ``split_lengths``.
The i-th output tensor shape is equal to the input tensor `data` shape, except for dimension along `axis` which is ``split_lengths[i]``.

.. math::

   shape\_output\_tensor = [data.shape[0], data.shape[1], \dotsc , split\_lengths[i], \dotsc , data.shape[D-1]]

Where D is the rank of input tensor `data`. The sum of elements in ``split_lengths`` must match ``data.shape[axis]``.

**Attributes**: *VariadicSplit* operation has no attributes.

**Inputs**

* **1**: ``data``. A tensor of type `T1` and arbitrary shape. **Required.**
* **2**: ``axis``. Axis along ``data`` to split. A scalar or tensor with shape ``[1]`` of type ``T2`` with value from range ``-rank(data) .. rank(data)-1``. Negative values address dimensions from the end. **Required.**
* **3**: ``split_lengths``. A list containing the dimension values of each output tensor shape along the split ``axis``. A 1D tensor of type ``T2``. The number of elements in ``split_lengths`` determines the number of outputs. The sum of elements in ``split_lengths`` must match ``data.shape[axis]``. In addition ``split_lengths`` can contain a single ``-1`` element, which means, all remaining items along specified ``axis`` that are not consumed by other parts. **Required.**

**Outputs**

* **Multiple outputs**: Tensors of type ``T1``. The i-th output has the same shape as `data` input tensor except for dimension along ``axis`` which is ``split_lengths[i]`` if ``split_lengths[i] != -1``. Otherwise, the dimension along ``axis`` is processed as described in ``split_lengths`` input description.

**Types**

* *T1*: any arbitrary supported type.
* *T2*: any integer type.

**Examples**

.. code-block:: xml
   :force:

    <layer id="1" type="VariadicSplit" ...>
        <input>
            <port id="0">            <!-- some data -->
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">            <!-- axis: 0 -->
            </port>
            <port id="2">
                <dim>3</dim>         <!-- split_lengths: [1, 2, 3] -->
            </port>
        </input>
        <output>
            <port id="3">
                <dim>1</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="4">
                <dim>2</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="5">
                <dim>3</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
        </output>
    </layer>


.. code-block:: xml
   :force:

    <layer id="1" type="VariadicSplit" ...>
        <input>
            <port id="0">            <!-- some data -->
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">            <!-- axis: 0 -->
            </port>
            <port id="2">
                <dim>2</dim>         <!-- split_lengths: [-1, 2] -->
            </port>
        </input>
        <output>
            <port id="3">
                <dim>4</dim>         <!--  4 = 6 - 2  -->
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="4">
                <dim>2</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
        </output>
    </layer>



