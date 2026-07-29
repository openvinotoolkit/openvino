ReduceSum
=========


.. meta::
  :description: Learn about ReduceSum-1 - a reduction operation, which can be
                performed on two required input tensors.

**Versioned name**: *ReduceSum-1*

**Category**: *Reduction*

**Short description**: *ReduceSum* operation performs the reduction with addition on a given input ``data`` along dimensions specified by ``axes`` input.

**Detailed Description**

*ReduceSum* operation performs the reduction with addition on a given input ``data`` along dimensions specified by ``axes`` input.
Each element in the output is calculated as follows:

.. code-block:: cpp

    output[i0, i1, ..., iN] = sum[j0, ..., jN](x[j0, ..., jN]))

where indices i0, ..., iN run through all valid indices for input ``data``, and summation ``sum[j0, ..., jN]`` has ``jk = ik`` for those dimensions ``k`` that are not in the set of indices specified by ``axes`` input.

Particular cases:

1. If ``axes`` is an empty list, *ReduceSum* corresponds to the identity operation.
2. If ``axes`` contains all dimensions of input ``data``, a single reduction value is calculated for the entire input tensor.

**Attributes**

* *keep_dims*

  * **Description**: If set to ``true``, it holds axes that are used for the reduction. For each such axis, the output dimension is equal to 1.
  * **Range of values**: ``true`` or ``false``
  * **Type**: ``boolean``
  * **Default value**: ``false``
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - A tensor of type *T* and arbitrary shape. **Required.**

* **2**: ``axes`` - Axis indices of ``data`` input tensor, along which the reduction is performed. A scalar or 1D tensor of unique elements and type *T_IND*. The range of elements is ``[-r, r-1]``, where ``r`` is the rank of ``data`` input tensor. **Required.**

**Outputs**

* **1**: The result of *ReduceSum* function applied to ``data`` input tensor. A tensor of type *T* and ``shape[i] = shapeOf(data)[i]`` for all ``i`` dimensions not in ``axes`` input tensor. For dimensions in ``axes``, ``shape[i] == 1`` if ``keep_dims == true``; otherwise, the ``i``-th dimension is removed from the output.

**Types**

* *T*: any supported numeric type.
* *T_IND*: any supported integer type.

**Examples**

.. code-block:: xml
   :force:

    <layer id="1" type="ReduceSum" ...>
        <data keep_dims="true" />
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">
                <dim>2</dim>         <!-- value is [2, 3] that means independent reduction in each channel and batch -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>6</dim>
                <dim>12</dim>
                <dim>1</dim>
                <dim>1</dim>
            </port>
        </output>
    </layer>


.. code-block:: xml
   :force:

    <layer id="1" type="ReduceSum" ...>
        <data keep_dims="false" />
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">
                <dim>2</dim>         <!-- value is [2, 3] that means independent reduction in each channel and batch -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>6</dim>
                <dim>12</dim>
            </port>
        </output>
    </layer>


.. code-block:: xml
   :force:

    <layer id="1" type="ReduceSum" ...>
        <data keep_dims="false" />
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">
                <dim>1</dim>         <!-- value is [1] that means independent reduction in each channel and spatial dimensions -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>6</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
        </output>
    </layer>


.. code-block:: xml
   :force:

    <layer id="1" type="ReduceSum" ...>
        <data keep_dims="false" />
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">
                <dim>1</dim>         <!-- value is [-2] that means independent reduction in each channel, batch and second spatial dimension -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>6</dim>
                <dim>12</dim>
                <dim>24</dim>
            </port>
        </output>
    </layer>


