NormalizeL2
===========


.. meta::
  :description: Learn about MVN-1 - a normalization operation, which can be
                performed on two required input tensors.

**Versioned name**: *NormalizeL2-1*

**Category**: *Normalization*

**Short description**: *NormalizeL2* operation performs L2 normalization on a given input ``data`` along dimensions specified by ``axes`` input.

**Detailed Description**

Each element in the output is the result of dividing the corresponding element of ``data`` input by the result of L2 reduction along dimensions specified by the ``axes`` input:

.. code-block::  cpp

    output[i0, i1, ..., iN] = x[i0, i1, ..., iN] / sqrt(eps_mode(sum[j0,..., jN](x[j0, ..., jN]**2), eps))

Where indices ``i0, ..., iN`` run through all valid indices for the ``data`` input and summation ``sum[j0, ..., jN]`` has ``jk = ik`` for those dimensions ``k`` that are not in the set of indices specified by the ``axes`` input of the operation.
``eps_mode`` selects how the reduction value and ``eps`` are combined. It can be ``max`` or ``add`` depending on ``eps_mode`` attribute value.

Particular cases:

1. If ``axes`` is an empty list, then each input element is divided by itself resulting value ``1`` for all non-zero elements.
2. If ``axes`` contains all dimensions of input ``data``, a single L2 reduction value is calculated for the entire input tensor and each input element is divided by that value.


**Attributes**

* *eps*

  * **Description**: *eps* is the number applied by *eps_mode* function to the sum of squares to avoid division by zero when normalizing the value.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *eps_mode*

  * **Description**: Specifies how *eps* is combined with the sum of squares to avoid division by zero.
  * **Range of values**: ``add`` or ``max``
  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**

* **1**: ``data`` - A tensor of type *T* and arbitrary shape. **Required.**

* **2**: ``axes`` - Axis indices of ``data`` input tensor, along which L2 reduction is calculated. A scalar or 1D tensor of unique elements and type *T_IND*. The range of elements is ``[-r, r-1]``, where ``r`` is the rank of ``data`` input tensor. **Required.**

**Outputs**

* **1**: The result of *NormalizeL2* function applied to ``data`` input tensor. Normalized tensor of the same type and shape as the data input.

**Types**

* *T*: arbitrary supported floating-point type.
* *T_IND*: any supported integer type.

**Examples**

Example: Normalization over channel dimension for ``NCHW`` layout

.. code-block:: xml
   :force:

    <layer id="1" type="NormalizeL2" ...>
        <data eps="1e-8" eps_mode="add"/>
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">
                <dim>1</dim>         <!-- axes list [1] means normalization over channel dimension -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
        </output>
    </layer>


Example: Normalization over channel and spatial dimensions for ``NCHW`` layout

.. code-block:: xml
   :force:

    <layer id="1" type="NormalizeL2" ...>
        <data eps="1e-8" eps_mode="add"/>
        <input>
            <port id="0">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
            <port id="1">
                <dim>3</dim>         <!-- axes list [1, 2, 3] means normalization over channel and spatial dimensions -->
            </port>
        </input>
        <output>
            <port id="2">
                <dim>6</dim>
                <dim>12</dim>
                <dim>10</dim>
                <dim>24</dim>
            </port>
        </output>
    </layer>



