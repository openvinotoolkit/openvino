GroupNormalization
==================


.. meta::
  :description: Learn about GroupNormalization-12 - a normalization operation,
                which can be performed on three required input tensors.

**Versioned name**: *GroupNormalization-12*

**Category**: *Normalization*

**Short description**: Performs normalization of the input tensor according to the method described in https://arxiv.org/abs/1803.08494

**Detailed description**

The GroupNormalization operation performs the following transformation of the input tensor:

.. math::

   y = scale * (x - mean) / sqrt(variance + epsilon) + bias

The operation is applied per batch, per group of channels. This means that the example input with ``N x C x H x W`` layout is transformed to the ``N x G x C/G x H x W`` form. The ``scale`` and ``bias`` coefficients are the inputs to the model and need to be specified separately for each channel. The ``mean`` and ``variance`` are calculated for each group.

**Attributes**

* *num_groups*

  * **Description**: Specifies the number of groups ``G`` that the channel dimension will be divided into.
  * **Range of values**: between ``1`` and the number of channels ``C`` in the input tensor
  * **Type**: ``int``
  * **Required**: *yes*

* *epsilon*

  * **Description**: A very small value added to the variance for numerical stability. Ensures that division by zero does not occur for any normalized element.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

**Inputs**

* **1**: ``data`` - The input tensor to be normalized. The type of this tensor is *T*. The tensor's shape is arbitrary but the first two dimensions are interpreted as ``batch`` and ``channels`` respectively. **Required.**

* **2**: ``scale`` - 1D tensor of type *T* containing the scale values for each channel. The expected shape of this tensor is ``[C]`` where ``C`` is the number of channels in the ``data`` tensor. **Required.**

* **3**: ``bias`` - 1D tensor of type *T* containing the bias values for each channel. The expected shape of this tensor is ``[C]`` where ``C`` is the number of channels in the ``data`` tensor. **Required.**

**Outputs**

* **1**: Output tensor of the same shape and type as the ``data`` input tensor.

**Types**

* *T*: any supported floating point type.

**Example**

.. code-block:: xml
   :force:

    <layer ... type="GroupNormalization">
        <data epsilon="1e-5" num_groups="4"/>
        <input>
            <port id="0">
                <dim>3</dim>
                <dim>12</dim>
                <dim>100</dim>
                <dim>100</dim>
            </port>
            <port id="1">
                <dim>12</dim> <!-- 12 scale values, 1 for each channel -->
            </port>
            <port id="2">
                <dim>12</dim> <!-- 12 bias values, 1 for each channel -->
            </port>
        </input>
        <output>
            <port id="3">
                <dim>3</dim>
                <dim>12</dim>
                <dim>100</dim>
                <dim>100</dim>
            </port>
        </output>
    </layer>



