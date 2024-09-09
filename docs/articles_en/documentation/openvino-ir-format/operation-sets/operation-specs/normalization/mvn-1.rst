MVN
===


.. meta::
  :description: Learn about MVN-1 - a normalization operation, which can be
                performed on a single input tensor.

**Versioned name**: *MVN-1*

**Category**: *Normalization*

**Short description**: Calculates mean-variance normalization of the input tensor. Supports two normalization techniques: `Instance/Contrast Normalization <https://arxiv.org/abs/1607.08022>`__ and `Layer Normalization <https://arxiv.org/abs/1607.06450>`__.

**Detailed description**

Based on ``across_channels`` attribute mean value is calculated using one of formulas below:

1. If ``true`` mean value is calculated using Layer Normalization:

   .. math::

      \mu_{n} = \frac{\sum_{c}^{C}\sum_{h}^{H}\sum_{w}^{W} i_{nchw}}{C * H * W}


2. If ``false`` mean value is calculated using Instance/Contrast Normalization:

   .. math::

      \mu_{nc} = \frac{\sum_{h}^{H}\sum_{w}^{W} i_{nchw}}{H * W}


where :math:`i_{nchw}` is an input tensor parametrized by :math:`n` batches, math:`c` channels and math:`h,w` spatial dimensions.

If ``reduction_axes`` attribute is provided mean value is calculated based on formula:

.. math::

   \mu_{n} = ReduceMean(i_{k}, reduction_axes)


Afterwards *MVN* subtracts mean value from the input blob.

If *normalize_variance* is set to ``true``, the output blob is divided by variance:

.. math::

   o_{i}=\frac{o_{i}}{\sqrt {\sum {\sigma_{k}^2}+\epsilon}}


where :math:`\sigma_{k}^2` is the variance calculated based on mean value, :math:`\epsilon` is a value added to the variance for numerical stability and corresponds to ``epsilon`` attribute.

**Attributes**

* *across_channels*

  * **Description**: *across_channels* is a flag that specifies whether mean values are shared across channels. If ``true`` mean values and variance are calculated for each sample across all channels and spatial dimensions (Layer Normalization), otherwise calculation is done for each sample and for each channel across spatial dimensions (Instance/Contrast Normalization).
  * **Range of values**:

    * ``false`` - do not share mean values across channels
    * ``true`` - share mean values across channels

  * **Type**: ``boolean``
  * **Required**: *yes*

* *reduction_axes*

  * **Description**: 1D tensor of unique elements and type *T_IND* which specifies indices of dimensions in ``data`` that define normalization slices. Negative value means counting dimensions from the back.
  * **Range of values**: allowed range of axes is ``[-r; r-1]`` where ``r = rank(data)``, the order cannot be sorted
  * **Type**: ``int``
  * **Required**: *yes*

* *normalize_variance*

  * **Description**: *normalize_variance* is a flag that specifies whether to perform variance normalization.
  * **Range of values**:

    * ``false`` - do not normalize variance
    * ``true`` - normalize variance

  * **Type**: ``boolean``
  * **Required**: *yes*

* *eps*

  * **Description**: *eps* is the number to be added to the variance to avoid division by zero when normalizing the value. For example, *epsilon* equal to 0.001 means that 0.001 is added to the variance.
  * **Range of values**: a positive floating-point number
  * **Type**: ``double``
  * **Required**: *yes*

*

  .. important::

     It is necessary to use only one of ``across_channels`` or ``reduction_axes`` attributes, they cannot be defined together.

**Inputs**

* **1**: ``data`` - input tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: normalized tensor of type *T* and shape as input tensor.

**Types**

* *T*: any floating point type.
* *T_IND*: ``int64`` or ``int32``.

**Examples**

*Example: with* ``across_channels`` *attribute*

.. code-block:: xml
   :force:

   <layer ... type="MVN">
       <data across_channels="true" eps="1e-9" normalize_variance="true"/>
       <input>
           <port id="0">
               <dim>6</dim>
               <dim>12</dim>
               <dim>10</dim>
               <dim>24</dim>
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


*Example: with* ``reduction_axes`` *attribute*

.. code-block:: xml
   :force:

   <layer ... type="MVN">
       <data reduction_axes="2,3" eps="1e-9" normalize_variance="true"/>
       <input>
           <port id="0">
               <dim>6</dim>
               <dim>12</dim>
               <dim>10</dim>
               <dim>24</dim>
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



