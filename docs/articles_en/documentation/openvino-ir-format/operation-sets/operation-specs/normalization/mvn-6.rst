MVN
===


.. meta::
  :description: Learn about MVN-6 - a normalization operation, which can be
                performed on two required input tensors.

**Versioned name**: *MVN-6*

**Category**: *Normalization*

**Short description**: Calculates mean-variance normalization of the input tensor.

**Detailed description**

*MVN* subtracts mean value from the input blob:

.. math::

   o_{i} = i_{i} - ReduceMean(i_{k}, axes)


If *normalize_variance* is set to ``true``, the output blob is divided by variance. When normalizing the value, the number ``eps`` is added to the variance to avoid division by zero. According to the ``eps_mode`` flag's value, ``eps`` is added inside or outside the sqrt:

* If ``eps_mode`` is ``inside_sqrt``:

  .. math::

     o_{i}=\frac{o_{i}}{\sqrt {\sum {o_{k}^2}+\epsilon}}


* If ``eps_mode`` is ``outside_sqrt``:

  .. math::

     o_{i}=\frac{o_{i}}{\sqrt {\sum {o_{k}^2}}+\epsilon}


**Attributes**

* *normalize_variance*

  * **Description**: *normalize_variance* is a flag that specifies whether to perform variance normalization.
  * **Range of values**:

    * ``false`` - do not normalize variance
    * ``true`` - normalize variance

  * **Type**: ``boolean``
  * **Required**: *yes*

* *eps*

  * **Description**: *eps* is the number to be added to the variance to avoid division by zero when normalizing the value.
  * **Range of values**: a positive floating-point number
  * **Type**: ``float``
  * **Required**: *yes*

* *eps_mode*

  * **Description**: Choose where to add epsilon.
  * **Range of values**:

    * ``inside_sqrt`` - add epsilon inside sqrt
    * ``outside_sqrt`` - add epsilon outside of sqrt

  * **Type**: ``string``
  * **Required**: *yes*

**Inputs**

* **1**: ``data`` - Input tensor to be normalized of type *T* and arbitrary shape. **Required.**

* **2**: ``axes`` - 1D tensor which specifies indices of dimensions in ``data`` that define normalization slices. Allowed range of axes is ``[-r; r-1]`` where ``r = rank(data)``, the order can be not sorted. Negative value means counting dimensions from the back. Type *T_IND*. **Required.**

**Outputs**

* **1**: Output tensor of the same shape and type as the ``data`` input tensor.

**Types**

* *T*: any floating point type.
* *T_IND*: ``int64`` or ``int32``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="MVN">
       <data eps="1e-9" eps_mode="inside_sqrt" normalize_variance="true"/>
       <input>
           <port id="0">
               <dim>6</dim>
               <dim>12</dim>
               <dim>10</dim>
               <dim>24</dim>
           </port>
           <port id="1">
               <dim>3</dim> <!-- value of [0,2,3] means independent normalization per channels -->
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



