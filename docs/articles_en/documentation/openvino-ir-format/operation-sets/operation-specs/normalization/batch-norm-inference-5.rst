BatchNormInference
==================


.. meta::
  :description: Learn about BatchNormInference-5 - a normalization operation, which can be
                performed on five required input tensors.

**Versioned name**: *BatchNormInference-5*

**Category**: *Normalization*

**Short description**: *BatchNormInference* performs Batch Normalization operation described in the `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167v2>`__ article.

**Detailed Description**

*BatchNormInference* performs the following operations on a given data batch input tensor ``data``:

* Normalizes each activation :math:`x^{(k)}` by the mean and variance.

  .. math::

     \hat{x}^{(k)}=\frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var(x^{(k)}) + \epsilon}}

  where :math:`E[x^{(k)}]` and :math:`Var(x^{(k)})` are the mean and variance, calculated per channel axis of ``data`` input, and correspond to ``mean`` and ``variance`` inputs, respectively. Additionally, :math:`\epsilon` is a value added to the variance for numerical stability and corresponds to ``epsilon`` attribute.

* Performs linear transformation of each normalized activation based on ``gamma`` and ``beta`` input, representing the scaling factor and shift, respectively.

  .. math::

     \hat{y}^{(k)}=\gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}

  where :math:`\gamma^{(k)}` and :math:`\beta^{(k)}` are learnable parameters, calculated per channel axis, and correspond to ``gamma`` and ``beta`` inputs.

**Mathematical Formulation**

Let ``x`` be a *d*-dimensional input, :math:`x=(x_{1}\dotsc x_{d})`. Since normalization is applied to each activation :math:`E[x^{(k)}]`, you can focus on a particular activation and omit k.

For a particular activation, consider a mini-batch :math:`\mathcal{B}` of m values. *BatchNormInference* performs Batch Normalization algorithm as follows:

* **Input**: Values of :math:`x` over a mini-batch:

  .. math::

     \mathcal{B} = {x_{1...m}}

* **Parameters to learn**: :math:`\gamma, \beta`
* **Output**:

  .. math::

     {o_{i} = BN_{\gamma, \beta} ( b_{i} )}

* **Mini-batch mean**:

  .. math::

     \mu_{\mathcal{B}} \leftarrow \frac{1}{m}\sum_{i=1}^{m}b_{i}

* **Mini-batch variance**:

  .. math::

     \sigma_{\mathcal{B}}^{2}\leftarrow \frac{1}{m}\sum_{i=1}^{m} ( b_{i} - \mu_{\mathcal{B}})^{2}

* **Normalize**:

  .. math::

     \hat{b_{i}} \leftarrow \frac{b_{i} - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^{2} + \epsilon }}

* **Scale and shift**:

  .. math::

     o_{i} \leftarrow \gamma\hat{b_{i}} + \beta = BN_{\gamma ,\beta } ( b_{i} )


**Attributes**:

* *epsilon*

  * **Description**: *epsilon* is a constant added to the variance for numerical stability.
  * **Range of values**: a floating-point number greater than or equal to zero
  * **Type**: ``float``
  * **Required**: *yes*

**Inputs**

* **1**: ``data`` - A tensor of type *T* and at least rank 2. The second dimension represents the channel axis and must have a span of at least 1. **Required.**
* **2**: ``gamma`` - Scaling factor for normalized value. A 1D tensor of type *T* with the same span as ``data`` channel axis. **Required.**
* **3**: ``beta`` - Bias added to the scaled normalized value. A 1D tensor of type *T* with the same span as ``data`` channel axis. **Required.**
* **4**: ``mean`` - Value for mean normalization. A 1D tensor of type *T* with the same span as ``data`` channel axis. **Required.**
* **5**: ``variance`` - Value for variance normalization. A 1D tensor of type *T* with the same span as ``data`` channel axis. **Required.**

**Outputs**

* **1**: The result of element-wise Batch Normalization operation applied to the input tensor ``data``. A tensor of type *T* and the same shape as ``data`` input tensor.

**Types**

* *T*: any supported floating-point type.

**Examples**

Example: 2D input tensor ``data``

.. code-block:: xml
   :force:

   <layer ... type="BatchNormInference" ...>
       <data epsilon="9.99e-06" />
       <input>
           <port id="0">  <!-- input -->
               <dim>10</dim>
               <dim>128</dim>
           </port>
           <port id="1">  <!-- gamma -->
               <dim>128</dim>
           </port>
           <port id="2">  <!-- beta -->
               <dim>128</dim>
           </port>
           <port id="3">  <!-- mean -->
               <dim>128</dim>
           </port>
           <port id="4">  <!-- variance -->
               <dim>128</dim>
           </port>
       </input>
       <output>
           <port id="5">
               <dim>10</dim>
               <dim>128</dim>
           </port>
       </output>
   </layer>

Example: 4D input tensor ``data``

.. code-block:: xml
   :force:

   <layer ... type="BatchNormInference" ...>
       <data epsilon="9.99e-06" />
       <input>
           <port id="0">  <!-- input -->
               <dim>1</dim>
               <dim>3</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
           <port id="1">  <!-- gamma -->
               <dim>3</dim>
           </port>
           <port id="2">  <!-- beta -->
               <dim>3</dim>
           </port>
           <port id="3">  <!-- mean -->
               <dim>3</dim>
           </port>
           <port id="4">  <!-- variance -->
               <dim>3</dim>
           </port>
       </input>
       <output>
           <port id="5">
               <dim>1</dim>
               <dim>3</dim>
               <dim>224</dim>
               <dim>224</dim>
           </port>
       </output>
   </layer>


