CTCLoss
=======


.. meta::
  :description: Learn about CTCLoss-4 - a sequence processing operation, which
                can be performed on four required and one optional input tensor.

**Versioned name**: *CTCLoss-4*

**Category**: *Sequence processing*

**Short description**: *CTCLoss* computes the CTC (Connectionist Temporal Classification) Loss.

**Detailed description**:

*CTCLoss* operation is presented in `Connectionist Temporal Classification - Labeling Unsegmented Sequence Data with Recurrent Neural Networks: Graves et al., 2016 <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`__

*CTCLoss* estimates likelihood that a target ``labels[i,:]`` can occur (or is real) for given input sequence of logits ``logits[i,:,:]``. Briefly, *CTCLoss* operation finds all sequences aligned with a target ``labels[i,:]``, computes log-probabilities of the aligned sequences using ``logits[i,:,:]`` and computes a negative sum of these log-probabilities.

Input sequences of logits ``logits`` can have different lengths. The length of each sequence ``logits[i,:,:]`` equals ``logit_length[i]``.
A length of target sequence ``labels[i,:]`` equals ``label_length[i]``. The length of the target sequence must not be greater than the length of corresponding input sequence ``logits[i,:,:]``.
Otherwise, the operation behaviour is undefined.

*CTCLoss* calculation scheme:

1. Compute probability of ``j``-th character at time step ``t`` for ``i``-th input sequence from ``logits`` using softmax formula:

.. math::

   p_{i,t,j} = \frac{\exp(logits[i,t,j])}{\sum^{K}_{k=0}{\exp(logits[i,t,k])}}

2. For a given ``i``-th target from ``labels[i,:]`` find all aligned paths. A path ``S = (c1,c2,...,cT)`` is aligned with a target ``G=(g1,g2,...,gT)`` if both chains are equal after decoding. The decoding extracts substring of length ``label_length[i]`` from a target ``G``, merges repeated characters in ``G`` in case *preprocess_collapse_repeated* equal to true and finds unique elements in the order of character occurrence in case *unique* equal to true. The decoding merges repeated characters in ``S`` in case *ctc_merge_repeated* equal to true and removes blank characters represented by ``blank_index``. By default, ``blank_index`` is equal to ``C-1``, where ``C`` is a number of classes including the blank. For example, in case default *ctc_merge_repeated*, *preprocess_collapse_repeated*, *unique* and ``blank_index`` a target sequence ``G=(0,3,2,2,2,2,2,4,3)`` of a length ``label_length[i]=4`` is processed to ``(0,3,2,2)`` and a path ``S=(0,0,4,3,2,2,4,2,4)`` of a length ``logit_length[i]=9`` is also processed to ``(0,3,2,2)``, where ``C=5``. There exist other paths that are also aligned with ``G``, for instance, ``0,4,3,3,2,4,2,2,2``. Paths checked for alignment with a target ``label[:,i]`` must be of length ``logit_length[i] = L_i``. Compute probabilities of these aligned paths (alignments) as follows:

.. math::

   p(S) = \prod_{t=1}^{L_i} p_{i,t,ct}

3. Finally, compute negative log of summed up probabilities of all found alignments:

.. math::

   CTCLoss = - \ln \sum_{S} p(S)

**Note 1**: This calculation scheme does not provide steps for optimal implementation and primarily serves for better explanation.

**Note 2**: This is recommended to compute a log-probability :math:`\ln p(S)` for an aligned path as a sum of log-softmax of input logits. It helps to avoid underflow and overflow during calculation.
Having log-probabilities for aligned paths, log of summed up probabilities for these paths can be computed as follows:

.. math::

   \ln(a + b) = \ln(a) + \ln(1 + \exp(\ln(b) - \ln(a)))

**Attributes**

* *preprocess_collapse_repeated*

  * **Description**: *preprocess_collapse_repeated* is a flag for a preprocessing step before loss calculation, wherein repeated labels in ``labels[i,:]`` passed to the loss are merged into single labels.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

* *ctc_merge_repeated*

  * **Description**: *ctc_merge_repeated* is a flag for merging repeated characters in a potential alignment during the CTC loss calculation.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *unique*

  * **Description**: *unique* is a flag to find unique elements for a target ``labels[i,:]`` before matching with potential alignments. Unique elements in the processed ``labels[i,:]`` are sorted in the order of their occurrence in original ``labels[i,:]``. For example, the processed sequence for ``labels[i,:]=(0,1,1,0,1,3,3,2,2,3)`` of length ``label_length[i]=10`` will be ``(0,1,3,2)`` in case *unique* equal to true.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: false
  * **Required**: *no*

**Inputs**

* **1**: ``logits`` - Input tensor with a batch of sequences of logits. Type of elements is *T_F*. Shape of the tensor is ``[N, T, C]``, where ``N`` is the batch size, ``T`` is the maximum sequence length and ``C`` is the number of classes including the blank. **Required.**
* **2**: ``logit_length`` - 1D input tensor of type *T1* and of a shape ``[N]``. The tensor must consist of non-negative values not greater than ``T``. Lengths of input sequences of logits ``logits[i,:,:]``. **Required.**
* **3**: ``labels`` - 2D tensor with shape ``[N, T]`` of type *T2*. A length of a target sequence ``labels[i,:]`` is equal to ``label_length[i]`` and must contain of integers from a range ``[0; C-1]`` except ``blank_index``. **Required.**
* **4**: ``label_length`` - 1D tensor of type *T1* and of a shape ``[N]``. The tensor must consist of non-negative values not greater than ``T`` and ``label_length[i] <= logit_length[i]`` for all possible ``i``. **Required.**
* **5**: ``blank_index`` - Scalar of type *T2*. Set the class index to use for the blank label. Default value is ``C-1``. **Optional.**

**Output**

* **1**: Output tensor with shape ``[N]``, negative sum of log-probabilities of alignments. Type of elements is *T_F*.

**Types**

* *T_F*: any supported floating-point type.
* *T1*, *T2*: ``int32`` or ``int64``.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="CTCLoss" ...>
       <input>
           <port id="0">
               <dim>8</dim>
               <dim>20</dim>
               <dim>128</dim>
           </port>
           <port id="1">
               <dim>8</dim>
           </port>
           <port id="2">
               <dim>8</dim>
               <dim>20</dim>
           </port>
           <port id="3">
               <dim>8</dim>
           </port>
           <port id="4">  <!-- blank_index value is: 120 -->
       </input>
       <output>
           <port id="0">
               <dim>8</dim>
           </port>
       </output>
   </layer>


