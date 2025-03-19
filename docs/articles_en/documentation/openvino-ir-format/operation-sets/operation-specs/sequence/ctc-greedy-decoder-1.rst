CTCGreedyDecoder
================


.. meta::
  :description: Learn about CTCGreedyDecoder-1 - a sequence processing operation,
                which can be performed on two required input tensors.

**Versioned name**: *CTCGreedyDecoder-1*

**Category**: *Sequence processing*

**Short description**: *CTCGreedyDecoder* performs greedy decoding on the logits given in input (best path).

**Detailed description**: Given an input sequence :math:`X` of length :math:`T`, *CTCGreedyDecoder* assumes the probability of a length :math:`T` character sequence :math:`C` is given by

.. math::

   p(C|X) = \prod_{t=1}^{T} p(c_{t}|X)

Sequences in the batch can have different length. The lengths of sequences are coded as values 1 and 0 in the second input tensor ``sequence_mask``. Value ``sequence_mask[j, i]`` specifies whether there is a sequence symbol at index ``i`` in the sequence ``i`` in the batch of sequences. If there is no symbol at ``j``-th position ``sequence_mask[j, i] = 0``, and ``sequence_mask[j, i] = 1`` otherwise. Starting from ``j = 0``, ``sequence_mass[j, i]`` are equal to 1 up to the particular index ``j = last_sequence_symbol``, which is defined independently for each sequence ``i``. For ``j > last_sequence_symbol``, values in ``sequence_mask[j, i]`` are all zeros.

**Note**: Regardless of the value of ``ctc_merge_repeated`` attribute, if the output index for a given batch and time step corresponds to the ``blank_index``, no new element is emitted.

**Attributes**

* *ctc_merge_repeated*

  * **Description**: *ctc_merge_repeated* is a flag for merging repeated labels during the CTC calculation.
  * **Range of values**: ``true`` or ``false``
  * **Type**: ``boolean``
  * **Default value**: ``true``
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - input tensor with batch of sequences of type *T_F* and shape ``[T, N, C]``, where ``T`` is the maximum sequence length, ``N`` is the batch size and ``C`` is the number of classes. **Required.**
* **2**: ``sequence_mask`` - input tensor with sequence masks for each sequence in the batch of type *T_F* populated with values ``0`` and ``1`` and shape ``[T, N]``. **Required.**

**Output**

* **1**: Output tensor of type *T_F* and shape ``[N, T, 1, 1]`` which is filled with integer elements containing final sequence class indices. A final sequence can be shorter that the size ``T`` of the tensor, all elements that do not code sequence classes are filled with ``-1``.

**Types**

* *T_F*: any supported floating-point type.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="CTCGreedyDecoder" ...>
       <data ctc_merge_repeated="true" />
       <input>
           <port id="0">
               <dim>20</dim>
               <dim>8</dim>
               <dim>128</dim>
          </port>
           <port id="1">
               <dim>20</dim>
               <dim>8</dim>
           </port>
       </input>
       <output>
           <port id="0">
               <dim>8</dim>
               <dim>20</dim>
               <dim>1</dim>
               <dim>1</dim>
          </port>
       </output>
   </layer>


