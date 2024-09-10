CTCGreedyDecoderSeqLen
======================


.. meta::
  :description: Learn about CTCGreedyDecoderSeqLen-6 - a sequence processing
                operation, which can be performed on two required input tensors.

**Versioned name**: *CTCGreedyDecoderSeqLen-6*

**Category**: *Sequence processing*

**Short description**: *CTCGreedyDecoderSeqLen* performs greedy decoding of the logits provided as the first input. The sequence lengths are provided as the second input.

**Detailed description**:

This operation is similar to the `TensorFlow CTCGreedyDecoder <https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder>`__.

The operation *CTCGreedyDecoderSeqLen* implements best path decoding.
Decoding is done in two steps:

1. Concatenate the most probable classes per time-step which yields the best path.

2. Remove duplicate consecutive elements if the attribute *merge_repeated* is true and then remove all blank elements.

Sequences in the batch can have different length. The lengths of sequences are coded in the second input integer tensor ``sequence_length``.

The main difference between :doc:`CTCGreedyDecoder <ctc-greedy-decoder-1>` and CTCGreedyDecoderSeqLen is in the second input. CTCGreedyDecoder uses 2D input floating-point tensor with sequence masks for each sequence in the batch while CTCGreedyDecoderSeqLen uses 1D integer tensor with sequence lengths.

**Attributes**

* *merge_repeated*

  * **Description**: *merge_repeated* is a flag for merging repeated labels during the CTC calculation. If the value is false the sequence ``ABB*B*B``  (where '*' is the blank class) will look like ``ABBBB``. But if the value is true, the sequence will be ``ABBB``.
  * **Range of values**: true or false
  * **Type**: ``boolean``
  * **Default value**: true
  * **Required**: *no*

* *classes_index_type*

  * **Description**: the type of output tensor with classes indices
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i32"
  * **Required**: *no*

* *sequence_length_type*

  * **Description**: the type of output tensor with sequence length
  * **Range of values**: "i64" or "i32"
  * **Type**: string
  * **Default value**: "i32"
  * **Required**: *no*

**Inputs**

* **1**: ``data`` - input tensor of type *T_F* of shape ``[N, T, C]`` with a batch of sequences. Where ``T`` is the maximum sequence length, ``N`` is the batch size and ``C`` is the number of classes. **Required.**
* **2**: ``sequence_length`` - input tensor of type *T_I* of shape ``[N]`` with sequence lengths. The values of sequence length must be less or equal to ``T``. **Required.**
* **3**: ``blank_index`` - scalar or 1D tensor with 1 element of type *T_I*. Specifies the class index to use for the blank class. Regardless of the value of ``merge_repeated`` attribute, if the output index for a given batch and time step corresponds to the ``blank_index``, no new element is emitted. Default value is `C-1`. **Optional.**

**Output**

* **1**: Output tensor of type *T_IND1* shape ``[N, T]`` and containing the decoded classes. All elements that do not code sequence classes are filled with -1.
* **2**: Output tensor of type *T_IND2* shape ``[N]`` and containing length of decoded class sequence for each batch.

**Types**

* *T_F*: any supported floating-point type.
* *T_I*: ``int32`` or ``int64``.
* *T_IND1*: ``int32`` or ``int64`` and depends on ``classes_index_type`` attribute.
* *T_IND2*: ``int32`` or ``int64`` and depends on ``sequence_length_type`` attribute.

**Example**

.. code-block:: xml
   :force:

   <layer ... type="CTCGreedyDecoderSeqLen" version="opset6">
       <data merge_repeated="true" classes_index_type="i64" sequence_length_type="i64"/>
       <input>
           <port id="0">
               <dim>8</dim>
               <dim>20</dim>
               <dim>128</dim>
           </port>
           <port id="1">
               <dim>8</dim>
           </port>
           <port id="2"/>  <!-- blank_index = 120 -->
       </input>
       <output>
           <port id="0" precision="I64">
               <dim>8</dim>
               <dim>20</dim>
           </port>
           <port id="1" precision="I64">
               <dim>8</dim>
           </port>
       </output>
   </layer>


