## CTCGreedyDecoder <a name="CTCGreedyDecoder"></a>

**Versioned name**: *CTCGreedyDecoder-1*

**Category**: Sequence processing

**Short description**: *CTCGreedyDecoder* performs greedy decoding on the logits given in input (best path).

**Detailed description**:

This operation is similar [Reference](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder)

Given an input sequence \f$X\f$ of length \f$T\f$, *CTCGreedyDecoder* assumes the probability of a length \f$T\f$ character sequence \f$C\f$ is given by
\f[
p(C|X) = \prod_{t=1}^{T} p(c_{t}|X)
\f]

Sequences in the batch can have different length. The lengths of sequences are coded as values 1 and 0 in the second input tensor `sequence_mask`. Value `sequence_mask[j, i]` specifies whether there is a sequence symbol at index `i` in the sequence `i` in the batch of sequences. If there is no symbol at `j`-th position `sequence_mask[j, i] = 0`, and `sequence_mask[j, i] = 1` otherwise. Starting from `j = 0`, `sequence_mass[j, i]` are equal to 1 up to the particular index `j = last_sequence_symbol`, which is defined independently for each sequence `i`. For `j > last_sequence_symbol`, values in `sequence_mask[j, i]` are all zeros.

**Attributes**

* *merge_repeated*

  * **Description**: *merge_repeated* is a flag for merging repeated labels during the CTC calculation.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: True
  * **Required**: *no*

**Inputs**

* **1**: `data` - Input tensor with a batch of sequences. Type of elements is any supported floating point type. Shape of the tensor is `[T, N, C]`, where `T` is the maximum sequence length, `N` is the batch size and `C` is the number of classes. Required.

* **2**: `sequence_mask` - 2D input floating point tensor with sequence masks for each sequence in the batch. Populated with values 0 and 1. Shape of this input is `[T, N]`. Required.

**Output**

* **1**: Output tensor with shape `[N, T, 1, 1]` and integer elements containing final sequence class indices. A final sequence can be shorter that the size `T` of the tensor, all elements that do not code sequence classes are filled with -1. Type of elements is floating point, but all values are integers.

**Example**

```xml
<layer ... type="CTCGreedyDecoder" ...>
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
```