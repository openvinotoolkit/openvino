## CTCGreedyDecoderSeqLen <a name="CTCGreedyDecoderSeqLen"></a> {#openvino_docs_ops_sequence_CTCGreedyDecoderSeqLen_6}

**Versioned name**: *CTCGreedyDecoderSeqLen-6*

**Category**: Sequence processing

**Short description**: *CTCGreedyDecoderSeqLen* performs greedy decoding on the logits given in input (best path).

**Detailed description**:

This operation is similar [Reference](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder)

Given an input sequence \f$X\f$ of length \f$T\f$, *CTCGreedyDecoderSeqLen* assumes the probability of a length \f$T\f$ character sequence \f$C\f$ is given by
\f[
p(C|X) = \prod_{t=1}^{T} p(c_{t}|X)
\f]

Sequences in the batch can have different length. The lengths of sequences are coded in the second input integer tensor `sequence_length`.

Operation different beetwine CTCGreedyDecoderSeqLen and CTCGreedyDecoder. 
The main diff is CTCGreedyDecoder use `sequence_mask` as second input. 2D input floating point tensor with sequence masks for each sequence in the batch. The lengths of sequences are coded as values 1 and 0. CTCGreedyDecoderSeqLen use 1D integer tensor with sequence lengths in second input.

**Attributes**

* *merge_repeated*

  * **Description**: *merge_repeated* is a flag for merging repeated labels during the CTC calculation.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: True
  * **Required**: *no*

**Inputs**

* **1**: `data` - Input tensor with a batch of sequences. Type of elements is any supported floating point type. Shape of the tensor is `[T, N, C]`, where `T` is the maximum sequence length, `N` is the batch size and `C` is the number of classes. A tensor of type *T_F*. **Required.**

* **2**: `sequence_length` - 1D input integer tensor with sequence lengths and having size batch. A tensor of type *T_I*. **Required.**

**Output**

* **1**: Output tensor with shape `[N, T]` and integer elements containing final sequence class indices. A final sequence can be shorter that the size `T` of the tensor, all elements that do not code sequence classes are filled with -1. Type of elements is floating point, but all values are integers. Type of elements is *T_F*.

**Types**

* *T_F*: any supported floating point type.

* *T_I*: `int32` or `int64`.

**Example**

```xml
<layer ... type="CTCGreedyDecoderSeqLen" ...>
    <input>
        <port id="0">
            <dim>20</dim>
            <dim>8</dim>
            <dim>128</dim>
       </port>
        <port id="1">
            <dim>8</dim>
        </port>
    </input>
    <output>
        <port id="0">
            <dim>8</dim>
            <dim>20</dim>
       </port>
    </output>
</layer>
```