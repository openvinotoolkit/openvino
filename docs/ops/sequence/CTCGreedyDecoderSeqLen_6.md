## CTCGreedyDecoderSeqLen <a name="CTCGreedyDecoderSeqLen"></a> {#openvino_docs_ops_sequence_CTCGreedyDecoderSeqLen_6}

**Versioned name**: *CTCGreedyDecoderSeqLen-6*

**Category**: Sequence processing

**Short description**: *CTCGreedyDecoderSeqLen* performs greedy decoding of the logits provided as the first input. The sequence lengths are provided as the second input.

**Detailed description**:

This operation is similar to TensorFlow CTCGreedyDecoder [TensorFlow CTCGreedyDecoder](https://www.tensorflow.org/api_docs/python/tf/nn/ctc_greedy_decoder).

The operation *CTCGreedyDecoderSeqLen* implements best path decoding.
Decoding is done in two steps:

1. Concatenate the most probable classes per time-step which yields the best path.

2. Remove duplicate consecutive elements if the attribute merge_repeated is true and then remove all blank elements.

Sequences in the batch can have different length. The lengths of sequences are coded in the second input integer tensor `sequence_length`.

The main difference between [CTCGreedyDecoder](CTCGreedyDecoder_1.md) and CTCGreedyDecoderSeqLen is in the second input. CTCGreedyDecoder uses 2D input floating point tensor with sequence masks for each sequence in the batch while CTCGreedyDecoderSeqLen uses 1D integer tensor with sequence lengths.

**Attributes**

* *merge_repeated*

  * **Description**: *merge_repeated* is a flag for merging repeated labels during the CTC calculation. If value is false the sequence `ABB*B*B`  (where '*' is the blank class) will look like `ABBBB`. But if we set it to true, the sequences will be `ABBB`. `ABB` will be merged to `AB`.
  * **Range of values**: true or false
  * **Type**: `boolean`
  * **Default value**: true
  * **Required**: *no*

**Inputs**

* **1**: `data` - input tensor of type *T_F* of shape `[T, N, C]` with a batch of sequences. Where `T` is the maximum sequence length, `N` is the batch size and `C` is the number of classes. **Required.**

* **2**: `sequence_length` - input tensor of type *T_I* of shape `[N]` with sequence lengths. The values of sequence length must be less or equal to `T`. **Required.**

* **3**: `blank_index` - Scalar of type *T_I*. Specifies the class index to use for the blank class. The `blank_index` is not saved to the result sequence and it is used for post-processing. Default value is `C-1`. **Optional**.

**Output**

* **1**: Output tensor of type *T_I* shape `[N * T, 2]` and containing final sequence class indices. A final sequence can be shorter than the size `T` of the tensor, all elements than do not code sequence classes are filled with -1.

* **2**: Output tensor of type *T_I* shape `[N, T]` and containing vector stores the decoded classes.

* **3**: Output tensor of type *T_I* shape `[2]` and containing shape of decoded classes vector.

* **4**: Output tensor of type *T_I* shape `[N]` and containing length of decoded class array for each batch.

**Types**

* *T_F*: any supported floating point type.

* *T_I*: `int32` or `int64`.

**Example**

```xml
<layer type="CTCGreedyDecoderSeqLen" merge_repeated="true">
    <input>
        <port id="0">
            <dim>20</dim>
            <dim>8</dim>
            <dim>128</dim>
        </port>
        <port id="1">
            <dim>8</dim>
        </port>
		<port id="2">
            <dim>1</dim>
        </port>
    </input>
    <output>
        <port id="0">
            <dim>160</dim>
            <dim>2</dim>
        </port>
		<port id="1">
            <dim>8</dim>
            <dim>20</dim>
        </port>
	    <port id="2">
            <dim>2</dim>
        </port>
		<port id="3">
            <dim>8</dim>
        </port>
    </output>
</layer>
```