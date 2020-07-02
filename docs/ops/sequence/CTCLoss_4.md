## CTCLoss <a name="CTCLoss"></a>

**Versioned name**: *CTCLoss-4*

**Category**: Sequence processing

**Short description**: *CTCLoss* computes the CTC (Connectionist Temporal Classification) Loss.

**Detailed description**:

This operation is similar to the TensorFlow* operation [CTCLoss](https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/ctc_loss)

*CTCLoss* calculates loss between input and target sequences. It sums over the probability of possible alignments of input to target.

Input sequences `inputs[:,i,:]` can have different length. The lengths of sequences are coded as values 1 and 0 in the second input tensor `sequence_mask`. 
Value `sequence_mask[j, i]` specifies whether there is a sequence symbol at index `j` in the sequence `i` in the batch of sequences. 
If there is no symbol at `j`-th position `sequence_mask[j, i] = 0`, and `sequence_mask[j, i] = 1` otherwise. 
Starting from `j = 0`, `sequence_mass[j, i]` are equal to 1 up to the particular index `j = last_sequence_symbol`, 
which is defined independently for each sequence `i`. For `j > last_sequence_symbol`, values in `sequence_mask[j, i]` are all zeros.
A length of target sequence from `labels[i,:]` must not be greater than a lenght of corresponding input sequence. Otherwise, the operation behaviour is undefined.

*CTCLoss* calculation scheme:

1. Compute probability of `j`-th character at time step `t` for `i`-th input sequence from `inputs` using softmax formula:
\f[
p_{t,i,j} = \frac{\exp(inputs[t,i,j])}{\sum^{K}_{k=0}{\exp(inputs[t,i,k])}}
\f]

2. For a given `i`-th target from `labels[i,:]` find all aligned paths.
A path `S = (c1,c2,...,cT)` is aligned with a target `G=(g1,g2,...,gT)` if both chains are equal after decoding.
The decoding removes pads `-1` from a target `G` and merges repeated characters in `G` in case *preprocess_collapse_repeated* equal to True.
The decoding merges repeated characters in `S` in case *merge_repeated* equal to True and removes blank characters represented by last class index `K-1`, 
where `K` is a number of classes including a blank symbol.
For example, in case default *merge_repeated* and *preprocess_collapse_repeated* a target sequence `(0,3,2,2,-1,-1,-1,-1,-1)` is processed to `(0,3,2,2)` and
an input `(0,0,4,3,2,2,4,2,4)` is also processed to `(0,3,2,2)`, where `K=5`.
Compute probabilities of these alignments as follows:
\f[
p(S) = \prod_{t=1}^{T} p_{t,i,ct}
\f]

3. Finally sum up probabilities of all aligned paths for a given target and compute negative logarithm of it:
\f[
CTCLoss = -\log \sum_{S} p(S)
\f]


**Attributes**

* *preprocess_collapse_repeated*

  * **Description**: *preprocess_collapse_repeated* is a flag for a preprocessing step before loss calculation, wherein repeated labels passed to the loss are merged into single labels.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: False
  * **Required**: *no*

* *merge_repeated*

  * **Description**: *merge_repeated* is a flag for merging repeated labels during the CTC loss calculation.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: True
  * **Required**: *no*

**Inputs**

* **1**: `inputs` - Input tensor with a batch of sequences. Type of elements is *T_F*. Shape of the tensor is `[T, N, C]`, where `T` is the maximum sequence length, `N` is the batch size and `C` is the number of classes. Required.

* **2**: `sequence_mask` - 2D input tensor of type *T_IND* with sequence masks for each sequence in the batch. Populated with values 0 and 1. Shape of this input is `[T, N]`. Required.

* **3**: `labels` - 2D tensor with shape `[N, T]` of type *T_IND*. A sequence can be shorter than the size `T` of the tensor, all elements that do not code sequence classes are filled with -1. Required.

**Output**

* **1**: Output tensor with shape `[N]`, negative log of summed up probabilities for aligned paths. Type of elements is *T_F*.

**Types**

* *T_F*: any supported floating point type.

* *T_IND*: any supported integer type.

**Example**

```xml
<layer ... type="CTCLoss" ...>
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
        <port id="2">
            <dim>8</dim>
            <dim>20</dim>
       </port>
    </input>
    <output>
        <port id="0">
            <dim>8</dim>
       </port>
    </output>
</layer>
```
