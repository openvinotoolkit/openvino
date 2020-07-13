## CTCLoss <a name="CTCLoss"></a>

**Versioned name**: *CTCLoss-4*

**Category**: Sequence processing

**Short description**: *CTCLoss* computes the CTC (Connectionist Temporal Classification) Loss.

**Detailed description**:

*CTCLoss* operation is presented in [Connectionist Temporal Classification - Labeling Unsegmented Sequence Data with Recurrent Neural Networks: Graves et al., 2016](http://www.cs.toronto.edu/~graves/icml_2006.pdf)

*CTCLoss* estimates a chance that a target can occur (or is real) for given input sequence of logits.
Briefly, *CTCLoss* operation finds all sequences aligned with a target sequence `labels[i,:]`, computes log-probabilities of these aligned sequences using `inputs[:,i,:]` of logits
and computes a negative sum of these log-probabilies.

Input sequences of logits from `inputs` can have different length. The length of each sequence `inputs[:,i,:]` equals `input_length[i]`.
A length of target sequence from `labels[i,:]` is determined by a pad `-1`. The length must not be greater than a lenght of corresponding input sequence `inputs[:,i,:]`.
Otherwise, the operation behaviour is undefined.

*CTCLoss* calculation scheme:

1. Compute probability of `j`-th character at time step `t` for `i`-th input sequence from `inputs` using softmax formula:
\f[
p_{t,i,j} = \frac{\exp(inputs[t,i,j])}{\sum^{K}_{k=0}{\exp(inputs[t,i,k])}}
\f]

2. For a given `i`-th target from `labels[i,:]` find all aligned paths.
A path `S = (c1,c2,...,cT)` is aligned with a target `G=(g1,g2,...,gT)` if both chains are equal after decoding.
The decoding removes pads `-1` from a target `G` and merges repeated characters in `G` in case *preprocess_collapse_repeated* equal to True.
The decoding merges repeated characters in `S` in case *ctc_merge_repeated* equal to True and removes blank characters represented by `blank_index`.
By default, `blank_index` is equal to `C-1`, where `C` is a number of classes including the blank.
For example, in case default *ctc_merge_repeated*, *preprocess_collapse_repeated*, *unique*, and `blank_index` a target sequence `(0,3,2,2,-1,-1,-1,-1,-1)` is processed to `(0,3,2,2)` and
a path `(0,0,4,3,2,2,4,2,4)` is also processed to `(0,3,2,2)`, where `C=5`. There exist other paths that are also aligned, for instance, `0,4,3,3,2,4,2,2,2`.
Paths checked for alignment with a target `label[:,i]` must be of length `input_length[i] = L_i`.
Compute probabilities of these aligned paths (alignments) as follows:
\f[
p(S) = \prod_{t=1}^{L_i} p_{t,i,ct}
\f]

3. Finally, compute negative sum of log-probabilities of all alignments:
\f[
CTCLoss = \minus \sum_{S} \ln p(S)
\f]


**Attributes**

* *preprocess_collapse_repeated*

  * **Description**: *preprocess_collapse_repeated* is a flag for a preprocessing step before loss calculation, wherein repeated labels passed to the loss are merged into single labels.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: False
  * **Required**: *no*

* *ctc_merge_repeated*

  * **Description**: *ctc_merge_repeated* is a flag for merging repeated labels during the CTC loss calculation.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: True
  * **Required**: *no*

* *unique*

  * **Description**: *unique* is a flag to find unique elements for a target `labels[i,:]` before matching with potential alignments.
  * **Range of values**: True or False
  * **Type**: `boolean`
  * **Default value**: False
  * **Required**: *no*

**Inputs**

* **1**: `inputs` - Input tensor with a batch of sequences. Type of elements is *T_F*. Shape of the tensor is `[T, N, C]`, where `T` is the maximum sequence length, `N` is the batch size and `C` is the number of classes including the blank. Required.

* **2**: `input_length` - 1D input tensor of type *T_IND* and of a shape `[N]`. The tensor must consist of values not greater than `T`. Lengths of input sequences `inputs[:,i,:]`. Required.

* **3**: `labels` - 2D tensor with shape `[N, T]` of type *T_IND*. A sequence can be shorter than the size `T` of the tensor, all elements that do not code sequence classes are filled with -1. Required.

* **4**: `blank_index` - Scalar of type *T_IND*. Set the class index to use for the blank label. Default value is `C-1`. Optional.

**Output**

* **1**: Output tensor with shape `[N]`, negative sum of log-probabilities of alignments. Type of elements is *T_F*.

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
        <port id="3">
    </input>
    <output>
        <port id="0">
            <dim>8</dim>
        </port>
    </output>
</layer>
```
