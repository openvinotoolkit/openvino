## MatMul <a name="MatMul"></a>

**Versioned name**: *MatMul-1*

**Category**: Matrix multiplication

**Short description**: Generalized matrix multiplication

**Detailed description**

*MatMul* operation takes two tensors and performs usual matrix-matrix multiplication, matrix-vector multiplication or vector-matrix multiplication depending on argument shapes. Input tensors can have any rank >= 1. Two right-most axes in each tensor are interpreted as matrix rows and columns dimensions while all left-most axes (if present) are interpreted as multi-dimensional batch: [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, ROW_INDEX_DIM, COL_INDEX_DIM]. The operation supports usual broadcast semantics for batch dimensions. It enables multiplication of batch of pairs of matrices in a single shot.

Before matrix multiplication, there is an implicit shape alignment for input arguments. It consists of the following steps:

1. If rank of an input less than 2 it is unsqueezed to 2D tensor by adding axes with size 1 to the left of the shape. For example, if input has shape `[S]` it will be reshaped to `[1, S]`. It is applied for each input independently.

2. Applied transpositions specified by optional `transpose_a` and `transpose_b` attributes.

3. If ranks of input arguments are different after steps 1 and 2, each is unsqueezed from the left side of the shape by necessary number of axes to make both shapes of the same rank.

3. Usual rules of the broadcasting are applied for batch dimensions.

Two attributes, transpose_a and transpose_b specifies embedded transposition for two right-most dimension for the first and the second input tensors correspondingly. It implies swapping of ROW_INDEX_DIM and COL_INDEX_DIM in the corresponding input tensor. Batch dimensions are not affected by these attributes.

**Attributes**

* *transpose_a*

  * **Description**: transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM of the 1st input; 0 means no transpose, 1 means transpose
  * **Range of values**: False or True
  * **Type**: boolean
  * **Default value**: False
  * **Required**: *no*

* *transpose_b*

  * **Description**: transposes dimensions ROW_INDEX_DIM and COL_INDEX_DIM of the 2nd input; 0 means no transpose, 1 means transpose
  * **Range of values**: False or True
  * **Type**: boolean
  * **Default value**: False
  * **Required**: *no*


**Inputs**:

*   **1**: Input batch of matrices A. Rank >= 1. Required.

*   **2**: Input batch of matrices B. Rank >= 1. Required.


**Example**

*Vector-matric multiplication*

```xml
<layer ... type="MatMul">
    <input>
        <port id="0">
            <dim>1024</dim>
        </port>
        <port id="1">
            <dim>1024</dim>
            <dim>1000</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>1000</dim>
        </port>
    </output>
</layer>
```

*Matrix-matrix multiplication (like FullyConnected with batch size 1)*

```xml
<layer ... type="MatMul">
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>1024</dim>
        </port>
        <port id="1">
            <dim>1024</dim>
            <dim>1000</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>1000</dim>
        </port>
    </output>
</layer>
```

*Matrix-vector multiplication with embedded transposition of the second matrix*

```xml
<layer ... type="MatMul">
    <data transpose_b="true"/>
    <input>
        <port id="0">
            <dim>1</dim>
            <dim>1024</dim>
        </port>
        <port id="1">
            <dim>1000</dim>
            <dim>1024</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>1</dim>
            <dim>1000</dim>
        </port>
    </output>
</layer>
```

*Matrix-matrix multiplication (like FullyConnected with batch size 10)*

```xml
<layer ... type="MatMul">
    <input>
        <port id="0">
            <dim>10</dim>
            <dim>1024</dim>
        </port>
        <port id="1">
            <dim>1024</dim>
            <dim>1000</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>10</dim>
            <dim>1000</dim>
        </port>
    </output>
</layer>
```

*Multiplication of batch of 5 matrices by a one matrix with broadcasting*

```xml
<layer ... type="MatMul">
    <input>
        <port id="0">
            <dim>5</dim>
            <dim>10</dim>
            <dim>1024</dim>
        </port>
        <port id="1">
            <dim>1024</dim>
            <dim>1000</dim>
        </port>
    </input>
    <output>
        <port id="2">
            <dim>5</dim>
            <dim>10</dim>
            <dim>1000</dim>
        </port>
    </output>
</layer>
```