# Broadcast Rules For Elementwise Operations {#openvino_docs_ops_broadcast_rules}

The purpose of this document is to provide a set of common rules which are applicable for ops using broadcasting.

## Description

Broadcast allows to perform element-wise operation for inputs of arbitrary number of dimensions. There are 2 types of broadcasts supported: Numpy and PDPD.

## Rules

**None broadcast**:
1. Input tensors dimensions must match.
2. No implicit broadcast rule is applied.

**Numpy broadcast**:
1. Right aligned dimensions of the two tensors are compared elementwise.
2. Smaller tensor is prepended with dimension(s) of size 1 in order to have the same shape as the larger tensor.
3. After alignment two tensors are compatible when both are equal or one of the dimensions is 1.
4. Tensor with dimension of size 1 will be implicitly broadcasted to match the size of the second tensor.
5. When both inputs are of rank = 0 the result is a scalar.

**PDPD broadcast**:
1. First input tensor A is of any rank, second input B has rank smaller or equal to the first input.
2. Input tensor B is a continuous subsequence of input A.
3. Apply broadcast B to match the shape of A, where provided *axis* is the start dimension index
   for broadcasting B onto A.
4. If *axis* is set to default (-1) calculate new value: `axis = rank(A) - rank(B)`.
5. The trailing dimensions of size 1 for input B will be ignored for the consideration of
   subsequence, such as `shape(B) = (3, 1) => (3)`.

## Numpy examples

*      `A: Shape(,) -> scalar` <br>
       `B: Shape(,) -> scalar` <br>
  `Result: Shape(,) -> scalar`

*     `A: Shape(2, 3)` <br>
      `B: Shape(   1)` <br>
 `Result: Shape(2, 3)`

*      `A: Shape(   3)` <br>
       `B: Shape(2, 3)` <br>
  `Result: Shape(2, 3)`

*      `A: Shape(2, 3, 5)` <br>
       `B: Shape(,) -> scalar` <br>
  `Result: Shape(2, 3, 5)`

*      `A: Shape(2, 1, 5)` <br>
       `B: Shape(1, 4, 5)` <br>
  `Result: Shape(2, 4, 5)`

*      `A: Shape(   6, 5)` <br>
       `B: Shape(2, 1, 5)` <br>
  `Result: Shape(2, 6, 5)`

*      `A: Shape(2, 1, 5)` <br>
       `B: Shape(   4, 1)` <br>
  `Result: Shape(2, 4, 5)` <br>

*      `A: Shape(3, 2, 1, 4)` <br>
       `B: Shape(      5, 4)` <br>
  `Result: Shape(3, 2, 5, 4)`

*      `A: Shape(   1, 5, 3)` <br>
       `B: Shape(5, 2, 1, 3)` <br>
  `Result: Shape(5, 2, 5, 3)`

*      `A: Shape(3)` <br>
       `B: Shape(2)` <br>
  `Result: broadcast won't happen due to dimensions mismatch`

*      `A: Shape(3, 1, 5)` <br>
       `B: Shape(4, 4, 5)` <br>
  `Result: broadcast won't happen due to dimensions mismatch on the leftmost axis`

## PDPD examples

*      `A: Shape(2, 3, 4, 5)` <br>
       `B: Shape(   3, 4   ) with axis = 1` <br>
  `Result: Shape(2, 3, 4, 5)`

*      `A: Shape(2, 3, 4, 5)` <br>
       `B: Shape(   3, 1   ) with axis = 1` <br>
  `Result: Shape(2, 3, 4, 5)`

*      `A: Shape(2, 3, 4, 5)` <br>
       `B: Shape(      4, 5) with axis=-1(default) or axis=2` <br>
  `Result: Shape(2, 3, 4, 5)`

*      `A: Shape(2, 3, 4, 5)` <br>
       `B: Shape(1, 3      ) with axis = 0` <br>
  `Result: Shape(2, 3, 4, 5)`

*      `A: Shape(2, 3, 4, 5)` <br>
       `B: Shape(,)` <br>
  `Result: Shape(2, 3, 4, 5)` <br>

*      `A: Shape(2, 3, 4, 5)` <br>
       `B: Shape(5,)` <br>
  `Result: Shape(2, 3, 4, 5)`

# Bidirectional Broadcast Rules {#openvino_docs_ops_bidirectional_broadcast_rules}

## Description

Bidirectional Broadcast is not intended for element-wise operations. Its purpose is to broadcast an array to a given shape.

## Rules

**Bidirectional broadcast**:
1. Dimensions of the input tensors are right alignment.
2. Following broadcast rule is applied: `numpy.array(input) * numpy.ones(target_shape)`.
3. Two corresponding dimension must have the same value, or one of them is equal to 1.
4. Output shape may not be equal to `target_shape` if:
   * `target_shape` contains dimensions of size 1,
   * `target_shape` rank is smaller than the rank of input tensor.

## Bidirectional examples

*      `A: Shape(5)` <br>
       `B: Shape(1)` <br>
  `Result: Shape(5)`

*      `A: Shape(2, 3)` <br>
       `B: Shape(   3)` <br>
  `Result: Shape(2, 3)`

*      `A: Shape(3, 1)` <br>
       `B: Shape(3, 4)` <br>
  `Result: Shape(3, 4)`

*      `A: Shape(3, 4)` <br>
       `B: Shape(,) -> scalar` <br>
  `Result: Shape(3, 4)`

*      `A: Shape(   3, 1)` <br>
       `B: Shape(2, 1, 6)` <br>
  `Result: Shape(2, 3, 6)`
