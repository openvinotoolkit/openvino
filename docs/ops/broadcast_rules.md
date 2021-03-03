# Broadcast Rules For Elementwise Operations {#openvino_docs_ops_broadcast_rules}

The purpose of this document is to provide a set of common rules which are applicable for ops using broadcasting.

## Description

Broadcast allows to perform element-wise operation for inputs with different shapes. Smaller input is *stretched* into an array with the same shape as the larger input. As a result both inputs have compatible shapes.

## Rules

**Numpy broadcast**:
1. Right aligned dimensions of the two tensors are compared elementwise.
2. Smaller tensor is prepended with dimension(s) of size 1 in order to have the same shape as the larger tensor.
3. After alignment two tensors are compatible when:
   * both are equal or
   * one of the dimensions is 1.
4. Tensor with dimension of size 1 will be implicitly broadcasted to match the size of the second tensor.
5. When both inputs are of rank = 0 the result is a scalar.

**Bidirectional broadcast**:
1. Dimensions of the input tensors are right alignment.
2. Following broadcast rule is applied: `numpy.array(input) * numpy.ones(target_shape)`.
3. Two corresponding dimension must have the same value, or one of them is equal to 1.
4. Output shape may not be equal to *desired* shape if:
   * *desired* shape contains dimensions of size 1,
   * *desired* shape rank is smaller than the rank of input tensor.

## Examples

*      `A: Shape() -> scalar`
       `B: Shape() -> scalar`
  `Result: Shape() -> scalar`

*      `A: Shape(2, 3)`
       `B: Shape(   1)`
  `Result: Shape(2, 3)`

*      `A: Shape(   3)`
       `B: Shape(2, 3)`
  `Result: Shape(2, 3)`

*      `A: Shape(2, 3, 5)`
       `B: Shape() -> scalar`
  `Result: Shape(2, 3, 5)`

*      `A: Shape(2, 1, 5)`
       `B: Shape(1, 4, 5)`
  `Result: Shape(2, 4, 5)`

*      `A: Shape(   6, 5)`
       `B: Shape(2, 1, 5)`
  `Result: Shape(2, 6, 5)`

*      `A: Shape(2, 1, 5)`
       `B: Shape(   4, 1)`
  `Result: Shape(2, 4, 5)`

*      `A: Shape(3, 2, 1, 4)`
       `B: Shape(      5, 4)`
  `Result: Shape(3, 2, 5, 4)`

*      `A: Shape(   1, 5, 3)`
       `B: Shape(5, 2, 1, 3)`
  `Result: Shape(5, 2, 5, 3)`

*      `A: Shape(3)`
       `B: Shape(2)`
  `Result: broadcast won't happen due to dimensions mismatch`

*      `A: Shape(3, 1, 5)`
       `B: Shape(4, 4, 5)`
  `Result: broadcast won't happen due to dimensions mismatch on the leftmost axis`
