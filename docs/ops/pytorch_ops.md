# PyTorch Frontend Operators

## aten::vstack

**Description**: Stacks tensors in sequence vertically (row wise). All tensors need to be of the same size along all but the first dimension. 1-D tensors are treated as row vectors.

**Inputs**:
* **tensors**: A list of tensors to stack.

**Outputs**:
* **output**: The stacked tensor.

**Example**:
```python
import torch

# 1D tensors
x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([4, 5, 6])
result = torch.vstack((x1, x2))  # shape: (2, 3)

# 2D tensors
y1 = torch.tensor([[1, 2], [3, 4]])
y2 = torch.tensor([[5, 6], [7, 8]])
result = torch.vstack((y1, y2))  # shape: (4, 2)
```

## aten::hstack

**Description**: Stacks tensors in sequence horizontally (column wise). For 1-D tensors, it concatenates along dimension 0. For N-D tensors, it concatenates along dimension 1.

**Inputs**:
* **tensors**: A list of tensors to stack.

**Outputs**:
* **output**: The stacked tensor.

**Example**:
```python
import torch

# 1D tensors
x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([4, 5, 6])
result = torch.hstack((x1, x2))  # shape: (6,)

# 2D tensors
y1 = torch.tensor([[1, 2], [3, 4]])
y2 = torch.tensor([[5, 6], [7, 8]])
result = torch.hstack((y1, y2))  # shape: (2, 4)
``` 