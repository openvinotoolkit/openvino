---
sidebar_position: 3
sidebar_label: "Step 1. Prerequisites Transformations"
description: "Learn about optional Prerequisites transformations, that prepare a model before applying other low precision transformations."
---

# Step 1. Prerequisites Transformations


Prerequisites transformations are optional. The transformations prepare a model before running other low precision transformations. The transformations do not operate with dequantization operations or update precisions. Prerequisites transformations include:

- [PullReshapeThroughDequantization](lin-op-sequence-fusion.md)
- [PullTransposeThroughDequantization](pull-transpose-through-dequantization.md)
- [LinOpSequenceFusion](lin-op-sequence-fusion.md)
- [ConvertSubtractConstant](convert-subtract-constant.md)
