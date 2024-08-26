// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

// Previous opsets operators
_OPENVINO_OP_REG(Parameter, ov::op::v0)
_OPENVINO_OP_REG(Convert, ov::op::v0)
_OPENVINO_OP_REG(ShapeOf, ov::op::v3)

// New operations added in opset15
_OPENVINO_OP_REG(ROIAlignRotated, ov::op::v15)
_OPENVINO_OP_REG(ScatterNDUpdate, ov::op::v15)
_OPENVINO_OP_REG(EmbeddingBagPacked, ov::op::v15)
_OPENVINO_OP_REG(EmbeddingBagOffsets, ov::op::v15)
_OPENVINO_OP_REG(Col2Im, ov::op::v15)
_OPENVINO_OP_REG(StringTensorUnpack, ov::op::v15)
_OPENVINO_OP_REG(StringTensorPack, ov::op::v15)
_OPENVINO_OP_REG(BitwiseLeftShift, ov::op::v15)
_OPENVINO_OP_REG(BitwiseRightShift, ov::op::v15)
_OPENVINO_OP_REG(SliceScatter, ov::op::v15)
