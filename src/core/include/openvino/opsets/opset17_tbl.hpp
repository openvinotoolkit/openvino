// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

// Previous opsets operators
// TODO (ticket: 179247): Add remaining operators from opset16 at the end of opset17 development
_OPENVINO_OP_REG(Parameter, ov::op::v0)
_OPENVINO_OP_REG(Convert, ov::op::v0)
_OPENVINO_OP_REG(ShapeOf, ov::op::v3)

// New operations added in opset17
