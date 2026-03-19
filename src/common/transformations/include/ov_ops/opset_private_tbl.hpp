// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _OPENVINO_OP_REG
#    warning "_OPENVINO_OP_REG not defined"
#    define _OPENVINO_OP_REG(x, y)
#endif

// Ensure BevPoolV2 type is visible to the registration macros
#include "openvino/op/bevpool_v2.hpp"

_OPENVINO_OP_REG(AUGRUCell, ov::op::internal)
_OPENVINO_OP_REG(AUGRUSequence, ov::op::internal)
_OPENVINO_OP_REG(RMS, ov::op::internal)
_OPENVINO_OP_REG(BevPoolV2, ov::op::v15)
