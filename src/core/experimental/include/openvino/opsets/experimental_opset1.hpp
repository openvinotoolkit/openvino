// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/core_visibility.hpp"
#include "openvino/opsets/opset.hpp"

namespace ov::op::experimental {

/// @brief Access the experimental opset1 instance containing BlockSparseAttention and DraftBlockMask.
OPENVINO_API const ov::OpSet& get_opset1();

}  // namespace ov::op::experimental