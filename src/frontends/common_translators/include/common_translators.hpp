// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace common_translators {
#define COMMON_OP_CONVERTER(op) OutputVector op(const ov::frontend::NodeContext& node)

COMMON_OP_CONVERTER(translate_complex);
COMMON_OP_CONVERTER(translate_real);
COMMON_OP_CONVERTER(translate_imag);

COMMON_OP_CONVERTER(translate_atan2);
COMMON_OP_CONVERTER(translate_angle);
COMMON_OP_CONVERTER(translate_erfc);

COMMON_OP_CONVERTER(translate_equal);

OutputVector translate_atan2_util(const NodeContext& context, const Output<Node>& lhs, const Output<Node>& rhs);
OutputVector translate_erfc_util(const NodeContext& context, const Output<Node>& data);

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
