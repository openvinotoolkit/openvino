// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace detectron {
namespace validate {

/**
 * @brief Validates if all op's inputs have got same floating type and return inputs shapes and element type.
 *
 * @param op   Pointer to detector operator.
 * @return Input shapes and element type as pair.
 */
std::pair<std::vector<PartialShape>, element::Type> all_inputs_same_floating_type(const Node* const op);
}  // namespace validate
}  // namespace detectron
}  // namespace op
}  // namespace ov
