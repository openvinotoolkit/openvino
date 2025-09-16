// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/convert_truncation.hpp"

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"
#include "snippets/itt.hpp"

ov::snippets::op::ConvertTruncation::ConvertTruncation(const Output<Node>& x, const ov::element::Type& destination_type)
    : ov::op::v0::Convert({x}, destination_type) {}

ov::snippets::op::ConvertTruncation::ConvertTruncation(const Output<Node>& x,
                                                       const ov::element::Type& destination_type,
                                                       bool use_rounding)
    : ov::op::v0::Convert({x}, destination_type, false, use_rounding),
      m_use_rounding(use_rounding) {}

std::shared_ptr<ov::Node> ov::snippets::op::ConvertTruncation::clone_with_new_inputs(
    const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ConvertTruncation_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ConvertTruncation>(new_args.at(0), m_destination_type, m_use_rounding);
}
