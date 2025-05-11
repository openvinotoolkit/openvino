// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/convert_truncation.hpp"

ov::snippets::op::ConvertTruncation::ConvertTruncation(const Output<Node>& x, const ov::element::Type& destination_type)
    : ov::op::v0::Convert({x}, destination_type) {
}

std::shared_ptr<ov::Node> ov::snippets::op::ConvertTruncation::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(ConvertTruncation_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ConvertTruncation>(new_args.at(0), m_destination_type);
}
