// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/utils.hpp"

#include "openvino/core/dimension_tracker.hpp"
#include "openvino/core/node.hpp"
#include "transformations/utils/utils.hpp"

bool ov::symbol::util::get_labels(const ov::PartialShape& shape, ov::TensorLabel& labels) {
    if (shape.rank().is_dynamic())
        return false;
    labels.clear();
    labels.reserve(shape.size());
    for (const auto& d : shape)
        labels.push_back((d.is_dynamic() ? ov::DimensionTracker::get_label(d) : ov::no_label));
    return true;
}

bool ov::symbol::util::get_labels(const ov::Output<ov::Node>& output, ov::TensorLabel& labels) {
    const auto& tensor = output.get_tensor();
    labels = tensor.get_value_label();
    return !labels.empty();
}

bool ov::symbol::util::are_unique_and_equal_labels(const ov::TensorLabel& lhs, const ov::TensorLabel& rhs) {
    if (rhs.size() != lhs.size() || rhs.empty())
        return false;
    for (size_t i = 0; i < lhs.size(); ++i)
        if (lhs[i] != rhs[i] || lhs[i] == ov::no_label)
            return false;
    return true;
}

bool ov::symbol::util::dims_are_equal(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    if (lhs.is_static() && lhs == rhs)
        return true;
    auto lhs_label = ov::DimensionTracker::get_label(lhs);
    auto rhs_label = ov::DimensionTracker::get_label(rhs);
    if (lhs_label == ov::no_label || rhs_label == ov::no_label)
        return false;
    if (lhs_label == rhs_label)
        return true;
    if (auto table_l = ov::DimensionTracker::get_table_of_equivalence(lhs))
        return table_l->are_equal(lhs, rhs);
    if (auto table_r = ov::DimensionTracker::get_table_of_equivalence(rhs))
        return table_r->are_equal(lhs, rhs);
    return false;
}
