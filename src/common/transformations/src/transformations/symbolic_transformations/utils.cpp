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
    bool labels_exist_and_equal = false;

    auto lhs_label = ov::DimensionTracker::get_label(lhs);
    auto rhs_label = ov::DimensionTracker::get_label(rhs);
    auto table_l = ov::DimensionTracker::get_table_of_equivalence(lhs);
    auto table_r = ov::DimensionTracker::get_table_of_equivalence(rhs);
    if (table_l)
        labels_exist_and_equal = lhs_label != ov::no_label && table_l->are_equal(lhs, rhs);
    else if (table_r)
        labels_exist_and_equal = lhs_label != ov::no_label && table_r->are_equal(lhs, rhs);
    else
        labels_exist_and_equal = lhs_label != ov::no_label && lhs_label == rhs_label;
    bool dims_are_static_and_equal = lhs.is_static() && lhs == rhs;
    return labels_exist_and_equal || dims_are_static_and_equal;
}
