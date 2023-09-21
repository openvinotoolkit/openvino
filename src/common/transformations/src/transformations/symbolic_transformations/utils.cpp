// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/utils.hpp"

#include <openvino/core/dimension_tracker.hpp>
#include <openvino/core/node.hpp>
#include <transformations/utils/utils.hpp>

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

bool labels_eq_or_eq_static_dims(const ov::Dimension& lhs, const ov::Dimension& rhs) {
    bool labels_exist_and_equal = false;

    auto lhs_label = ov::DimensionTracker::get_label(lhs);
    auto rhs_label = ov::DimensionTracker::get_label(rhs);
    auto table_l = ov::DimensionTracker::get_table_of_equivalence(lhs);
    auto table_r = ov::DimensionTracker::get_table_of_equivalence(rhs);
    if (table_l)
        labels_exist_and_equal = lhs_label != 0 && table_l->are_equal(lhs, rhs);
    else if (table_r)
        labels_exist_and_equal = lhs_label != 0 && table_r->are_equal(lhs, rhs);
    else
        labels_exist_and_equal = lhs_label != 0 && lhs_label == rhs_label;
    bool dims_are_static_and_equal = lhs.is_static() && lhs == rhs;
    return labels_exist_and_equal || dims_are_static_and_equal;
}

bool last_two_dims_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs) {
    if (lhs.rank().is_dynamic() || lhs.size() < 2)
        return false;
    if (rhs.rank().is_dynamic() || rhs.size() < 2)
        return false;
    for (size_t i = 2; i > 0; --i)
        if (!labels_eq_or_eq_static_dims(lhs[lhs.size() - i], rhs[rhs.size() - i]))
            return false;
    return true;
}

bool reshape_keeps_last_two_dims(const std::shared_ptr<ov::Node>& op) {
    return last_two_dims_are_equal(op->get_input_partial_shape(0), op->get_output_partial_shape(0));
}

bool batches_are_equal(const ov::PartialShape& lhs, const ov::PartialShape& rhs, bool one_dim_can_differ) {
    if (lhs.rank().is_dynamic() || rhs.rank().is_dynamic() || lhs.size() != rhs.size())
        return false;
    size_t num_dims_differ = 0;
    for (size_t i = 0; i < lhs.size() - 2; ++i)
        num_dims_differ += !labels_eq_or_eq_static_dims(lhs[i], rhs[i]);
    return num_dims_differ <= one_dim_can_differ;
}

bool batches_are_equal(const std::shared_ptr<ov::Node>& op_0, const std::shared_ptr<ov::Node>& op_1) {
    auto input_0 = op_0->get_input_partial_shape(0);
    auto input_1 = op_1->get_input_partial_shape(0);
    auto output_0 = op_0->get_output_partial_shape(0);
    auto output_1 = op_1->get_output_partial_shape(0);
    return batches_are_equal(input_0, input_1, true) && batches_are_equal(output_0, output_1);
}
