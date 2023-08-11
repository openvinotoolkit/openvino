// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/utils.hpp"

#include <openvino/core/bound_evaluation_util.hpp>
#include <openvino/core/dimension_tracker.hpp>
#include <openvino/core/node.hpp>
#include <transformations/utils/utils.hpp>

bool get_labels(const ov::PartialShape& shape, ov::TensorLabel& labels) {
    if (shape.rank().is_dynamic())
        return false;
    labels.clear();
    labels.reserve(shape.size());
    for (const auto& d : shape)
        labels.push_back((d.is_dynamic() ? ov::DimensionTracker::get_label(d) : ov::no_label));
    return true;
}

bool get_labels(const ov::Output<ov::Node>& output, ov::TensorLabel& labels) {
    const auto& tensor = output.get_tensor();
    labels = tensor.get_value_label();
    return !labels.empty();
}

bool are_unique_and_equal_labels(const ov::TensorLabel& lhs, const ov::TensorLabel& rhs) {
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

bool equalize_two_last_dims(const ov::PartialShape& from, ov::PartialShape& to) {
    if (from.rank().is_dynamic() || from.size() < 2 || to.rank().is_dynamic() || to.size() < 2)
        return false;
    for (size_t i = 2; i > 0; --i) {
        const auto& from_dim = from[from.size() - i];
        auto& to_dim = to[to.size() - i];
        if (from_dim.is_static() || to_dim.is_static())
            continue;
        auto from_label = ov::DimensionTracker::get_label(from_dim);
        if (from_label == ov::no_label)
            continue;
        ov::DimensionTracker::set_label(to_dim, from_label);
        auto from_table = ov::DimensionTracker::get_table_of_equivalence(from_dim);
        if (from_table)
            from_table->set_as_equal(from_dim, to_dim);
    }
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

ov::Output<ov::Node> get_shape_from_sources(const ov::Output<ov::Node>& batch_dims_source,
                                            const ov::Output<ov::Node>& non_batch_dims_source) {
    auto batch_indices = std::vector<size_t>(batch_dims_source.get_partial_shape().size() - 2);
    std::iota(batch_indices.begin(), batch_indices.end(), 0);
    auto batch_dims =
        ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(batch_dims_source, batch_indices);
    auto non_batch_indices = std::vector<size_t>(2);
    std::iota(non_batch_indices.begin(), non_batch_indices.end(), non_batch_dims_source.get_partial_shape().size() - 2);
    auto non_batch_dims =
        ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(non_batch_dims_source, non_batch_indices);
    auto target_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch_dims, non_batch_dims}, 0);
    return target_shape->output(0);
}

int64_t get_idx_of_label_in_source(const ov::Output<ov::Node>& source, const ov::label_t& label) {
    int64_t idx = -1;
    if (label == ov::no_label)
        return idx;
    auto pshape = source.get_partial_shape();
    auto rank = pshape.rank();
    if (rank.is_dynamic())
        return idx;
    for (int64_t i = 0; i < rank.get_length(); ++i) {
        auto l = ov::DimensionTracker::get_label(pshape[i]);
        if (l == label) {
            idx = i;
            break;
        }
    }
    return idx;
}

std::shared_ptr<ov::Node> get_node_representing_label_from_source_by_idx(const ov::Output<ov::Node>& source,
                                                                         const ov::element::Type& et,
                                                                         const ov::Shape& shape,
                                                                         const int64_t& idx) {
    if (idx == -1)
        return nullptr;  // -1 index doesn't mean counting from the back -- it means we didn't find where the label
    // comes from
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(source, et);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto indices = ov::op::v0::Constant::create(ov::element::i64, shape, {idx});
    auto gather = std::make_shared<ov::op::v8::Gather>(shape_of, indices, axis);
    evaluate_both_bounds(gather->output(0));
    return gather;
}

std::shared_ptr<ov::Node> get_node_representing_label_from_source_by_label(const ov::Output<ov::Node>& source,
                                                                           const ov::element::Type& et,
                                                                           const ov::Shape& shape,
                                                                           const ov::label_t& label) {
    return get_node_representing_label_from_source_by_idx(source, et, shape, get_idx_of_label_in_source(source, label));
}

// label to source map
using LTS_map = std::unordered_map<ov::label_t, ov::Output<ov::Node>>;

void optimize_value_usage(ov::Output<ov::Node>& output, LTS_map& label_shape_source, LTS_map& label_value_source) {
    auto value_labels = output.get_tensor().get_value_label();
    if (value_labels.size() != 1)
        return;
    auto label = value_labels[0];
    if (label == ov::no_label)
        return;
    auto pshape = output.get_partial_shape();
    if (pshape.is_dynamic() || ov::shape_size(pshape.to_shape()) != 1)
        return;
    auto shape = pshape.to_shape();  // scalar of some form of tensor with one element
    auto et = output.get_element_type();

    auto default_et = ov::element::i64;
    auto default_shape = ov::Shape{1};

    std::shared_ptr<ov::Node> alternative_source = nullptr;

    if (label_shape_source.count(label)) {
        auto source = label_shape_source[label];
        auto concat = ov::as_type_ptr<ov::op::v0::Concat>(source.get_node_shared_ptr());
        int64_t idx = get_idx_of_label_in_source(source, label);
        if (concat && idx != -1 && idx == concat->get_concatenation_axis() && concat->get_input_size() == 2) {
            // optimize using the knowledge of the Concat SI and what happens on the axis
            auto lhs_source = concat->input_value(0);
            const auto& lhs_pshape = lhs_source.get_partial_shape();
            auto rhs_source = concat->input_value(1);
            const auto& rhs_pshape = rhs_source.get_partial_shape();
            if (lhs_pshape.rank().is_static() && rhs_pshape.rank().is_static()) {
                auto lhs_label = ov::DimensionTracker::get_label(lhs_pshape[idx]);
                auto rhs_label = ov::DimensionTracker::get_label(rhs_pshape[idx]);
                std::shared_ptr<ov::Node> lhs_alternative = nullptr, rhs_alternative = nullptr;
                // get lhs_label from value or shape source
                if (label_value_source.count(lhs_label))
                    lhs_alternative = label_value_source[lhs_label].get_node_shared_ptr();
                if (!lhs_alternative && label_shape_source.count(lhs_label)) {
                    lhs_alternative = get_node_representing_label_from_source_by_label(label_shape_source[lhs_label],
                                                                                       default_et,
                                                                                       default_shape,
                                                                                       lhs_label);
                    if (lhs_alternative)
                        label_value_source[lhs_label] = lhs_alternative->output(0);
                }
                // get rhs_label from value or shape source
                if (label_value_source.count(rhs_label))
                    rhs_alternative = label_value_source[rhs_label].get_node_shared_ptr();
                if (!rhs_alternative && label_shape_source.count(rhs_label)) {
                    rhs_alternative = get_node_representing_label_from_source_by_label(label_shape_source[rhs_label],
                                                                                       default_et,
                                                                                       default_shape,
                                                                                       rhs_label);
                    if (rhs_alternative)
                        label_value_source[rhs_label] = rhs_alternative->output(0);
                }
                if (lhs_alternative && rhs_alternative) {
                    alternative_source = std::make_shared<ov::op::v1::Add>(lhs_alternative, rhs_alternative);
                    alternative_source->output(0).get_tensor().set_value_label({label});
                    label_value_source[label] = alternative_source->output(0);
                }
            }
        }
    }
    if (!alternative_source && label_value_source.count(label)) {
        auto value_source = label_value_source[label];
        alternative_source = value_source.get_node_shared_ptr();
    }
    if (!alternative_source && label_shape_source.count(label)) {
        // replacement via constructing the label source and saving it for the future
        alternative_source = get_node_representing_label_from_source_by_label(label_shape_source[label],
                                                                              default_et,
                                                                              default_shape,
                                                                              label);
        if (alternative_source)
            label_value_source[label] = alternative_source->output(0);
    }
    if (alternative_source != nullptr) {
        auto value_source = alternative_source->output(0);
        if (value_source.get_shape() != shape && (shape.empty() || shape == ov::Shape{0}))
            value_source = std::make_shared<ov::op::v0::Squeeze>(value_source);
        else if (value_source.get_shape() != shape)
            value_source = std::make_shared<ov::op::v1::Reshape>(
                value_source,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape),
                false);
        if (value_source.get_element_type() != et)
            value_source = std::make_shared<ov::op::v0::Convert>(value_source, et);
        evaluate_both_bounds(value_source);
        output.replace(value_source);
    } else {
        // in case we can not optimize it -- it is label which appeared just now on the value path
        label_value_source[label] = output;
    }
}

void save_shape_sources(const ov::Output<ov::Node>& output, LTS_map& label_shape_source) {
    auto shape = output.get_partial_shape();
    for (const auto& d : shape) {
        if (d.is_static())
            continue;
        auto label = ov::DimensionTracker::get_label(d);
        if (label == ov::no_label || label_shape_source.count(label))
            continue;
        label_shape_source[label] = output;
    }
}
