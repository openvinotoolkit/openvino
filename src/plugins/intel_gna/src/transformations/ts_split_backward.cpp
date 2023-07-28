// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/ts_split_backward.hpp"

#include <openvino/cc/ngraph/itt.hpp>

#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::intel_gna::graph_utils;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;
using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::graph_utils;

namespace {

bool is_split_sinked(const Output<Node>& output) {
    auto split_node = output.get_node_shared_ptr();
    for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
        for (auto& input : split_node->get_output_target_inputs(output_idx)) {
            auto target_node =
                get_next_node_skipping_certain(input.get_node()->shared_from_this(), is_gna_non_functional_node);
            std::shared_ptr<ov::Node> transpose = ov::as_type_ptr<Transpose>(target_node);
            if (transpose && !Limitations::is_transpose_supported(transpose))
                return true;
        }
    }
    return false;
}
}  // namespace

TSSplitBackward::TSSplitBackward() {
    MATCHER_SCOPE(TSSplitBackward);

    auto split_node_label = wrap_type<Split, VariadicSplit>(is_split_sinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& split_node_label_output = pattern_to_output.at(split_node_label);
        std::shared_ptr<ov::Node> split_node = as_type_ptr<Split>(split_node_label_output.get_node_shared_ptr());
        if (!split_node) {
            split_node = as_type_ptr<VariadicSplit>(split_node_label_output.get_node_shared_ptr());
        }

        ov::AxisVector gather_ids = {};
        std::vector<AxisVector> gather_indices_vecs;
        std::vector<size_t> split_slices;
        for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
            for (auto& input : split_node->get_output_target_inputs(output_idx)) {
                auto target_node =
                    get_next_node_skipping_certain(input.get_node()->shared_from_this(), is_gna_non_functional_node);
                std::shared_ptr<ov::Node> transpose = ov::as_type_ptr<Transpose>(target_node);

                ov::Shape transpose_shape = split_node->get_output_shape(output_idx);
                ov::AxisVector transpose_order(transpose_shape.size());

                if (transpose) {
                    std::shared_ptr<Constant> transpose_const =
                        ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
                    transpose_order = transpose_const->get_axis_vector_val();
                    transpose_shape = transpose->get_input_shape(0);
                } else {
                    std::iota(transpose_order.begin(), transpose_order.end(), 0);
                }

                ov::AxisVector slice_ids =
                    graph_utils::make_gather_indexes_from_transpose_axes(transpose_shape, transpose_order);
                // shift slice indexes and insert at the end of the gather indexes vector
                size_t id = gather_ids.size();
                std::for_each(slice_ids.begin(), slice_ids.end(), [&id](size_t& i) {
                    i += id;
                });
                gather_ids.insert(gather_ids.end(), slice_ids.begin(), slice_ids.end());
                // collect slice sizes
                split_slices.push_back(slice_ids.size());
            }
        }

        const Shape& split_input_shape = split_node->get_input_shape(0);
        const size_t split_input_dims = std::accumulate(split_input_shape.begin(),
                                                        split_input_shape.end(),
                                                        std::size_t{1},
                                                        std::multiplies<Shape::value_type>());

        const Shape reshape_input_shape = {1, split_input_dims};
        auto reshape_input_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{2}, reshape_input_shape);
        auto reshape_input = std::make_shared<Reshape>(split_node->input_value(0), reshape_input_const, false);

        ov::copy_runtime_info(split_node, {reshape_input, reshape_input_const});

        std::shared_ptr<ov::Node> gather;
        auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        auto gather_indices = std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        if (graph_utils::are_shapes_equal(split_input_shape, reshape_input_shape)) {
            gather = std::make_shared<Gather>(split_node->input_value(0), gather_indices, gather_axis);
        } else {
            gather = std::make_shared<Gather>(reshape_input, gather_indices, gather_axis);
        }

        auto split_new_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        auto split_new_lengths =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{split_slices.size()}, split_slices);
        auto split_new = std::make_shared<VariadicSplit>(gather, split_new_axis, split_new_lengths);

        ov::copy_runtime_info(split_node, {gather_axis, gather_indices, gather, split_new_axis, split_new});

        for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
            for (auto& input : split_node->get_output_target_inputs(output_idx)) {
                auto target_node = get_next_node_skipping_certain(input.get_node()->shared_from_this(),
                                                                  graph_utils::is_gna_non_functional_node);
                std::shared_ptr<ov::Node> transpose = ov::as_type_ptr<Transpose>(target_node);
                if (transpose) {
                    auto reshape_output_const_new =
                        std::make_shared<Constant>(ov::element::i64,
                                                   ov::Shape{transpose->get_output_shape(0).size()},
                                                   transpose->get_output_shape(0));
                    auto reshape_output_new =
                        std::make_shared<Reshape>(split_new->output(output_idx), reshape_output_const_new, false);
                    ov::copy_runtime_info(split_node, {reshape_output_const_new, reshape_output_new});
                    ov::replace_node_update_name(transpose, reshape_output_new);
                } else {
                    for (auto consumer : split_node->output(output_idx).get_target_inputs()) {
                        consumer.replace_source_output(split_new->output(output_idx));
                    }
                }
            }
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(split_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
