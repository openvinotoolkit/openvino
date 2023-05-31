// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/ts_split.hpp"

#include <openvino/cc/ngraph/itt.hpp>

#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::pass::pattern;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;
using namespace ov::intel_gna::limitations;
using namespace ov::intel_gna::graph_utils;

namespace {
bool is_sinked(const Output<Node>& output) {
    auto split_node = output.get_node_shared_ptr();
    for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
        for (auto& input : split_node->get_output_target_inputs(output_idx)) {
            auto transpose = ov::as_type_ptr<Transpose>(input.get_node()->shared_from_this());
            if (transpose && !Limitations::is_transpose_supported(transpose))
                return true;
        }
    }
    return false;
}

std::vector<size_t> CreateGatherIndices(const ov::Shape& input_shape, const ov::Shape& order) {
    if (input_shape.size() < 2 || input_shape.size() > 4) {
        THROW_GNA_EXCEPTION << "Usupported shape size: " << input_shape.size();
    }

    ov::Shape input_shape_4d = input_shape;
    ov::Shape order_4d = order;
    // Just to simplify the code we transform all shapes to 4d by adding 1 dimentions at the end
    while (input_shape_4d.size() < 4) {
        input_shape_4d.push_back(1);
        order_4d.push_back(order_4d.size());
    }
    ov::Shape output_shape_4d = transpose_shape(input_shape_4d, order_4d);

    // common case when shape is 4d
    std::vector<size_t> xyz_4d = {input_shape_4d[3] * input_shape_4d[2] * input_shape_4d[1],
                                  input_shape_4d[3] * input_shape_4d[2],
                                  input_shape_4d[3],
                                  1};

    std::vector<size_t> xyz = transpose_shape(xyz_4d, order_4d);
    std::vector<size_t> gather_order;

    for (size_t n = 0; n < output_shape_4d[0]; ++n) {
        for (size_t i = 0; i < output_shape_4d[1]; ++i) {
            for (size_t j = 0; j < output_shape_4d[2]; ++j) {
                for (size_t k = 0; k < output_shape_4d[3]; ++k) {
                    gather_order.push_back(n * xyz[0] + i * xyz[1] + j * xyz[2] + k * xyz[3]);
                }
            }
        }
    }

    return gather_order;
}

}  // namespace

TSSplitBackward::TSSplitBackward() {
    MATCHER_SCOPE(TSSplitBackward);

    auto split_node_label = wrap_type<Split>(is_sinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& split_node_label_output = pattern_to_output.at(split_node_label);
        auto split_node = as_type_ptr<Split>(split_node_label_output.get_node_shared_ptr());

        std::vector<AxisVector> gather_indices_vecs;
        for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
            for (auto& input : split_node->get_output_target_inputs(output_idx)) {
                auto transpose = ov::as_type_ptr<Transpose>(input.get_node()->shared_from_this());

                if (transpose && !Limitations::is_transpose_supported(transpose)) {
                    auto transpose_const = ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
                    if (!transpose_const)
                        return false;
                    auto gather_indices_value =
                        CreateGatherIndices(transpose->get_input_shape(0), transpose_const->get_axis_vector_val());
                    gather_indices_vecs.push_back(gather_indices_value);
                } else {
                    const Shape& input_shape = input.get_shape();
                    const size_t input_dims = std::accumulate(input_shape.begin(),
                                                              input_shape.end(),
                                                              1,
                                                              std::multiplies<Shape::value_type>());
                    std::vector<size_t> indices(input_dims);
                    std::iota(indices.begin(), indices.end(), 1);
                    gather_indices_vecs.push_back(indices);
                    continue;
                }
            }
        }

        const Shape& split_input_shape = split_node->get_input_shape(0);
        const size_t split_input_dims = std::accumulate(split_input_shape.begin(),
                                                        split_input_shape.end(),
                                                        1,
                                                        std::multiplies<Shape::value_type>());

        const Shape reshape_input_shape = {1, split_input_dims};
        auto reshape_input_const = std::make_shared<Constant>(ov::element::i64, ov::Shape{2}, reshape_input_shape);
        auto reshape_input = std::make_shared<Reshape>(split_node->input_value(0), reshape_input_const, false);

        ov::copy_runtime_info(split_node, {reshape_input, reshape_input_const});

        std::vector<size_t> gather_indices_value;
        {
            size_t shift = 0;
            for (const auto& indices : gather_indices_vecs) {
                for (size_t i = 0; i < indices.size(); ++i) {
                    gather_indices_value.push_back(indices[i] + shift);
                }
                shift += indices.size();
            }
        }

        auto gather_axis = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        auto gather_indices =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_indices_value.size()}, gather_indices_value);
        auto gather = std::make_shared<Gather>(reshape_input, gather_indices, gather_axis);

        auto split_axis_new = std::make_shared<Constant>(ov::element::i64, ov::Shape{}, 1);
        auto split_new = std::make_shared<Split>(gather, split_axis_new, split_node->get_num_splits());

        ov::copy_runtime_info(split_node, {gather_axis, gather_indices, gather, split_axis_new, split_new});

        for (size_t output_idx = 0; output_idx < split_node->get_output_size(); ++output_idx) {
            for (auto& input : split_node->get_output_target_inputs(output_idx)) {
                auto transpose = ov::as_type_ptr<Transpose>(input.get_node()->shared_from_this());
                if (transpose && !Limitations::is_transpose_supported(transpose)) {
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
