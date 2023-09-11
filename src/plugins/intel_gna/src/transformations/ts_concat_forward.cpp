// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/ts_concat_forward.hpp"

#include <openvino/cc/ngraph/itt.hpp>

#include "backend/gna_limitations.hpp"
#include "common/graph_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov;
using namespace ov::opset12;
using namespace ov::pass::pattern;
using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;
using namespace ov::intel_gna::limitations;

namespace {

bool is_concat_sinked(const Output<Node>& output) {
    auto concat_node = ov::as_type_ptr<Concat>(output.get_node_shared_ptr());

    const Shape concat_output_shape = concat_node->get_output_shape(0);
    const int64_t axis = concat_node->get_concatenation_axis();
    if (graph_utils::get_first_valuable_dim_id(concat_output_shape) != axis)
        return false;

    for (size_t i = 0; i < concat_node->get_input_size(); ++i) {
        auto concat_input = concat_node->input_value(i);

        auto target_node = graph_utils::get_prev_node_skipping_certain(concat_input.get_node_shared_ptr(),
                                                                       graph_utils::is_gna_non_functional_node);
        std::shared_ptr<ov::Node> transpose = ov::as_type_ptr<Transpose>(target_node);

        if (transpose && !Limitations::is_transpose_supported(transpose))
            return true;
    }

    return false;
}

}  // namespace

TSConcatForward::TSConcatForward() {
    MATCHER_SCOPE(TSConcatForward);

    auto concat_node_label = wrap_type<Concat>(is_concat_sinked);

    matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto& concat_node_output = pattern_to_output.at(concat_node_label);
        auto concat_node = as_type_ptr<Concat>(concat_node_output.get_node_shared_ptr());

        ov::AxisVector gather_ids = {};
        OutputVector concat_inputs = {};
        for (size_t i = 0; i < concat_node->get_input_size(); ++i) {
            ov::Output<ov::Node> concat_input = concat_node->input_value(i);

            auto target_node = graph_utils::get_prev_node_skipping_certain(concat_input.get_node_shared_ptr(),
                                                                           graph_utils::is_gna_non_functional_node);
            std::shared_ptr<ov::Node> transpose = ov::as_type_ptr<Transpose>(target_node);

            ov::Shape transpose_shape = concat_input.get_shape();
            ov::AxisVector transpose_order(transpose_shape.size());
            if (transpose) {
                std::shared_ptr<Constant> transpose_const =
                    ov::as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
                transpose_order = transpose_const->get_axis_vector_val();
                transpose_shape = transpose->get_input_shape(0);
                concat_input = transpose->input_value(0);
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

            std::vector<int8_t> reshape_dims = {1, -1};
            auto reshape_input_const =
                std::make_shared<Constant>(ov::element::i64, ov::Shape{reshape_dims.size()}, reshape_dims);
            auto reshape_input = std::make_shared<Reshape>(concat_input, reshape_input_const, false);
            concat_inputs.push_back(reshape_input->output(0));
        }

        // new concat node
        size_t concat_axis = 1;
        auto concat_new = std::make_shared<Concat>(concat_inputs, concat_axis);

        // gather node
        size_t gather_axis = 1;
        auto gather_const_axis = std::make_shared<Constant>(ov::element::u8, ov::Shape{}, gather_axis);
        auto gather_const_ids = std::make_shared<Constant>(ov::element::i64, ov::Shape{gather_ids.size()}, gather_ids);
        auto gather_node = std::make_shared<Gather>(concat_new, gather_const_ids, gather_const_axis);

        // reshape after gather
        ov::Shape concat_shape_out = concat_node->get_output_shape(0);
        auto reshape_output_const =
            std::make_shared<Constant>(ov::element::i64, ov::Shape{concat_shape_out.size()}, concat_shape_out);
        auto reshape_output = std::make_shared<Reshape>(gather_node, reshape_output_const, false);

        // skip reshape if the input and output shapes are the same
        if (graph_utils::are_shapes_equal(reshape_output->get_input_shape(0), reshape_output->get_output_shape(0))) {
            ov::replace_node_update_name(concat_node, gather_node);
        } else {
            ov::replace_node_update_name(concat_node, reshape_output);
        }

        return true;
    };

    auto m = std::make_shared<Matcher>(concat_node_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}
