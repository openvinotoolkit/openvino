// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/reshape_to_squeeze.hpp"

#include "common/graph_utils.hpp"
#include "openvino/cc/ngraph/itt.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/transformation_helper.hpp"

using namespace ov::opset11;
using namespace ov::pass;
using namespace ov::intel_gna;
using namespace ov::intel_gna::pass;

namespace {

bool is_reshape_squeeze(const ov::Output<ov::Node>& output) {
    std::shared_ptr<ov::Node> reshape = output.get_node_shared_ptr();
    const ov::Shape shape_input = reshape->get_input_shape(0);
    const ov::Shape shape_output = reshape->get_output_shape(0);
    if (shape_input.size() > shape_output.size()) {
        const ov::Shape shape_input_sq = graph_utils::squeeze_shape(shape_input);
        const ov::Shape shape_output_sq = graph_utils::squeeze_shape(shape_output);
        return shape_input_sq.size() == shape_output_sq.size() &&
               std::equal(shape_input_sq.begin(), shape_input_sq.end(), shape_output_sq.begin());
    }

    return false;
}

bool is_reshape_unsqueeze(const ov::Output<ov::Node>& output) {
    std::shared_ptr<ov::Node> reshape = output.get_node_shared_ptr();
    const ov::Shape shape_input = reshape->get_input_shape(0);
    const ov::Shape shape_output = reshape->get_output_shape(0);
    if (shape_input.size() < shape_output.size()) {
        const ov::Shape shape_input_sq = graph_utils::squeeze_shape(shape_input);
        const ov::Shape shape_output_sq = graph_utils::squeeze_shape(shape_output);
        return shape_input_sq.size() == shape_output_sq.size() &&
               std::equal(shape_input_sq.begin(), shape_input_sq.end(), shape_output_sq.begin());
    }

    return false;
}

}  // namespace

ReshapeToSqueeze::ReshapeToSqueeze() {
    MATCHER_SCOPE(ReshapeToSqueeze);

    auto reshape_pattern =
        pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()}, is_reshape_squeeze);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape_node = pattern_map.at(reshape_pattern).get_node_shared_ptr();
        const ov::Shape& shape_in = reshape_node->get_input_shape(0);
        const ov::Shape& shape_out = reshape_node->get_output_shape(0);

        ov::AxisVector squeeze_axis = {};
        for (size_t i = 0; i <= shape_out.size(); ++i) {
            for (size_t j = 0; j < shape_in.size(); ++j) {
                if (i < shape_out.size() && shape_out[i] == shape_in[j]) {
                    i++;
                } else {
                    squeeze_axis.push_back(j);
                }
            }
        }

        if (squeeze_axis.size() != shape_in.size() - shape_out.size()) {
            return false;
        }

        // Squeeze
        auto squeeze_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{squeeze_axis.size()}, squeeze_axis);
        auto squeeze = std::make_shared<Squeeze>(reshape_node->input_value(0), squeeze_const);

        ov::replace_node_update_name(reshape_node, squeeze);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ReshapeToUnsqueeze::ReshapeToUnsqueeze() {
    MATCHER_SCOPE(ReshapeToUnsqueeze);

    auto reshape_pattern =
        pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()}, is_reshape_unsqueeze);

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape_node = pattern_map.at(reshape_pattern).get_node_shared_ptr();
        const ov::Shape& shape_in = reshape_node->get_input_shape(0);
        const ov::Shape& shape_out = reshape_node->get_output_shape(0);

        ov::AxisVector unsqueeze_axis = {};
        for (size_t i = 0; i <= shape_in.size(); ++i) {
            for (size_t j = 0; j < shape_out.size(); ++j) {
                if (i < shape_in.size() && shape_in[i] == shape_out[j]) {
                    i++;
                } else {
                    unsqueeze_axis.push_back(j);
                }
            }
        }

        if (unsqueeze_axis.size() != shape_out.size() - shape_in.size()) {
            return false;
        }

        // Unsqueeze
        auto squeeze_const =
            std::make_shared<Constant>(ov::element::i32, ov::Shape{unsqueeze_axis.size()}, unsqueeze_axis);
        auto squeeze = std::make_shared<Unsqueeze>(reshape_node->input_value(0), squeeze_const);

        ov::replace_node_update_name(reshape_node, squeeze);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ReshapeFuse::ReshapeFuse() {
    MATCHER_SCOPE(ReshapeFuse);

    auto reshape_in_pattern = pattern::wrap_type<Reshape>({pattern::any_input(), pattern::any_input()});
    auto reshape_out_pattern =
        pattern::wrap_type<Reshape, Squeeze, Unsqueeze>({reshape_in_pattern, pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto reshape_in_node = pattern_map.at(reshape_in_pattern).get_node_shared_ptr();
        const auto reshape_out_node = pattern_map.at(reshape_out_pattern).get_node_shared_ptr();
        const ov::Shape& shape_out = reshape_out_node->get_output_shape(0);

        // new reshape
        auto reshape_new_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{shape_out.size()}, shape_out);
        auto reshape_node_new = std::make_shared<Reshape>(reshape_in_node->input_value(0), reshape_new_const, false);

        ov::replace_node_update_name(reshape_out_node, reshape_node_new);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reshape_out_pattern, matcher_name);
    this->register_matcher(m, callback);
}

bool ReshapeReduction::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(ReshapeReduction);

    ov::pass::Manager manager(get_pass_config());
    manager.register_pass<ReshapeFuse>();
    manager.register_pass<ReshapeToSqueeze>();
    manager.register_pass<ReshapeToUnsqueeze>();

    return manager.run_passes(m);
}