// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_unsqueeze.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/transpose_sinking/ts_utils.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace opset10;
using namespace ov::pass::pattern;
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {

bool shape_to_unsqueeze_axes(const std::shared_ptr<Node>& reshape,
                           const std::shared_ptr<Constant>& reshape_to_shape,
                           std::vector<size_t>& result_axes) {
    result_axes.clear();
    auto reduction_axes_values = reshape_to_shape->cast_vector<int64_t>();
    // supported the case if Reshape is equal to Unsqueeze
    const auto &new_shape = reduction_axes_values;
    const auto &input_pshape = reshape->get_input_partial_shape(0);
    // todo: support dynamic case
    if (input_pshape.is_dynamic()) {
        return false;
    }

    const auto input_shape = input_pshape.to_shape();
    if (new_shape.size() > input_shape.size()) {
        for (size_t i = 0, j = 0; i < input_shape.size();j++) {
            if (input_shape[i] == new_shape[j]) {
                i++;
            } else if (input_shape[i] != new_shape[j] && new_shape[j] != 1) {
                return false;
            } else {
                result_axes.push_back(j);
            }
        }
    } else {
        // another reshape type, not Unsqueeze
        // todo: move this checks in the pattern
        return false;
    }
    return true;
}

std::vector<size_t> unsqueeze_axes_to_shape(const std::shared_ptr<Node>& input_node, std::vector<size_t> unsqueeze_axes) {
    const auto& input_shape = input_node->input(0).get_shape(); // check is static
    std::vector<size_t> to_shape(input_shape.size() + unsqueeze_axes.size());
    std::sort(unsqueeze_axes.begin(), unsqueeze_axes.end());
    std::stack<size_t, std::vector<size_t>> shape_to_add(input_shape);
    for (size_t i = 0, j = 0; i < to_shape.size(); ++i) {
        if (j < unsqueeze_axes.size() && i == unsqueeze_axes[j]) {
            to_shape[i] = 1;
            j++;
            continue;
        }
        to_shape[i] = shape_to_add.top();
        shape_to_add.pop();
    }
    return to_shape;
}
}  // namespace

TSUnsqueezeForward::TSUnsqueezeForward() {
    MATCHER_SCOPE(TSUnsqueezeForward);

    auto transpose_label = wrap_type<Transpose>({any_input(), wrap_type<Constant>()});
    auto unsqueeze_label = wrap_type<Unsqueeze, Reshape>({transpose_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto unsqueeze = pattern_to_output.at(unsqueeze_label);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto unsqueeze_axes = as_type_ptr<Constant>(unsqueeze->get_input_node_shared_ptr(1));
        if (!transpose_order || !unsqueeze_axes) {
            return false;
        }

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<Reshape>(unsqueeze)) {
            auto success = shape_to_unsqueeze_axes(unsqueeze, unsqueeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            auto rank = unsqueeze->get_output_partial_shape(0).rank();
            non_negative_axes = normalize_axes(unsqueeze->get_friendly_name(), unsqueeze_axes->cast_vector<int64_t>(), rank);
        }
        auto ts_order_values = transpose_order->cast_vector<size_t>();

/*        std::vector<size_t> new_values;
        new_values.reserve(non_negative_axes.size());
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(ts_order_values[axis]);
        }*/

        ts_order_values = GetOrderBeforeReduction(non_negative_axes, ts_order_values);
        auto new_transpose_order = Constant::create(transpose_order->get_element_type(),
                                                    {ts_order_values.size()},
                                                    ts_order_values);

        /*if (as_type_ptr<Reshape>(unsqueeze)) {
            new_values = unsqueeze_axes_to_shape(unsqueeze, new_values);
        }*/
        auto new_unsqueeze = unsqueeze->clone_with_new_inputs({transpose->input_value(0), unsqueeze->input_value(1)});
        auto new_transpose = transpose->clone_with_new_inputs({new_unsqueeze, new_transpose_order});

        replace_node(unsqueeze, new_transpose);
        new_unsqueeze->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(unsqueeze->get_friendly_name());
        UpdateForwardSinkingAbility(new_transpose);
        register_new_node(new_transpose);
        copy_runtime_info({transpose, unsqueeze}, {new_transpose, new_unsqueeze});

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(unsqueeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSUnsqueezeBackward::TSUnsqueezeBackward() {
    MATCHER_SCOPE(TSUnsqueezeBackward);

    auto unsqueeze_label = wrap_type<Unsqueeze, Reshape>({any_input(), wrap_type<Constant>()}, HasSameOutputTransposeNodes);
    auto transpose_label = wrap_type<Transpose>({unsqueeze_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto unsqueeze = pattern_to_output.at(unsqueeze_label);

        auto transpose_order = std::dynamic_pointer_cast<Constant>(transpose->get_input_node_shared_ptr(1));
        auto unsqueeze_axes = std::dynamic_pointer_cast<Constant>(unsqueeze->get_input_node_shared_ptr(1));
        if (!transpose_order || !unsqueeze_axes)
            return false;

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<Reshape>(unsqueeze)) {
            auto success = shape_to_unsqueeze_axes(unsqueeze, unsqueeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            auto rank = unsqueeze->get_output_partial_shape(0).rank();
            non_negative_axes = normalize_axes(unsqueeze->get_friendly_name(), unsqueeze_axes->cast_vector<int64_t>(), rank);
        }

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        auto old_transpose_order_values = transpose_order_values;
        std::vector<size_t> new_values;

        if (non_negative_axes.size() == transpose_order_values.size()) {
            // input is a scalar, we unsqueeze all dims
            // it's enough to eliminate such Transpose
            transpose->output(0).replace(unsqueeze);
            return true;
        }

        for (const auto& axis : non_negative_axes) {
            auto it = std::find(old_transpose_order_values.begin(), old_transpose_order_values.end(), axis);
            if (it != old_transpose_order_values.end()) {
                new_values.push_back(it - old_transpose_order_values.begin());
            }
        }

        transpose_order_values = GetOrderAfterReduction(new_values, transpose_order_values);
        auto new_transpose_order = std::make_shared<Constant>(transpose_order->get_element_type(),
                                                              Shape{transpose_order_values.size()},
                                                              transpose_order_values);
        if (as_type_ptr<Reshape>(unsqueeze)) {
            new_values = unsqueeze_axes_to_shape(unsqueeze, new_values);
        }
        auto new_const = Constant::create(unsqueeze_axes->get_element_type(), unsqueeze_axes->get_shape(), new_values);
        auto new_transpose = transpose->clone_with_new_inputs({unsqueeze->input_value(0), new_transpose_order});
        auto new_unsqueeze = unsqueeze->clone_with_new_inputs({new_transpose, new_const});

        replace_node(transpose, new_unsqueeze);
        copy_runtime_info({transpose, unsqueeze}, {new_transpose, new_unsqueeze});
        UpdateForwardSinkingAbility(new_transpose);
        new_unsqueeze->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(unsqueeze->get_friendly_name());
        register_new_node(new_transpose);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}