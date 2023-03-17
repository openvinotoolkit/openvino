// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_squeeze.hpp"

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

bool shape_to_squeeze_axes(const std::shared_ptr<Node>& reshape,
                           const std::shared_ptr<Constant>& reshape_to_shape,
                           std::vector<size_t>& result_axes) {
    result_axes.clear();
    auto reduction_axes_values = reshape_to_shape->cast_vector<int64_t>();
    // supported the case if Reshape is equal to Squeeze
    const auto &new_shape = reduction_axes_values;
    const auto &input_pshape = reshape->get_input_partial_shape(0);
    // todo: support dynamic case
    if (input_pshape.is_dynamic()) {
        return false;
    }

    const auto input_shape = input_pshape.to_shape();
    if (new_shape.size() < input_shape.size()) {
        for (size_t i = 0, j = 0; i < new_shape.size(); j++) {
            const auto input_dim = static_cast<int64_t>(input_shape[j]);
            if (new_shape[i] == input_dim) {
                i++;
            } else if (new_shape[i] != input_dim && input_dim != 1) {
                return false;
            } else {
                result_axes.push_back(j);
            }
        }
    } else {
        // another reshape type, not Squeeze
        // todo: move this checks in the pattern
        return false;
    }
    return true;
}

std::vector<size_t> squeeze_axes_to_shape(const std::shared_ptr<Node>& input_node, std::vector<size_t> squeeze_axes) {
    std::vector<size_t> to_shape;
    std::sort(squeeze_axes.begin(), squeeze_axes.end());
    const auto& input_shape = input_node->input(0).get_shape(); // check is static
    for (size_t i = 0, j = 0; i < input_shape.size(); ++i) {
        if (j < squeeze_axes.size() && i == squeeze_axes[j]) {
            ++j;
            continue;
        }
        to_shape.push_back(input_shape[i]);
    }
    return to_shape;
}

}

TSSqueezeForward::TSSqueezeForward() {
    MATCHER_SCOPE(TSSqueezeForward);

    auto transpose_label = wrap_type<Transpose>({any_input(), wrap_type<Constant>()});
    auto squeeze_label = wrap_type<Squeeze, Reshape>({transpose_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto squeeze = pattern_to_output.at(squeeze_label);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto squeeze_axes = as_type_ptr<Constant>(squeeze->get_input_node_shared_ptr(1));
        if (!transpose_order || !squeeze_axes) {
            return false;
        }

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<Reshape>(squeeze)) {
            auto success = shape_to_squeeze_axes(squeeze, squeeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            auto rank = squeeze->get_input_partial_shape(0).rank();
            non_negative_axes = normalize_axes(squeeze->get_friendly_name(), squeeze_axes->cast_vector<int64_t>(), rank);
        }

        // if 2nd input to squeeze is empty then all '1' dims will be deleted.
        if (non_negative_axes.empty()) {
            auto input_pshape = transpose->input_value(0).get_partial_shape();
            if (input_pshape.is_dynamic()) {
                return false;
            }
            for (size_t i = 0; i < input_pshape.size(); ++i) {
                if (input_pshape[i].get_length() == 1) {
                    non_negative_axes.push_back(i);
                }
            }
        }

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        std::vector<size_t> new_values;
        new_values.reserve(non_negative_axes.size());
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(transpose_order_values[axis]);
        }

        transpose_order_values = GetOrderAfterReduction(non_negative_axes, transpose_order_values);
        auto new_transpose_order = Constant::create(transpose_order->get_element_type(),
                                                    {transpose_order_values.size()},
                                                    transpose_order_values);

        if (as_type_ptr<Reshape>(squeeze)) {
            new_values = squeeze_axes_to_shape(transpose, new_values);
        }

        auto new_const = Constant::create(squeeze_axes->get_element_type(), squeeze_axes->get_shape(), new_values);
        auto new_squeeze = squeeze->clone_with_new_inputs({transpose->input_value(0), new_const});
        auto new_transpose = transpose->clone_with_new_inputs({new_squeeze, new_transpose_order});

        replace_node(squeeze, new_transpose);
        new_squeeze->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(squeeze->get_friendly_name());
        UpdateForwardSinkingAbility(new_transpose);
        register_new_node(new_transpose);
        copy_runtime_info({transpose, squeeze}, {new_transpose, new_squeeze});

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(squeeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSSqueezeBackward::TSSqueezeBackward() {
    MATCHER_SCOPE(TSSqueezeBackward);

    auto squeeze_label = wrap_type<Squeeze, Reshape>({any_input(), wrap_type<Constant>()}, HasSameOutputTransposeNodes);
    auto transpose_label = wrap_type<Transpose>({squeeze_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto squeeze = pattern_to_output.at(squeeze_label);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto squeeze_axes = as_type_ptr<Constant>(squeeze->get_input_node_shared_ptr(1));
        if (!transpose_order || !squeeze_axes) {
            return false;
        }

        std::vector<size_t> non_negative_axes;
        if (as_type_ptr<Reshape>(squeeze)) {
            auto success = shape_to_squeeze_axes(squeeze, squeeze_axes, non_negative_axes);
            if (!success) {
                return false;
            }
        } else {
            auto rank = squeeze->get_input_partial_shape(0).rank();
            non_negative_axes = normalize_axes(squeeze->get_friendly_name(), squeeze_axes->cast_vector<int64_t>(), rank);
        }

        bool squeeze_all_dims = false;
        if (non_negative_axes.empty()) {
            auto input_pshape = squeeze->input_value(0).get_partial_shape();
            if (input_pshape.is_dynamic()) {
                return false;
            }
            for (size_t i = 0; i < input_pshape.size(); ++i) {
                if (input_pshape[i] == 1) {
                    non_negative_axes.push_back(i);
                }
            }
            squeeze_all_dims = true;
        }

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        transpose_order_values = GetOrderBeforeReduction(non_negative_axes, transpose_order_values);
        auto reversed_order_values = ReverseTransposeOrder(transpose_order_values);

        std::vector<size_t> new_values;
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(reversed_order_values[axis]);
        }

        auto new_transpose_order = Constant::create(transpose_order->get_element_type(),
                                                    {transpose_order_values.size()},
                                                    transpose_order_values);
        if (as_type_ptr<Reshape>(squeeze)) {
            new_values = squeeze_axes_to_shape(squeeze, new_values);
        }

        std::shared_ptr<Node> new_squeeze;
        auto new_transpose = transpose->clone_with_new_inputs({squeeze->input_value(0), new_transpose_order});
        if (squeeze_all_dims) {
            new_squeeze = squeeze->clone_with_new_inputs({new_transpose, squeeze->input_value(1)});
        } else {
            auto new_const = std::make_shared<Constant>(squeeze_axes->get_element_type(), squeeze_axes->get_shape(), new_values);
            new_squeeze = squeeze->clone_with_new_inputs({new_transpose, new_const});
        }

        replace_node(transpose, new_squeeze);
        copy_runtime_info({transpose, squeeze}, {new_transpose, new_squeeze});
        UpdateForwardSinkingAbility(new_transpose);
        new_squeeze->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(squeeze->get_friendly_name());
        register_new_node(new_transpose);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}