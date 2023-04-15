// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_reduction.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/arithmetic_reductions_keep_dims.hpp"
#include "openvino/op/util/logical_reduction_keep_dims.hpp"
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

bool get_keep_dims(const std::shared_ptr<Node>& reduction) {
    auto arithmetic_reduce = as_type_ptr<ov::op::util::ArithmeticReductionKeepDims>(reduction);
    auto logical_reduce = as_type_ptr<ov::op::util::LogicalReductionKeepDims>(reduction);

    bool keep_dims = false;  // squeeze/unsqueeze always reduces number of output dimensions
    if (logical_reduce)
        keep_dims = logical_reduce->get_keep_dims();
    else if (arithmetic_reduce)
        keep_dims = arithmetic_reduce->get_keep_dims();
    return keep_dims;
}

}  // namespace

TSReductionForward::TSReductionForward() {
    MATCHER_SCOPE(TSReductionForward);

    auto transpose_label = wrap_type<Transpose>({any_input(), wrap_type<Constant>()});
    auto reduce_label = wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims>(
        {transpose_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();

        auto transpose = pattern_to_output.at(transpose_label);
        auto reduction = pattern_to_output.at(reduce_label);
        if (transformation_callback(reduction)) {
            return false;
        }

        auto keep_dims = get_keep_dims(reduction);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = as_type_ptr<Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        auto rank = reduction->get_input_partial_shape(0).rank();
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto non_negative_axes =
            normalize_axes(reduction->get_friendly_name(), reduction_axes->cast_vector<int64_t>(), rank);
        OPENVINO_SUPPRESS_DEPRECATED_END

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        std::vector<size_t> new_values;
        new_values.reserve(non_negative_axes.size());
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(transpose_order_values[axis]);
        }

        if (!keep_dims) {
            transpose_order_values = GetOrderAfterReduction(non_negative_axes, transpose_order_values);
        }

        auto new_transpose_order = Constant::create(transpose_order->get_element_type(),
                                                    {transpose_order_values.size()},
                                                    transpose_order_values);

        auto new_const = Constant::create(reduction_axes->get_element_type(), reduction_axes->get_shape(), new_values);
        auto new_reduction = reduction->clone_with_new_inputs({transpose->input_value(0), new_const});
        auto new_transpose = transpose->clone_with_new_inputs({new_reduction, new_transpose_order});

        replace_node(reduction, new_transpose);
        new_reduction->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(reduction->get_friendly_name());
        UpdateForwardSinkingAbility(new_transpose);
        register_new_node(new_transpose);
        copy_runtime_info({transpose, reduction}, {new_transpose, new_reduction});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reduce_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSReductionBackward::TSReductionBackward() {
    MATCHER_SCOPE(TSReductionBackward);

    auto reduce_label = wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims>(
        {any_input(), wrap_type<Constant>()},
        HasSameOutputTransposeNodes);
    auto transpose_label = wrap_type<Transpose>({reduce_label, wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_map();
        auto transpose = pattern_to_output.at(transpose_label);
        auto reduction = pattern_to_output.at(reduce_label);
        if (transformation_callback(reduction)) {
            return false;
        }

        auto keep_dims = get_keep_dims(reduction);

        auto transpose_order = as_type_ptr<Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = as_type_ptr<Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        auto rank = reduction->get_input_partial_shape(0).rank();
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto non_negative_axes =
            normalize_axes(reduction->get_friendly_name(), reduction_axes->cast_vector<int64_t>(), rank);
        OPENVINO_SUPPRESS_DEPRECATED_END
        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        if (!keep_dims) {
            transpose_order_values = GetOrderBeforeReduction(non_negative_axes, transpose_order_values);
        }
        auto reversed_order_values = ReverseTransposeOrder(transpose_order_values);
        auto new_transpose_order = Constant::create(transpose_order->get_element_type(),
                                                    {transpose_order_values.size()},
                                                    transpose_order_values);

        std::vector<size_t> new_values;
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(reversed_order_values[axis]);
        }

        auto new_const = Constant::create(reduction_axes->get_element_type(), reduction_axes->get_shape(), new_values);
        auto new_transpose = transpose->clone_with_new_inputs({reduction->input_value(0), new_transpose_order});
        auto new_reduction = reduction->clone_with_new_inputs({new_transpose, new_const});

        replace_node(transpose, new_reduction);
        copy_runtime_info({transpose, reduction}, {new_transpose, new_reduction});
        UpdateForwardSinkingAbility(new_transpose);
        new_reduction->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(reduction->get_friendly_name());
        register_new_node(new_transpose);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}