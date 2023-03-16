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
using namespace ov::pass::transpose_sinking;
using namespace ov::pass::transpose_sinking::utils;

namespace {
std::vector<size_t> get_updated_order_forward(const std::vector<size_t>& axes_values,
                                              const std::vector<size_t>& order_values) {
    size_t buffer_size = order_values.size() - axes_values.size();
    std::vector<size_t> aligned_order(buffer_size, 0);
    std::vector<size_t> values_to_reduce(axes_values);
    for (size_t i = 0; i < values_to_reduce.size(); ++i) {
        values_to_reduce[i] = order_values[axes_values[i]];
    }
    std::sort(values_to_reduce.begin(), values_to_reduce.end());
    for (size_t i = 0, j = 0; i < order_values.size(); ++i) {
        if (std::find(axes_values.begin(), axes_values.end(), i) != axes_values.end()) {
            continue;
        }

        auto lb = std::lower_bound(values_to_reduce.begin(), values_to_reduce.end(), order_values[i]);
        aligned_order[j] = order_values[i] - (lb - values_to_reduce.begin());
        ++j;
    }
    return aligned_order;
}

std::vector<size_t> get_updated_order_backward(const std::vector<size_t>& axes_values,
                                               const std::vector<size_t>& order_values) {
    size_t buffer_size = order_values.size() + axes_values.size();
    std::vector<size_t> aligned_order(buffer_size);

    std::vector<int64_t> cnt_deleted(buffer_size);
    int64_t cnt = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(cnt_deleted.size()); ++i) {
        if (std::find(axes_values.begin(), axes_values.end(), i) != axes_values.end()) {
            cnt++;
        }
        cnt_deleted[i] = i - cnt;
    }

    for (size_t i = 0, j = 0; i < aligned_order.size(); ++i) {
        if (std::find(axes_values.begin(), axes_values.end(), i) != axes_values.end()) {
            aligned_order[i] = i;
            continue;
        }

        aligned_order[i] = std::find(cnt_deleted.begin(), cnt_deleted.end(), order_values[j]) - cnt_deleted.begin();
        ++j;
    }
    return aligned_order;
}

bool get_keep_dims(const std::shared_ptr<Node>& reduction) {
    auto arithmetic_reduce = std::dynamic_pointer_cast<ov::op::util::ArithmeticReductionKeepDims>(reduction);
    auto logical_reduce = std::dynamic_pointer_cast<ov::op::util::LogicalReductionKeepDims>(reduction);

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

    auto transpose_label = pattern::wrap_type<Transpose>({pattern::any_input(), pattern::wrap_type<Constant>()},
                                                         pattern::consumers_count(1));
    auto reduce_or_squeeze_label = pattern::
        wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims, Squeeze, Unsqueeze>(
            {transpose_label, pattern::wrap_type<Constant>()});

    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reduction = pattern_to_output.at(reduce_or_squeeze_label).get_node_shared_ptr();
        auto keep_dims = get_keep_dims(reduction);

        auto transpose_order = std::dynamic_pointer_cast<Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = std::dynamic_pointer_cast<Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        auto unsqueeze = std::dynamic_pointer_cast<Unsqueeze>(reduction);
        auto rank =
            unsqueeze ? reduction->get_output_partial_shape(0).rank() : reduction->get_input_partial_shape(0).rank();
        auto non_negative_axes =
            normalize_axes(reduction->get_friendly_name(), reduction_axes->cast_vector<int64_t>(), rank);

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        std::vector<size_t> new_values;
        new_values.reserve(non_negative_axes.size());
        for (const auto& axis : non_negative_axes) {
            new_values.push_back(transpose_order_values[axis]);
        }

        if (!keep_dims) {
            if (non_negative_axes.empty()) {
                auto input_pshape = transpose->input_value(0).get_partial_shape();

                if (input_pshape.is_static()) {
                    for (size_t i = 0; i < input_pshape.size(); ++i) {
                        if (input_pshape[i] == 1) {
                            non_negative_axes.push_back(i);
                        }
                    }
                } else {
                    return false;
                }
            }
            if (unsqueeze) {
                transpose_order_values = get_updated_order_backward(non_negative_axes, transpose_order_values);
            } else {
                transpose_order_values = get_updated_order_forward(non_negative_axes, transpose_order_values);
            }
        }
        auto new_transpose_order = std::make_shared<Constant>(transpose_order->get_element_type(),
                                                              Shape{transpose_order_values.size()},
                                                              transpose_order_values);

        std::shared_ptr<Node> new_reduction;
        if (!unsqueeze) {
            auto new_const =
                std::make_shared<Constant>(reduction_axes->get_element_type(), reduction_axes->get_shape(), new_values);
            new_reduction = reduction->clone_with_new_inputs({transpose->input_value(0), new_const});
        } else {
            new_reduction = reduction->clone_with_new_inputs({transpose->input_value(0), reduction->input_value(1)});
        }
        auto new_transpose = transpose->clone_with_new_inputs({new_reduction, new_transpose_order});
        replace_node(reduction, new_transpose);
        new_reduction->set_friendly_name(transpose->get_friendly_name());
        new_transpose->set_friendly_name(reduction->get_friendly_name());
        UpdateForwardSinkingAbility(new_transpose);
        register_new_node(new_transpose);
        copy_runtime_info({transpose, reduction}, {new_transpose, new_reduction});
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(reduce_or_squeeze_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}

TSReductionBackward::TSReductionBackward() {
    MATCHER_SCOPE(TSReductionBackward);

    auto reduce_or_squeeze_label = pattern::
        wrap_type<op::util::ArithmeticReductionKeepDims, op::util::LogicalReductionKeepDims, Squeeze, Unsqueeze>(
            {pattern::any_input(), pattern::wrap_type<Constant>()},
            HasSameOutputTransposeNodes);
    auto transpose_label = pattern::wrap_type<Transpose>({reduce_or_squeeze_label, pattern::wrap_type<Constant>()});
    ov::matcher_pass_callback matcher_pass_callback = [=](pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        auto transpose = pattern_to_output.at(transpose_label).get_node_shared_ptr();
        auto reduction = pattern_to_output.at(reduce_or_squeeze_label).get_node_shared_ptr();
        auto keep_dims = get_keep_dims(reduction);
        auto transpose_order = std::dynamic_pointer_cast<Constant>(transpose->get_input_node_shared_ptr(1));
        auto reduction_axes = std::dynamic_pointer_cast<Constant>(reduction->get_input_node_shared_ptr(1));
        if (!transpose_order || !reduction_axes)
            return false;

        auto unsqueeze = std::dynamic_pointer_cast<Unsqueeze>(reduction);
        auto rank =
            unsqueeze ? reduction->get_output_partial_shape(0).rank() : reduction->get_input_partial_shape(0).rank();
        auto non_negative_axes =
            normalize_axes(reduction->get_friendly_name(), reduction_axes->cast_vector<int64_t>(), rank);

        auto transpose_order_values = transpose_order->cast_vector<size_t>();
        auto old_transpose_order_values = transpose_order_values;
        std::vector<size_t> new_values;
        if (unsqueeze) {
            if (non_negative_axes.size() == transpose_order_values.size()) {
                // input is a scalar, we unsqueeze all dims
                // it's enough to eliminate such Transpose
                transpose->output(0).replace(reduction);
                return true;
            }
            for (const auto& axis : non_negative_axes) {
                auto it = std::find(old_transpose_order_values.begin(), old_transpose_order_values.end(), axis);
                if (it != old_transpose_order_values.end()) {
                    new_values.push_back(it - old_transpose_order_values.begin());
                }
            }
        }
        bool squeeze_all_dims = false;
        if (!keep_dims) {
            if (non_negative_axes.empty()) {
                auto input_pshape = reduction->input_value(0).get_partial_shape();
                if (input_pshape.is_static()) {
                    for (size_t i = 0; i < input_pshape.size(); ++i) {
                        if (input_pshape[i] == 1) {
                            non_negative_axes.push_back(i);
                        }
                    }
                    squeeze_all_dims = true;
                } else {
                    return false;
                }
            }
            if (unsqueeze) {
                transpose_order_values = get_updated_order_forward(new_values, transpose_order_values);
            } else {
                transpose_order_values = get_updated_order_backward(non_negative_axes, transpose_order_values);
            }
        }

        if (!unsqueeze) {
            auto reversed_order_values = ReverseTransposeOrder(transpose_order_values);
            for (const auto& axis : non_negative_axes) {
                new_values.push_back(reversed_order_values[axis]);
            }
        }

        auto new_transpose_order = std::make_shared<Constant>(transpose_order->get_element_type(),
                                                              Shape{transpose_order_values.size()},
                                                              transpose_order_values);
        std::shared_ptr<Node> new_transpose, new_reduction;
        if (squeeze_all_dims) {
            new_transpose = transpose->clone_with_new_inputs({reduction->input_value(0), new_transpose_order});
            new_reduction = reduction->clone_with_new_inputs({new_transpose, reduction->input_value(1)});
        } else {
            auto new_const =
                std::make_shared<Constant>(reduction_axes->get_element_type(), reduction_axes->get_shape(), new_values);
            new_transpose = transpose->clone_with_new_inputs({reduction->input_value(0), new_transpose_order});
            new_reduction = reduction->clone_with_new_inputs({new_transpose, new_const});
        }
        replace_node(transpose, new_reduction);
        copy_runtime_info({transpose, reduction}, {new_transpose, new_reduction});
        UpdateForwardSinkingAbility(new_transpose);
        new_reduction->set_friendly_name(transpose->get_friendly_name());
        register_new_node(new_transpose);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(transpose_label, matcher_name);
    register_matcher(m, matcher_pass_callback);
}