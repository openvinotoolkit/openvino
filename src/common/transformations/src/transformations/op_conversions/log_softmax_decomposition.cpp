// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/log_softmax_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/log_softmax.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::LogSoftmaxDecomposition::LogSoftmaxDecomposition() {
    MATCHER_SCOPE(LogSoftmaxDecomposition);
    // Decomposes LogSoftmax(x, axis) op into sub-graph x - log(reduce_sum(exp(x), axis))
    auto log_softmax = ov::pass::pattern::wrap_type<ov::op::v5::LogSoftmax>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto log_softmax_node =
            ov::as_type_ptr<ov::op::v5::LogSoftmax>(pattern_to_output.at(log_softmax).get_node_shared_ptr());

        if (log_softmax_node == nullptr || transformation_callback(log_softmax_node)) {
            return false;
        }

        auto axis1 = ov::op::v0::Constant::create(element::Type_t::i64, ov::Shape{1}, {log_softmax_node->get_axis()});
        auto axis2 = ov::op::v0::Constant::create(element::Type_t::i64, ov::Shape{1}, {log_softmax_node->get_axis()});
        auto max = std::make_shared<ov::op::v1::ReduceMax>(log_softmax_node->input_value(0), axis1, true);
        auto sub = std::make_shared<ov::op::v1::Subtract>(log_softmax_node->input_value(0), max);
        auto exp = std::make_shared<ov::op::v0::Exp>(sub);
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(exp, axis2, true);
        auto log = std::make_shared<ov::op::v0::Log>(sum);
        auto sub_end = std::make_shared<ov::op::v1::Subtract>(sub, log);

        sub_end->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(log_softmax_node, {axis1, axis2, max, sub, exp, sum, log, sub_end});
        ov::replace_node(m.get_match_root(), sub_end);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(log_softmax, matcher_name);
    register_matcher(m, callback);
}
