// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_concat_fusion.hpp"

#include <memory>
#include <vector>

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace v0 = ov::op::v0;

namespace ov::pass {

namespace {

bool inputs_equal_or_same_constant(const Output<Node>& lhs, const Output<Node>& rhs) {
    if (lhs == rhs) {
        return true;
    }

    const auto lhs_const = ov::as_type_ptr<v0::Constant>(lhs.get_node_shared_ptr());
    const auto rhs_const = ov::as_type_ptr<v0::Constant>(rhs.get_node_shared_ptr());
    if (!lhs_const || !rhs_const) {
        return false;
    }

    return ov::compare_constants(lhs_const, rhs_const);
}

bool have_same_fake_quantize_params(const std::shared_ptr<v0::FakeQuantize>& lhs,
                                    const std::shared_ptr<v0::FakeQuantize>& rhs) {
    if (!lhs || !rhs || lhs->get_levels() != rhs->get_levels()) {
        return false;
    }

    for (size_t index = 1; index < lhs->get_input_size(); ++index) {
        if (!inputs_equal_or_same_constant(lhs->input_value(index), rhs->input_value(index))) {
            return false;
        }
    }

    return true;
}

}  // namespace

FakeQuantizeConcatFusion::FakeQuantizeConcatFusion() {
    MATCHER_SCOPE(FakeQuantizeConcatFusion);
    auto concat_pattern = pattern::wrap_type<v0::Concat>({}, pattern::consumers_count(1));
    auto fq_pattern = pattern::wrap_type<v0::FakeQuantize>(
        {concat_pattern, pattern::any_input(), pattern::any_input(), pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto output_fq = ov::as_type_ptr<v0::FakeQuantize>(m.get_match_root());
        if (!output_fq) {
            return false;
        }

        const auto concat = ov::as_type_ptr<v0::Concat>(output_fq->input_value(0).get_node_shared_ptr());
        if (!concat || concat->get_input_size() == 0) {
            return false;
        }

        OutputVector new_concat_inputs;
        ov::NodeVector old_nodes{concat, output_fq};

        for (const auto& concat_input : concat->input_values()) {
            const auto input_fq = ov::as_type_ptr<v0::FakeQuantize>(concat_input.get_node_shared_ptr());
            if (!have_same_fake_quantize_params(input_fq, output_fq)) {
                return false;
            }

            new_concat_inputs.push_back(input_fq->input_value(0));
            old_nodes.push_back(input_fq);
        }

        auto new_concat = std::make_shared<v0::Concat>(new_concat_inputs, concat->get_axis());
        auto new_output_fq = output_fq->clone_with_new_inputs({new_concat,
                                                               output_fq->input_value(1),
                                                               output_fq->input_value(2),
                                                               output_fq->input_value(3),
                                                               output_fq->input_value(4)});
        if (transformation_callback(new_output_fq)) {
            return false;
        }

        register_new_node(new_concat);
        register_new_node(new_output_fq);
        new_output_fq->set_friendly_name(output_fq->get_friendly_name());
        ov::copy_runtime_info(old_nodes, {new_concat, new_output_fq});
        ov::replace_node(output_fq, new_output_fq);
        return true;
    };

    const auto matcher = std::make_shared<pattern::Matcher>(fq_pattern, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
