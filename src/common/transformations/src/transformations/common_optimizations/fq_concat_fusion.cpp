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

bool have_same_fake_quantize_params(const std::shared_ptr<v0::FakeQuantize>& lhs,
                                    const std::shared_ptr<v0::FakeQuantize>& rhs) {
    if (!lhs || !rhs || lhs->get_levels() != rhs->get_levels() ||
        lhs->get_auto_broadcast() != rhs->get_auto_broadcast()) {
        return false;
    }

    for (size_t index = 1; index < lhs->get_input_size(); ++index) {
        if (!ov::compare_constants(lhs->input_value(index).get_node_shared_ptr(),
                                   rhs->input_value(index).get_node_shared_ptr())) {
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
        ov::NodeVector old_nodes{concat};

        for (const auto& concat_input : concat->input_values()) {
            const auto input_fq = ov::as_type_ptr<v0::FakeQuantize>(concat_input.get_node_shared_ptr());
            if (!have_same_fake_quantize_params(input_fq, output_fq)) {
                return false;
            }

            new_concat_inputs.push_back(input_fq->input_value(0));
            old_nodes.push_back(input_fq);
        }
        auto new_concat = std::make_shared<v0::Concat>(new_concat_inputs, concat->get_axis());
        new_concat->set_friendly_name(concat->get_friendly_name());

        register_new_node(new_concat);
        output_fq->input(0).replace_source_output(new_concat->output(0));
        ov::copy_runtime_info(old_nodes, new_concat);
        return true;
    };

    const auto matcher = std::make_shared<pattern::Matcher>(fq_pattern, matcher_name);
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
