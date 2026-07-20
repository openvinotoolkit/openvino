// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_concat_fusion.hpp"

#include <memory>
#include <vector>

#include "compare.hpp"
#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace op_util = ov::op::util;

namespace ov::pass {

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
            if (!op_util::have_same_fake_quantize_params(input_fq, output_fq)) {
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
