// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_bf16_fallback.hpp"

#include <array>
#include <set>
#include <string>

#include "intel_gpu/primitives/paged_attention.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

PagedAttentionBF16Fallback::PagedAttentionBF16Fallback() {
    using Idx = cldnn::paged_attention::PagedAttentionInputIdx;
    auto pattern = ov::pass::pattern::wrap_type<ov::op::PagedAttentionExtension>([](const ov::Output<ov::Node>& output) {
        const auto* pa = output.get_node();
        return pa->get_input_element_type(Idx::QUERY) == ov::element::bf16 &&
               pa->get_input_element_type(Idx::KEY) == ov::element::bf16 &&
               pa->get_input_element_type(Idx::VALUE) == ov::element::bf16 &&
               pa->get_input_element_type(Idx::KEY_CACHE) == ov::element::f16 &&
               pa->get_input_element_type(Idx::VALUE_CACHE) == ov::element::f16;
    });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto pa = ov::as_type_ptr<ov::op::PagedAttentionExtension>(m.get_match_root());
        if (transformation_callback(pa)) {
            return false;
        }

        std::array<ov::element::Type, 3> output_types;
        std::array<std::set<ov::Input<ov::Node>>, 3> output_consumers;
        for (size_t i = 0; i < 3; ++i) {
            output_types[i] = pa->get_output_element_type(i);
            output_consumers[i] = pa->output(i).get_target_inputs();
        }
        for (const auto idx : {Idx::QUERY, Idx::KEY, Idx::VALUE, Idx::SCALE, Idx::ALIBI, Idx::SINKS}) {
            auto source = pa->input_value(idx);
            if (source.get_element_type() == ov::element::bf16) {
                auto replacement = ov::op::util::make_try_fold<ov::op::v0::Convert>(source, ov::element::f16);
                ov::copy_runtime_info(source.get_node_shared_ptr(), replacement);
                pa->input(idx).replace_source_output(replacement);
            }
        }

        for (int i = 0; i < 3; ++i) {
            if (i == 0 || output_types[i] == ov::element::bf16) {
                pa->set_out_type(i, ov::element::f16);
            }
        }
        pa->validate_and_infer_types();
        for (size_t i = 0; i < 3; ++i) {
            if (!output_consumers[i].empty() && output_types[i] != pa->get_output_element_type(i)) {
                auto convert = std::make_shared<ov::op::v0::Convert>(pa->output(i), output_types[i]);
                convert->set_friendly_name(pa->get_friendly_name() + "/output_" + std::to_string(i) + "_convert");
                convert->output(0).set_names(pa->output(i).get_names());
                pa->output(i).set_names({});
                ov::copy_runtime_info(pa, convert);
                for (auto input : output_consumers[i]) {
                    input.replace_source_output(convert);
                }
            }
        }
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "PagedAttentionBF16Fallback"), callback);
}

}  // namespace ov::intel_gpu
