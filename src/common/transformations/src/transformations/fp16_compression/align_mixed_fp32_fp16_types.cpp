// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/align_mixed_fp32_fp16_types.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

using namespace ov;

bool ov::pass::AlignMixedFP32FP16Types::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(AlignMixedFP32FP16Types);
    std::unordered_set<std::string> new_friendly_names;

    auto generate_uniq_name = [&new_friendly_names](const std::string& initial_name) {
        int idx = 0;
        auto cur_name = initial_name;
        while (new_friendly_names.find(cur_name) != new_friendly_names.end()) {
            cur_name = initial_name + ":" + std::to_string(idx++);
        }
        new_friendly_names.insert(cur_name);
        return cur_name;
    };

    std::function<bool(const std::shared_ptr<Node>&)> insert_converts_before_if_needed =
        [&](const std::shared_ptr<Node>& node) {
            bool is_changed = false;
            for (const auto& input : node->inputs()) {
                const auto& incoming_output = input.get_source_output();
                const auto& incoming_node = incoming_output.get_node_shared_ptr();

                if (fp16_compression_is_disabled(incoming_node))
                    continue;  // we are in the middle

                if (!incoming_output.get_element_type().is_real())
                    continue;

                auto convert =
                    std::make_shared<ov::op::v0::Convert>(incoming_output, incoming_output.get_element_type());
                auto init_name = incoming_node->get_friendly_name() + "_decompressed_to_f32";
                convert->set_friendly_name(generate_uniq_name(init_name));
                copy_runtime_info(incoming_node, convert);
                input.replace_source_output(convert);
                disable_fp16_compression(convert);
                pass::disable_constant_folding(convert);
                is_changed = true;
            }
            return is_changed;
        };

    std::function<bool(const std::shared_ptr<Node>&)> insert_converts_after_if_needed =
        [&](const std::shared_ptr<Node>& node) {
            bool is_changed = false;
            for (const auto& output : node->outputs()) {
                for (const auto& out_inputs : output.get_target_inputs()) {
                    auto out_node = out_inputs.get_node()->shared_from_this();
                    if (fp16_compression_is_disabled(out_node) || is_precision_sensitive(out_inputs))
                        continue;
                    if (!out_inputs.get_element_type().is_real())
                        continue;

                    // todo xxx-101766: if we don't skip Results there is an error on GPU
                    if (ov::as_type_ptr<ov::op::v0::Result>(out_node))
                        continue;

                    // element_type of this convert will be changed automatically to f16 after
                    // ConvertPrecision(f32 -> f16). It's kept here f32 to keep ov::Model validatable
                    auto convert = std::make_shared<ov::op::v0::Convert>(output, out_inputs.get_element_type());
                    copy_runtime_info(node, convert);
                    auto init_name = node->get_friendly_name() + "_compressed_to_f16";
                    convert->set_friendly_name(generate_uniq_name(init_name));
                    out_inputs.replace_source_output(convert);
                    pass::disable_constant_folding(convert);
                    is_changed = true;
                }
            }
            return is_changed;
        };

    bool is_changed = false;
    for (auto& node : model->get_ordered_ops()) {
        if (!fp16_compression_is_disabled(node))
            continue;

        is_changed = insert_converts_before_if_needed(node) || is_changed;
        is_changed = insert_converts_after_if_needed(node) || is_changed;
    }

    return is_changed;
}
