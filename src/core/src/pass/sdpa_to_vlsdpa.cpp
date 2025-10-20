// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/sdpa_to_vlsdpa.hpp"

#include <array>
#include <string_view>
#include <utility>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/util/node_util.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/vl_sdpa.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov::pass;

SDPAToVLSDPA::SDPAToVLSDPA() = default;

bool SDPAToVLSDPA::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(SDPAToVLSDPA);

    // We rely on user (GENAI) to determine "attention_mask" input of model is
    // able to map to "cu_seqlens".
    if (!model->has_rt_info("model_type_hint"))
        return false;
    const std::string& model_type = model->get_rt_info<std::string>("model_type_hint");
    if (model_type != "QWenVL") {
        return false;
    }

    OPENVINO_ASSERT(ov::op::util::has_op_with_type<ov::op::v13::ScaledDotProductAttention>(model),
                    "No ScaledDotProductAttention operation observed in the graph, cannot perform "
                    "the SDPAToVLSDPA transformation.");
    if (transformation_callback(nullptr)) {  // verify plugin-specific determinations
        return false;
    }

    auto get_parameter = [=](const std::shared_ptr<ov::Model>& model,
                             const std::string& name) -> std::shared_ptr<v0::Parameter> {
        for (const auto& param : model->inputs()) {
            const auto& names = param.get_names();
            if (names.count(name)) {
                return ov::as_type_ptr<v0::Parameter>(param.get_node_shared_ptr());
            }
        }

        return nullptr;
    };

    // change "attention_mask" to "cu_seq_lens", and "window_attention_mask" to "cu_window_seqlens"
    constexpr std::array<std::pair<std::string_view, std::string_view>, 2> mask_2_seqlens_mapping = {
        {{"attention_mask", "cu_seq_lens"}, {"window_attention_mask", "cu_window_seqlens"}}};
    for (const auto& [old_name, new_name] : mask_2_seqlens_mapping) {
        if (auto attn_param = get_parameter(model, std::string(old_name))) {
            // all consumers should be SDPA
            bool consumers_are_sdpa = true;
            for (auto target : attn_param->get_output_target_inputs(0)) {
                if (auto sdpa = ov::as_type<v13::ScaledDotProductAttention>(target.get_node())) {
                    // when sdpa only has inputs q,k,v,attention_mask and is_causal==False
                    if (sdpa->get_input_size() > 4 || sdpa->get_causal()) {
                        consumers_are_sdpa = false;
                        break;
                    }
                } else {
                    consumers_are_sdpa = false;
                    break;
                }
            }

            if (!consumers_are_sdpa)
                continue;

            auto cu_seqlens_param = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
            // Optionally copy all names from old input except replaced name
            op::util::set_name(*cu_seqlens_param, std::string(new_name));
            model->replace_parameter(model->get_parameter_index(attn_param), cu_seqlens_param);

            for (auto target : cu_seqlens_param->get_output_target_inputs(0)) {
                auto sdpa = ov::as_type<v13::ScaledDotProductAttention>(target.get_node());
                OPENVINO_ASSERT(sdpa, "all consumers should be SDPA!");

                const auto sdpa_consumers = sdpa->get_output_target_inputs(0);
                const auto new_args = sdpa->input_values();
                OutputVector inputs{new_args.at(0), new_args.at(1), new_args.at(2), cu_seqlens_param};

                std::shared_ptr<op::internal::VLSDPA> vl_sdpa;
                vl_sdpa = std::make_shared<op::internal::VLSDPA>(inputs);
                vl_sdpa->set_friendly_name(sdpa->get_friendly_name());

                for (auto& consumer : sdpa_consumers)
                    consumer.replace_source_output(vl_sdpa);
            }
        }
    }

    model->validate_nodes_and_infer_types();
    return true;
}
