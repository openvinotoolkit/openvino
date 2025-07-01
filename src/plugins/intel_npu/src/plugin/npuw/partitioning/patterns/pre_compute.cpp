// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../logging.hpp"
#include "pre_compute.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/ops.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/common_optimizations/fuse_rotary_positional_embeddings.hpp"
#include "openvino/pass/manager.hpp"

namespace opp = ov::pass::pattern;


// From DeepSeek
ov::npuw::patterns::pre_compute::SinCosLLama2::SinCosLLama2() {
    // auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({opp::any_input()});
    // auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    // auto concat_1 = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});
    // auto broadcast = opp::wrap_type<ov::op::v3::Broadcast>({opp::wrap_type<ov::op::v0::Constant>(), concat_1});
    // auto unsqueeze = opp::wrap_type<ov::op::v0::Unsqueeze>({opp::any_input(), opp::wrap_type<ov::op::v0::Constant>()});
    // auto convert = opp::wrap_type<ov::op::v0::Convert>({unsqueeze});
    // auto matmul = opp::wrap_type<ov::op::v0::MatMul>({broadcast, convert});
    // auto transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    // auto concat_2 = opp::wrap_type<ov::op::v0::Concat>({transpose, opp::any_input()});
    // auto sin_cos = opp::wrap_type<ov::op::v0::Sin, ov::op::v0::Cos>({concat_2});

    // auto callback = [=](ov::pass::pattern::Matcher& m) {
    //     auto& node_to_output = m.get_pattern_value_map();

    //     auto matched_shape_of = node_to_output.at(shape_of).get_node_shared_ptr();
    //     auto matched_gather = node_to_output.at(gather).get_node_shared_ptr();
    //     auto matched_concat_1 = node_to_output.at(concat_1).get_node_shared_ptr();
    //     auto matched_broadcast = node_to_output.at(broadcast).get_node_shared_ptr();
    //     auto matched_unsqueeze = node_to_output.at(unsqueeze).get_node_shared_ptr();
    //     auto matched_convert = node_to_output.at(convert).get_node_shared_ptr();
    //     auto matched_matmul = node_to_output.at(matmul).get_node_shared_ptr();
    //     auto matched_transpose = node_to_output.at(transpose).get_node_shared_ptr();
    //     auto matched_concat_2 = node_to_output.at(concat_2).get_node_shared_ptr();
    //     auto matched_sin_cos = node_to_output.at(sin_cos).get_node_shared_ptr();


    //     return false;  // root hasn't changed
    // };
    // register_matcher(std::make_shared<opp::Matcher>(sin_cos, "TagSinCos"), std::move(callback));

    auto rpe = opp::wrap_type<op::internal::RoPE>({opp::any_input()});

    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto rope_layer = node_to_output.at(rpe).get_node_shared_ptr();

        LOG_INFO("Rope detected at"<< rope_layer->get_name());
        //TODO: next stage actually replace rope layer by subgraph
        return false;  // root hasn't changed
    };

    register_matcher(std::make_shared<opp::Matcher>(rpe, "TagSinCos"), std::move(callback));
}

bool ov::npuw::patterns::pre_compute::RopeCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager("NPUW:RopeFusion");
    auto pass_config = manager.get_pass_config();
    manager.register_pass<ov::pass::RoPEFusion>(true);
    // pass_config->disable<ov::pass::RoPEFusionGPTJ>();
    // pass_config->disable<ov::pass::RoPEFusionIOSlicing>();
    // pass_config->disable<ov::pass::RoPEShareCosSin>();
    // pass_config->disable<ov::pass::RoPEFusionCosSinPreprocess>();
    manager.run_passes(model);

    if (m_build_cache) {
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::pre_compute::SinCosLLama2>();
        return rewr.run_on_model(model);
    }
    return true;
}
