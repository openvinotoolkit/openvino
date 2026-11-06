// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cut_lm_head.hpp"

#include <algorithm>
#include <memory>

#include "../llm_compiled_model.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class CutLMHeadMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::patterns::CutLMHeadMatcher");
    explicit CutLMHeadMatcher(std::shared_ptr<ov::Model>& lm_head_model) {
        // The first input to MatMul is treated as the cut point.
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});
        const ov::npuw::patterns::LMHeadTerminal terminal(matmul);
        auto res = opp::wrap_type<ov::op::v0::Result>({terminal.output});

        auto callback = [=, &lm_head_model](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_matmul =
                std::static_pointer_cast<ov::op::v0::MatMul>(node_to_output.at(matmul).get_node_shared_ptr());
            auto matched_result =
                std::static_pointer_cast<ov::op::v0::Result>(node_to_output.at(res).get_node_shared_ptr());

            const auto matched_node_last_op = terminal.last_op(node_to_output);

            // Skip Result nodes that are not logits.
            // Note: checking that Result's output name is "logits" is a sufficiently
            //       reliable check for finding exactly the logits output, because:
            //       1. LLMInferRequest always relies on the "logits" name to get logits
            //          from prefill/kvcache models.
            //       2. - Following Exporter configs: OnnxConfig, OnnxConfigWithPast,
            //            TextDecoderOnnxConfig and TextDecoderWithPositionIdsOnnxConfig
            //            from optimum-onnx name LLM output with "logits".
            //          - Most of optimum-intel OpenVINO Exporter configs are derived
            //            from the configs above.
            //          - optimum-intel `export()` function sets names for output tensors
            //            from Exporter config:
            //            https://github.com/huggingface/optimum-intel/blob/main/optimum/exporters/openvino/convert.py#L442-L445
            if (matched_result->output(0).get_names().count(ov::npuw::LLMCompiledModel::layer_names::logits) == 0) {
                return false;
            }

            // Cut point:
            auto matmul_first_source = matched_matmul->input(0).get_source_output();

            // If the cut-point tensor already feeds another Result (e.g. a hidden-states
            // output), the embeddings Result must own a distinct tensor. Otherwise both
            // Results share one descriptor and collapse to a single named port in
            // ov::ICompiledModel::ICompiledModel(). A cheap pass-through Convert gives the
            // embeddings its own tensor; NopElimination keeps it alive because dropping it
            // would merge two Result-feeding outputs (see can_output_be_replaced()).
            ov::Output<ov::Node> embeds_source = matmul_first_source;
            const auto& cut_readers = matmul_first_source.get_target_inputs();
            const bool producer_feeds_result =
                std::any_of(cut_readers.begin(), cut_readers.end(), [](const ov::Input<ov::Node>& in) {
                    return ov::is_type<ov::op::v0::Result>(in.get_node());
                });
            if (producer_feeds_result) {
                auto embeds_convert =
                    std::make_shared<ov::op::v0::Convert>(matmul_first_source, matmul_first_source.get_element_type());
                ov::copy_runtime_info(matmul_first_source.get_node_shared_ptr(), embeds_convert);
                embeds_source = embeds_convert->output(0);
            }

            // Cut original model: repurpose the logits Result as the embeddings output.
            matched_result->input(0).replace_source_output(embeds_source);
            // Name the producing tensor too, so the output name survives
            // ov::ICompiledModel::ICompiledModel() reconstruction.
            embeds_source.set_names({ov::npuw::LLMCompiledModel::layer_names::output_embeds});
            matched_result->output(0).set_names({ov::npuw::LLMCompiledModel::layer_names::output_embeds});
            matched_result->validate_and_infer_types();

            // Create an additional model after cut point:
            auto new_param = std::make_shared<ov::op::v0::Parameter>(matmul_first_source.get_element_type(),
                                                                     matmul_first_source.get_partial_shape());
            new_param->output(0).add_names({ov::npuw::LLMCompiledModel::layer_names::output_embeds});
            matched_matmul->input(0).replace_source_output(new_param);
            auto new_result = std::make_shared<ov::op::v0::Result>(matched_node_last_op);
            lm_head_model =
                std::make_shared<ov::Model>(ov::OutputVector{new_result->output(0)}, ov::ParameterVector{new_param});
            // The Model constructor does not deep-copy graph nodes, so tied-weight dequantization nodes
            // may remain shared with the original model. Any later pass modifying these nodes affects
            // both models and can cause incorrect behavior. Clone to isolate the graph nodes while
            // keeping constant data buffers shared.
            lm_head_model = lm_head_model->clone();

            return true;
        };
        register_matcher(std::make_shared<opp::Matcher>(res, "CutLMHeadMatcher"), std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

namespace ov::npuw {

patterns::LMHeadTerminal::LMHeadTerminal(const std::shared_ptr<ov::Node>& matmul_pattern) : matmul(matmul_pattern) {
    add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
    transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
    convert = opp::wrap_type<ov::op::v0::Convert>({matmul});
    scale = opp::wrap_type<ov::op::v1::Multiply, ov::op::v1::Divide>({matmul, opp::any_input()});
    const auto tanh = opp::wrap_type<ov::op::v0::Tanh>({scale});
    softcap = opp::wrap_type<ov::op::v1::Multiply>({tanh, opp::any_input()});
    output = std::make_shared<opp::op::Or>(ov::OutputVector{matmul->output(0),
                                                            add->output(0),
                                                            transpose->output(0),
                                                            convert->output(0),
                                                            softcap->output(0),
                                                            scale->output(0)});
}

std::shared_ptr<ov::Node> patterns::LMHeadTerminal::last_op(const ov::pass::pattern::PatternValueMap& values) const {
    // `scale` must be checked after `softcap`: the scale op is also a part of the softcap chain.
    for (const auto& p : {add, transpose, convert, softcap, scale}) {
        const auto it = values.find(p);
        if (it != values.end()) {
            return it->second.get_node_shared_ptr();
        }
    }
    return values.at(matmul).get_node_shared_ptr();
}

CutLMHead::CutLMHead(std::shared_ptr<ov::Model>& lm_head_model) : m_lm_head_model(lm_head_model) {}

bool CutLMHead::run_on_model(const std::shared_ptr<ov::Model>& model) {
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<CutLMHeadMatcher>(m_lm_head_model);
    rewr.run_on_model(model);
    return m_lm_head_model != nullptr;
}

}  // namespace ov::npuw
