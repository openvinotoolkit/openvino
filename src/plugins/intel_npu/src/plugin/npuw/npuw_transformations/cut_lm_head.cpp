// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cut_lm_head.hpp"

#include <memory>

#include "../llm_compiled_model.hpp"
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
    explicit CutLMHeadMatcher(std::shared_ptr<ov::Model>& lm_head_model,
                              std::shared_ptr<ov::op::v0::Result>& drop_result,
                              std::string& output_embeds_name) {
        // We are interested at first input to MatMul as a cut point
        auto matmul = opp::wrap_type<ov::op::v0::MatMul>({opp::any_input(), opp::any_input()});

        // There are several patterns for matmul we are looking for:
        // Matmul -> Result
        // Matmul -> Add -> Result
        auto matmul_add = opp::wrap_type<ov::op::v1::Add>({matmul, opp::any_input()});
        // Matmul -> Transpose -> Result
        auto matmul_transpose = opp::wrap_type<ov::op::v1::Transpose>({matmul, opp::any_input()});
        //  Matmul -> Convert -> Result
        auto matmul_convert = opp::wrap_type<ov::op::v0::Convert>({matmul});
        // MatMul -> Divide -> Tanh -> Multiply -> Result
        auto div = opp::wrap_type<ov::op::v1::Multiply, ov::op::v1::Divide>({matmul, opp::any_input()});
        auto tanh = opp::wrap_type<ov::op::v0::Tanh>({div});
        auto matmul_multiply = opp::wrap_type<ov::op::v1::Multiply>({tanh, opp::any_input()});

        auto last_op = std::make_shared<opp::op::Or>(ov::OutputVector{matmul->output(0),
                                                                      matmul_add->output(0),
                                                                      matmul_transpose->output(0),
                                                                      matmul_convert->output(0),
                                                                      matmul_multiply->output(0)});
        auto res = opp::wrap_type<ov::op::v0::Result>({last_op->output(0)});

        auto callback = [=, &lm_head_model, &drop_result, &output_embeds_name](opp::Matcher& m) {
            auto& node_to_output = m.get_pattern_value_map();

            auto matched_matmul =
                std::static_pointer_cast<ov::op::v0::MatMul>(node_to_output.at(matmul).get_node_shared_ptr());
            auto matched_result =
                std::static_pointer_cast<ov::op::v0::Result>(node_to_output.at(res).get_node_shared_ptr());

            std::shared_ptr<ov::Node> matched_node_last_op = matched_matmul;
            for (const auto& p : {matmul_add, matmul_transpose, matmul_convert, matmul_multiply}) {
                const auto it = node_to_output.find(p);
                if (it != node_to_output.end()) {
                    matched_node_last_op = it->second.get_node_shared_ptr();
                    break;
                }
            }

            // Skip Result nodes that are not logits.
            // Note: We can check that Result's output name is "logits" and it will be a
            //       sufficiently reliable check for finding exatly logits output, because:
            //       1. LLMInferRequest always rely on "logits" name to get logits from
            ///         prefill/kvcache models.
            //       2. - Following Exporter configs: OnnxConfig, OnnxConfigWithPast,
            //            TextDecoderOnnxConfig and TextDecoderWithPositionIdsOnnxConfig
            //            from optimum-onnx name LLM output with "logits".
            //          - Most of optimum-intel OpenVINO Exporter configs are derived
            //            from the configs above.
            //          - optimum-intel `export()` function set names for output tensors
            //            from Exporter config:
            //            https://github.com/huggingface/optimum-intel/blob/main/optimum/exporters/openvino/convert.py#L442-L445
            if (matched_result->output(0).get_names().count(ov::npuw::LLMCompiledModel::layer_names::logits) == 0) {
                return false;
            }

            // Cut point:
            auto matmul_first_source = matched_matmul->input(0).get_source_output();

            // Reuse any Result already attached to the cut point (e.g. OmniThinker exposes
            // the pre-head embeddings as a model output); otherwise repurpose the matched
            // logits Result as the output-embeddings Result of the original model.
            std::shared_ptr<ov::op::v0::Result> embeds_result;
            for (const auto& consumer : matmul_first_source.get_target_inputs()) {
                embeds_result = ov::as_type_ptr<ov::op::v0::Result>(consumer.get_node()->shared_from_this());
                if (embeds_result) {
                    break;
                }
            }
            if (embeds_result) {
                // Reroute to keep the model valid; the outer ModelPass drops the matched Result.
                matched_result->input(0).replace_source_output(matmul_first_source);
                drop_result = matched_result;
            } else {
                matched_result->input(0).replace_source_output(matmul_first_source);
                // FIXME: Somehow for KVCache model result output gets renamed in
                //        ICompiledModel::ICompiledModel().
                //        As a WA, setting the same name to output from MatMul
                //        avoids the issue.
                matmul_first_source.set_names({ov::npuw::LLMCompiledModel::layer_names::output_embeds});
                matched_result->output(0).set_names({ov::npuw::LLMCompiledModel::layer_names::output_embeds});
                embeds_result = matched_result;
            }
            embeds_result->validate_and_infer_types();
            OPENVINO_ASSERT(!embeds_result->output(0).get_names().empty(), "Output embeds result must have a name.");
            output_embeds_name = *embeds_result->output(0).get_names().begin();

            // Create an additional model after cut point:
            auto new_param = std::make_shared<ov::op::v0::Parameter>(matmul_first_source.get_element_type(),
                                                                     matmul_first_source.get_partial_shape());
            new_param->output(0).add_names({ov::npuw::LLMCompiledModel::layer_names::output_embeds});
            matched_matmul->input(0).replace_source_output(new_param);
            auto new_result = std::make_shared<ov::op::v0::Result>(matched_node_last_op);
            lm_head_model =
                std::make_shared<ov::Model>(ov::OutputVector{new_result->output(0)}, ov::ParameterVector{new_param});

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

CutLMHead::CutLMHead(std::shared_ptr<ov::Model>& lm_head_model,
                     std::string& output_embeds_name)
    : m_lm_head_model(lm_head_model),
      m_output_embeds_name(output_embeds_name) {}

bool CutLMHead::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::shared_ptr<ov::op::v0::Result> drop_result;
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<CutLMHeadMatcher>(m_lm_head_model, drop_result, m_output_embeds_name);
    rewr.run_on_model(model);
    if (drop_result) {
        model->remove_result(drop_result);
    }
    return m_lm_head_model != nullptr;
}

}  // namespace ov::npuw
