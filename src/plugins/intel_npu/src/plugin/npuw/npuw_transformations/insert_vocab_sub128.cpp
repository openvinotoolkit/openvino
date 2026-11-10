// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_vocab_sub128.hpp"

#include <memory>
#include <vector>

#include "../logging.hpp"
#include "cut_lm_head.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"

namespace opp = ov::pass::pattern;

namespace {

// `subtract` is the dequantization Subtract(Convert(W_u8), Convert(Z_u8)) of the LM head.
void insert_sub128_shifts(const std::shared_ptr<ov::op::v1::Subtract>& subtract) {
    const auto weight_convert = subtract->input_value(0);
    const auto zerop_convert = subtract->input_value(1);
    const auto is_u8_convert = [](const ov::Output<ov::Node>& value) {
        const auto convert = ov::as_type_ptr<ov::op::v0::Convert>(value.get_node_shared_ptr());
        return convert && convert->get_input_element_type(0) == ov::element::u8;
    };
    OPENVINO_ASSERT(is_u8_convert(weight_convert) && is_u8_convert(zerop_convert),
                    "InsertVocabSub128: both inputs of '",
                    subtract->get_friendly_name(),
                    "' must be Convert ops from u8");
    OPENVINO_ASSERT(weight_convert.get_element_type() == zerop_convert.get_element_type(),
                    "InsertVocabSub128: weight and zero point of '",
                    subtract->get_friendly_name(),
                    "' must be converted to one element type");
    const auto compute_type = weight_convert.get_element_type();
    const auto shift = ov::op::v0::Constant::create(compute_type, ov::Shape{}, {128});

    ov::mark_as_decompression(weight_convert.get_node_shared_ptr());
    ov::mark_as_decompression(zerop_convert.get_node_shared_ptr());

    const auto shifted_weight = std::make_shared<ov::op::v1::Subtract>(weight_convert, shift);
    const auto shifted_zerop = std::make_shared<ov::op::v1::Subtract>(zerop_convert, shift);
    shifted_weight->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
    shifted_zerop->get_rt_info()[ov::npuw::NPUW_SUB128_SHIFT_RT_INFO] = true;
    subtract->input(0).replace_source_output(shifted_weight);
    subtract->input(1).replace_source_output(shifted_zerop);
}

// diagnostics warnings on OPENVINO_MATCHER_PASS_RTTI() definition: visibility hidden
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wattributes"
#endif

class InsertVocabSub128Matcher final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::npuw::InsertVocabSub128Matcher");

    InsertVocabSub128Matcher() {
        const auto qweight = opp::wrap_type<ov::op::v0::Constant>();
        const auto qzerop = opp::wrap_type<ov::op::v0::Constant>();
        const auto qcoeff = opp::wrap_type<ov::op::v0::Constant>();
        const auto qweight_convert = opp::wrap_type<ov::op::v0::Convert>({qweight});
        const auto qzerop_convert = opp::wrap_type<ov::op::v0::Convert>({qzerop});
        const auto qsub = opp::wrap_type<ov::op::v1::Subtract>({qweight_convert, qzerop_convert});
        const auto qscale = opp::wrap_type<ov::op::v1::Multiply>({qsub, qcoeff});
        const auto qconvert = opp::optional<ov::op::v0::Convert>({qscale});
        const auto hidden = opp::any_input();
        const auto matmul = opp::wrap_type<ov::op::v0::MatMul>({hidden, qconvert});

        // Only vocabulary MatMuls that form an LM-head result are eligible for Sub128 insertion.
        const ov::npuw::patterns::LMHeadTerminal terminal(matmul);
        const auto result = opp::wrap_type<ov::op::v0::Result>({terminal.output});

        auto callback = [=](opp::Matcher& matcher) {
            const auto& values = matcher.get_pattern_value_map();
            const auto weight = values.at(qweight).get_node_shared_ptr();
            const auto zerop = values.at(qzerop).get_node_shared_ptr();
            const auto scale = values.at(qcoeff).get_node_shared_ptr();
            const auto subtract = std::static_pointer_cast<ov::op::v1::Subtract>(values.at(qsub).get_node_shared_ptr());
            const auto matched_matmul =
                std::static_pointer_cast<ov::op::v0::MatMul>(values.at(matmul).get_node_shared_ptr());
            const auto matched_result =
                std::static_pointer_cast<ov::op::v0::Result>(values.at(result).get_node_shared_ptr());

            if (matched_result->get_rt_info().count("manually_added_output")) {
                return false;
            }

            // Only the [V, H] layout (transpose_b, scale [V, 1]) can share storage with the
            // embedding table, so DetectVocabSharing never opens the gate for another layout.
            const auto scale_shape = std::static_pointer_cast<ov::op::v0::Constant>(scale)->get_shape();
            const bool vocab_layout = scale_shape.size() == 2 && scale_shape[1] == 1 &&
                                      !matched_matmul->get_transpose_a() && matched_matmul->get_transpose_b();

            if (weight->get_element_type() != ov::element::u8 || zerop->get_element_type() != ov::element::u8 ||
                weight->get_shape().size() != 2 || !vocab_layout) {
                return false;
            }

            insert_sub128_shifts(subtract);
            return true;
        };

        register_matcher(std::make_shared<opp::Matcher>(result, "InsertVocabSub128"), std::move(callback));
    }
};

#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

}  // namespace

ov::npuw::InsertVocabSub128::InsertVocabSub128() {
    add_matcher<InsertVocabSub128Matcher>();
}

namespace {

struct Vocab {
    std::shared_ptr<ov::op::v0::Constant> weight;
    std::shared_ptr<ov::op::v0::Constant> zerop;  // nullptr if absent
    std::shared_ptr<ov::op::v0::Constant> scale;  // nullptr if absent
};

bool same_storage(const std::shared_ptr<ov::op::v0::Constant>& lhs, const std::shared_ptr<ov::op::v0::Constant>& rhs) {
    if (!lhs || !rhs) {
        return !lhs && !rhs;
    }
    return lhs->get_element_type() == rhs->get_element_type() && lhs->get_shape() == rhs->get_shape() &&
           lhs->get_data_ptr() == rhs->get_data_ptr();
}

class CollectVocabCandidates final : public ov::pass::MatcherPass {
public:
    // `matched` counts every matched root, also roots that are not stored as candidates.
    CollectVocabCandidates(std::vector<Vocab>& vocabs, std::size_t& matched, bool lm_head) {
        const auto asymmetric_weight = opp::wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
            return output.get_element_type() == ov::element::u8 && output.get_shape().size() == 2;
        });
        const auto symmetric_weight = opp::wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
            return (output.get_element_type() == ov::element::i8 || output.get_element_type() == ov::element::i4) &&
                   output.get_shape().size() == 2;
        });
        const auto full_precision_weight = opp::wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
            const auto type = output.get_element_type();
            return (type == ov::element::f16 || type == ov::element::f32 || type == ov::element::bf16) &&
                   output.get_shape().size() == 2;
        });
        const auto zerop = opp::wrap_type<ov::op::v0::Constant>([](const ov::Output<ov::Node>& output) {
            return output.get_element_type() == ov::element::u8 && output.get_shape().size() == 2;
        });
        const auto scale = opp::wrap_type<ov::op::v0::Constant>();
        const auto asymmetric_convert = opp::wrap_type<ov::op::v0::Convert>({asymmetric_weight});
        const auto zerop_convert = opp::wrap_type<ov::op::v0::Convert>({zerop});
        const auto subtract = opp::wrap_type<ov::op::v1::Subtract>({asymmetric_convert, zerop_convert});
        const auto symmetric_convert = opp::wrap_type<ov::op::v0::Convert>({symmetric_weight});
        const auto dequantized = std::make_shared<opp::op::Or>(ov::OutputVector{subtract, symmetric_convert});
        const auto scaled = opp::wrap_type<ov::op::v1::Multiply>({dequantized, scale});
        const auto weight = std::make_shared<opp::op::Or>(ov::OutputVector{scaled, full_precision_weight});
        const auto converted = opp::optional<ov::op::v0::Convert>({weight});

        std::shared_ptr<ov::Node> root;
        std::shared_ptr<ov::Node> matmul;
        if (lm_head) {
            // Both weight layouts match here, so that each LM head that InsertVocabSub128 can mark
            // is counted. Only the [V, H] layout can share storage with the embedding table.
            matmul = opp::wrap_type<ov::op::v0::MatMul>(
                {opp::any_input(), converted},
                [](const ov::Output<ov::Node>& output) {
                    const auto node = ov::as_type_ptr<ov::op::v0::MatMul>(output.get_node_shared_ptr());
                    return !node->get_transpose_a();
                });
            const ov::npuw::patterns::LMHeadTerminal terminal(matmul);
            root = opp::wrap_type<ov::op::v0::Result>({terminal.output}, [](const ov::Output<ov::Node>& output) {
                return output.get_node_shared_ptr()->get_rt_info().count("manually_added_output") == 0;
            });
        } else {
            root = opp::wrap_type<ov::op::v8::Gather>({converted, opp::any_input(), opp::any_input()});
        }

        register_matcher(std::make_shared<opp::Matcher>(root, lm_head ? "CollectLmHeadVocab" : "CollectEmbeddingVocab"),
                         [=, &vocabs, &matched](opp::Matcher& matcher) {
                             const auto& values = matcher.get_pattern_value_map();
                             ++matched;
                             if (lm_head) {
                                 const auto matched_matmul =
                                     ov::as_type_ptr<ov::op::v0::MatMul>(values.at(matmul).get_node_shared_ptr());
                                 if (!matched_matmul->get_transpose_b()) {
                                     return false;
                                 }
                             }
                             const auto constant = [&values](const std::shared_ptr<ov::Node>& pattern) {
                                 return ov::as_type_ptr<ov::op::v0::Constant>(values.at(pattern).get_node_shared_ptr());
                             };
                             if (values.count(full_precision_weight)) {
                                 vocabs.push_back({constant(full_precision_weight), nullptr, nullptr});
                             } else if (values.count(asymmetric_weight)) {
                                 const auto matched_weight = constant(asymmetric_weight);
                                 const auto matched_zerop = constant(zerop);
                                 if (matched_zerop->get_shape()[0] != matched_weight->get_shape()[0]) {
                                     return false;
                                 }
                                 vocabs.push_back({matched_weight, matched_zerop, constant(scale)});
                             } else {
                                 vocabs.push_back({constant(symmetric_weight), nullptr, constant(scale)});
                             }
                             return false;
                         });
    }
};

void log_vocab_names(const char* label, const Vocab& vocab) {
    LOG_WARN(label << " weight='" << vocab.weight->get_friendly_name() << "', zero_point='"
                   << (vocab.zerop ? vocab.zerop->get_friendly_name() : "<none>") << "', scale='"
                   << (vocab.scale ? vocab.scale->get_friendly_name() : "<none>") << "'");
}

}  // namespace

ov::npuw::DetectVocabSharing::DetectVocabSharing(bool& shared) : m_shared(shared) {}

bool ov::npuw::DetectVocabSharing::run_on_model(const std::shared_ptr<ov::Model>& model) {
    m_shared = false;

    std::vector<Vocab> embedding_vocabs;
    std::vector<Vocab> lm_head_vocabs;
    std::size_t embedding_count = 0;
    std::size_t lm_head_count = 0;
    ov::pass::GraphRewrite rewrite;
    rewrite.add_matcher<CollectVocabCandidates>(embedding_vocabs, embedding_count, false);
    rewrite.add_matcher<CollectVocabCandidates>(lm_head_vocabs, lm_head_count, true);
    rewrite.run_on_model(model);

    // InsertVocabSub128 marks every LM head that it matches. With more than one LM head, a head
    // that does not share the embedding vocabulary would be marked too, so detection stops here.
    if (lm_head_count > 1) {
        LOG_WARN("NPUW_LLM_VOCAB_ASYM_SHARED is enabled, but the model has "
                 << lm_head_count
                 << " LM-head outputs. Only one LM head is supported. Sub128 graph transformations are skipped.");
        return false;
    }

    for (const auto& embedding : embedding_vocabs) {
        for (const auto& lm_head : lm_head_vocabs) {
            if (same_storage(embedding.weight, lm_head.weight) && same_storage(embedding.zerop, lm_head.zerop) &&
                same_storage(embedding.scale, lm_head.scale)) {
                m_shared = true;
                return false;
            }
        }
    }

    if (embedding_vocabs.empty() && lm_head_vocabs.empty()) {
        LOG_WARN("NPUW_LLM_VOCAB_ASYM_SHARED is enabled, but no embedding Gather or LM-head MatMul "
                 "vocabulary was detected. Sub128 graph transformations are skipped.");
    } else if (embedding_vocabs.empty()) {
        LOG_WARN("NPUW_LLM_VOCAB_ASYM_SHARED is enabled, but no embedding Gather vocabulary was detected. "
                 "Sub128 graph transformations are skipped.");
        log_vocab_names("LM-head candidate:", lm_head_vocabs.front());
    } else if (lm_head_vocabs.empty()) {
        LOG_WARN("NPUW_LLM_VOCAB_ASYM_SHARED is enabled, but no LM-head MatMul vocabulary was detected. "
                 "Sub128 graph transformations are skipped.");
        log_vocab_names("Embedding candidate:", embedding_vocabs.front());
    } else {
        LOG_WARN("NPUW_LLM_VOCAB_ASYM_SHARED is enabled, but embedding and LM-head vocabularies are "
                 "not backed by the same storage. Sub128 graph transformations are skipped.");
        log_vocab_names("Embedding candidate:", embedding_vocabs.front());
        log_vocab_names("LM-head candidate:", lm_head_vocabs.front());
    }
    return false;
}
