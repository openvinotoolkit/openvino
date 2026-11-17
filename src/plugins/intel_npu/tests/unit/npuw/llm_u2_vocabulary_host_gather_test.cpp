// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// The u2 group-quantized vocabulary of the GGUF frontend is moved to the host gather by several passes that run
// in different places: RestoreU2GatherDequant and BroadcastU2ZeroPoint in LLMCompiledModel, StripNoopGatherIdsReshape
// and DQLiftGatherSymGQ in pre_load_transform, HostGatherDQ in partitioning. Each pass has its own test; this test
// checks that they connect on the embedding of an LLM.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "llm_test_helpers.hpp"
#include "npuw_transformations/strip_noop_gather_ids_reshape.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "partitioning/patterns/opt.hpp"

namespace {

using namespace ov;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::RecordingFactory;

constexpr size_t kVocab = 256;
constexpr size_t kGroups = 2;
constexpr size_t kPackedGroupSize = 256;                    // bytes, four 2-bit codes each
constexpr size_t kHidden = kGroups * kPackedGroupSize * 4;  // the host gather requires the embedding size >= 2048
constexpr float kZeroPoint = 1.f;
const char* const kEmbeddingName = "model.embed_tokens";

uint8_t packed_byte(const size_t index) {
    return static_cast<uint8_t>((index * 37 + 11) & 0xFF);
}

float scale_value(const size_t index) {
    return 0.25f * static_cast<float>(index % 7 + 1);
}

// The embedding of the GGUF frontend: Gather over the packed u8 rows, bitwise unpacking of the 2-bit codes, the zero
// point as Convert(Constant u8), Gather over the scales, and a Reshape + Squeeze pair on the input ids.
Output<Node> make_frontend_embedding(const Output<Node>& input_ids) {
    std::vector<uint8_t> bytes(kVocab * kGroups * kPackedGroupSize);
    for (size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = packed_byte(i);
    }
    std::vector<float16> scales(kVocab * kGroups);
    for (size_t i = 0; i < scales.size(); ++i) {
        scales[i] = float16(scale_value(i));
    }
    const auto packed = op::v0::Constant::create(element::u8, Shape{kVocab, kGroups, kPackedGroupSize}, bytes);
    const auto scales_const = op::v0::Constant::create(element::f16, Shape{kVocab, kGroups, 1}, scales);
    const auto axis = op::v0::Constant::create(element::i32, Shape{}, {0});

    const auto ids_shape = std::make_shared<op::v0::Concat>(
        OutputVector{op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 1}),
                     std::make_shared<op::v3::ShapeOf>(input_ids, element::i64)},
        0);
    const auto ids =
        std::make_shared<op::v0::Squeeze>(std::make_shared<op::v1::Reshape>(input_ids, ids_shape, false),
                                          op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{0, 1}));

    const auto widened =
        std::make_shared<op::v0::Convert>(std::make_shared<op::v8::Gather>(packed, ids, axis), element::i32);
    const auto expanded = std::make_shared<op::v1::Reshape>(
        widened,
        op::v0::Constant::create(element::i64, Shape{5}, std::vector<int64_t>{0, 0, kGroups, kPackedGroupSize, 1}),
        true);
    const auto shifted = std::make_shared<op::v15::BitwiseRightShift>(
        expanded,
        op::v0::Constant::create(element::i32, Shape{4}, std::vector<int32_t>{0, 2, 4, 6}));
    const auto codes =
        std::make_shared<op::v13::BitwiseAnd>(shifted, op::v0::Constant::create(element::i32, Shape{}, {3}));
    const auto grouped = std::make_shared<op::v1::Reshape>(
        codes,
        op::v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{0, 0, kGroups, kPackedGroupSize * 4}),
        true);
    const auto values = std::make_shared<op::v0::Convert>(grouped, element::f16);
    const auto zero_point = std::make_shared<op::v0::Convert>(
        op::v0::Constant::create(element::u8, Shape{}, {static_cast<uint8_t>(kZeroPoint)}),
        element::f16);
    const auto centered = std::make_shared<op::v1::Subtract>(values, zero_point);
    const auto dequantized =
        std::make_shared<op::v1::Multiply>(centered, std::make_shared<op::v8::Gather>(scales_const, ids, axis));
    const auto restored = std::make_shared<op::v1::Reshape>(
        dequantized,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, kHidden}),
        true);
    const auto embeds = std::make_shared<op::v0::Convert>(restored, element::f32);
    embeds->set_friendly_name(kEmbeddingName);
    return embeds->output(0);
}

// An LLM whose embedding is the frontend form above
std::shared_ptr<Model> build_llm_with_u2_vocabulary() {
    auto cfg = ov::test::npuw::make_test_model_config();
    cfg.hidden_size = kHidden;
    cfg.vocab_size = kVocab;
    cfg.intermediate_size = 64;
    const auto model = ov::test::npuw::build_llm_test_model(cfg);

    const auto ids = model->input("input_ids");
    for (const auto& node : model->get_ordered_ops()) {
        const auto gather = ov::as_type_ptr<op::v8::Gather>(node);
        if (gather && ov::is_type<op::v0::Constant>(gather->input_value(0).get_node())) {
            ov::replace_output_update_name(gather->output(0), make_frontend_embedding(ids));
            break;
        }
    }
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<Node> find_by_name(const std::shared_ptr<Model>& model, const std::string& name) {
    for (const auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == name) {
            return node;
        }
    }
    return nullptr;
}

int64_t token_id(const size_t position) {
    return static_cast<int64_t>((position * 7 + 3) % kVocab);
}

// The embedding of a token as the frontend form dequantizes it: LSB-first u2 codes of the packed row
float expected_embedding(const int64_t token, const size_t column) {
    const size_t group = column / (kPackedGroupSize * 4);
    const size_t position = column % (kPackedGroupSize * 4);
    const size_t group_index = static_cast<size_t>(token) * kGroups + group;
    const int code = (packed_byte(group_index * kPackedGroupSize + position / 4) >> (2 * (position % 4))) & 3;
    return (static_cast<float>(code) - kZeroPoint) * scale_value(group_index);
}

Tensor evaluate_embedding(const Output<Node>& embedding, const std::shared_ptr<op::v0::Parameter>& ids) {
    Tensor ids_tensor(element::i64, ids->get_shape());
    for (size_t i = 0; i < ids_tensor.get_size(); ++i) {
        ids_tensor.data<int64_t>()[i] = token_id(i);
    }
    const auto sub_model = std::make_shared<Model>(OutputVector{embedding}, ParameterVector{ids});
    TensorVector outputs{Tensor()};
    EXPECT_TRUE(sub_model->evaluate(outputs, TensorVector{ids_tensor}));
    return outputs.front();
}

}  // namespace

TEST(LLMU2VocabularyHostGatherTest, VocabularyReachesTheHostGatherAndKeepsTheEmbeddings) {
    const auto model = build_llm_with_u2_vocabulary();

    RecordingFactory recorder;
    const ov::AnyMap props{{"NPUW_LLM", "YES"},
                           {"NPUW_LLM_MAX_PROMPT_LEN", "128"},
                           {"NPUW_LLM_MIN_RESPONSE_LEN", "64"}};
    const auto compiled = std::make_unique<npuw::LLMCompiledModel>(model->clone(),
                                                                   std::make_shared<NullPlugin>(),
                                                                   props,
                                                                   recorder.make_factory());
    const auto* prefill_call = recorder.find_suffix("_prefill");
    ASSERT_NE(prefill_call, nullptr);
    const auto prefill = prefill_call->model->clone();
    const auto ids = ov::as_type_ptr<op::v0::Parameter>(prefill->input("input_ids").get_node_shared_ptr());
    ASSERT_TRUE(ids->get_output_partial_shape(0).is_static());

    // LLMCompiledModel: the packed gather is a Gather over the dequantized u2 vocabulary
    const auto embedding = find_by_name(prefill, kEmbeddingName);
    ASSERT_NE(embedding, nullptr);
    ASSERT_TRUE(ov::is_type<op::v8::Gather>(embedding));
    const auto ops = prefill->get_ordered_ops();
    EXPECT_TRUE(std::none_of(ops.begin(), ops.end(), [](const std::shared_ptr<Node>& node) {
        return ov::is_type<op::v15::BitwiseRightShift>(node);
    }));
    const auto embedding_consumers = embedding->output(0).get_target_inputs();
    ASSERT_FALSE(embedding_consumers.empty());

    // pre_load_transform: the ids reach the Gather from the Parameter, and the Gather is lifted above the
    // dequantization
    npuw::StripNoopGatherIdsReshape().run_on_model(prefill);
    pass::GraphRewrite lift;
    lift.add_matcher<npuw::patterns::opt::DQLiftGatherSymGQ>();
    lift.run_on_model(prefill);
    prefill->validate_nodes_and_infer_types();

    const auto lifted = embedding_consumers.begin()->get_source_output();
    const auto lifted_model = std::make_shared<Model>(OutputVector{lifted}, ParameterVector{ids});
    size_t gathers = 0;
    for (const auto& node : lifted_model->get_ordered_ops()) {
        if (ov::is_type<op::v8::Gather>(node)) {
            ++gathers;
            auto source = node->input_value(1);
            if (ov::is_type<op::v0::Convert>(source.get_node())) {
                source = source.get_node()->input_value(0);
            }
            EXPECT_EQ(source.get_node(), ids.get()) << node->get_friendly_name();
        }
    }
    EXPECT_EQ(gathers, 3u);

    // The embeddings are the ones that the frontend form dequantizes
    const auto actual = evaluate_embedding(lifted, ids);
    ASSERT_EQ(actual.get_element_type(), element::f32);
    ASSERT_EQ(actual.get_size(), ids->get_shape()[1] * kHidden);
    for (size_t token = 0; token < ids->get_shape()[1]; ++token) {
        for (size_t column = 0; column < kHidden; ++column) {
            ASSERT_EQ(actual.data<float>()[token * kHidden + column], expected_embedding(token_id(token), column))
                << "token " << token << " column " << column;
        }
    }

    // Partitioning: the closures are Parameters, and HostGatherDQ moves the Gathers to the host
    ParameterVector closures;
    for (const auto& node : lifted_model->get_ordered_ops()) {
        const auto constant = ov::as_type_ptr<op::v0::Constant>(node);
        if (constant && constant->get_shape().size() == 3 && constant->get_shape()[0] == kVocab) {
            const auto closure =
                std::make_shared<op::v0::Parameter>(constant->get_element_type(), constant->get_shape());
            ov::replace_node(constant, closure);
            closures.push_back(closure);
        }
    }
    ASSERT_EQ(closures.size(), 3u);
    lifted_model->add_parameters(closures);

    npuw::patterns::opt::Context ctx(lifted_model, 1);
    pass::GraphRewrite host_gather;
    host_gather.add_matcher<npuw::patterns::opt::HostGatherDQ>(std::ref(ctx));
    host_gather.run_on_model(lifted_model);

    ASSERT_TRUE(ctx.params_to_gather.has_value());
    ASSERT_EQ(ctx.params_to_unpack.size(), 1u);
    const auto& unpack = ctx.params_to_unpack.begin()->second;
    EXPECT_NE(unpack.w, nullptr);
    EXPECT_NE(unpack.z, nullptr);
    EXPECT_NE(unpack.s, nullptr);
    EXPECT_EQ(ctx.params_to_unpack.begin()->first->get_shape(), (Shape{kVocab, kHidden}));
    const auto host_gathered_ops = lifted_model->get_ordered_ops();
    EXPECT_TRUE(std::none_of(host_gathered_ops.begin(), host_gathered_ops.end(), [](const std::shared_ptr<Node>& node) {
        return ov::is_type<op::v8::Gather>(node);
    }));
}
