// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "model_serializer.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/parameter.hpp"

// serializeIR()'s common pipeline decomposes GroupQueryAttention unconditionally, since a stale compiler
// can't be detected at runtime (see model_serializer.cpp). These tests check it fires wherever the operator is,
// including inside sub-graphs, and leaves everything else untouched.
namespace {

using namespace ov::op;
using intel_npu::compiler_utils::serializeIR;

constexpr int64_t NUM_HEADS = 24;
constexpr int64_t KV_NUM_HEADS = 8;
constexpr int64_t HEAD_SIZE = 128;
constexpr int64_t SEQ_LEN = 128;
constexpr int64_t CACHE_LEN = 1024;
constexpr int64_t HALF_ROTARY_DIM = HEAD_SIZE / 2;

// A high, opset-agnostic compiler version and supported-opset so only the GQA pass is under test: the other two
// compatibility passes stay disabled (see run_common_pipeline's own gates in model_serializer.cpp).
ze_graph_compiler_version_info_t modernCompilerVersion() {
    return {/*major=*/99, /*minor=*/0};
}

size_t countGQA(const std::shared_ptr<ov::Model>& model) {
    size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        if (ov::is_type<internal::GroupQueryAttention>(node)) {
            ++count;
        }
    }
    return count;
}

void applyCompatibilityPasses(const std::shared_ptr<ov::Model>& model) {
    const auto isOptionValueSupportedByCompiler = [](const std::string&, const std::optional<std::string>&) {
        return true;
    };
    // ALL_WEIGHTS_COPY sidesteps the AUTO-version compiler-support query; this test only cares about the
    // decomposition side effect serializeIR's common pipeline has on "model", not the returned buffer.
    serializeIR(model,
                modernCompilerVersion(),
                /*supportedOpsetVersion=*/std::numeric_limits<uint32_t>::max(),
                ov::intel_npu::ModelSerializerVersion::ALL_WEIGHTS_COPY,
                isOptionValueSupportedByCompiler);
}

std::shared_ptr<v0::Constant> makeEmptyPlaceholder() {
    return v0::Constant::create(ov::element::dynamic, ov::Shape{0}, {});
}

enum class Optionals {
    None,        // only the 9 inputs the operator actually uses: minimal arity, no placeholders
    Trailing,    // the 7 absent optional inputs appended as placeholders (what the ONNX frontend emits)
    InteriorGap  // a real head_sink at its fixed position, so the placeholders before it cannot be dropped
};

/// All shapes are static: the NPU only compiles static-shaped graphs.
std::shared_ptr<ov::Model> makeGQAModel(const Optionals optionals, const int64_t localWindowSize = -1) {
    const auto query = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, NUM_HEADS, SEQ_LEN, HEAD_SIZE});
    const auto key = std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, KV_NUM_HEADS, SEQ_LEN, HEAD_SIZE});
    const auto value =
        std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, KV_NUM_HEADS, SEQ_LEN, HEAD_SIZE});
    const auto pastKey =
        std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, KV_NUM_HEADS, CACHE_LEN, HEAD_SIZE});
    const auto pastValue =
        std::make_shared<v0::Parameter>(ov::element::f16, ov::Shape{1, KV_NUM_HEADS, CACHE_LEN, HEAD_SIZE});
    const auto seqLensK = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{1, 1});
    const auto totalSeqLen = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto cosCache =
        v0::Constant::create(ov::element::f16, ov::Shape{CACHE_LEN, HALF_ROTARY_DIM}, std::vector<float>{0.f});
    const auto sinCache =
        v0::Constant::create(ov::element::f16, ov::Shape{CACHE_LEN, HALF_ROTARY_DIM}, std::vector<float>{0.f});

    ov::OutputVector inputs{query, key, value, pastKey, pastValue, seqLensK, totalSeqLen, cosCache, sinCache};

    if (optionals == Optionals::Trailing) {
        // position_ids, attention_bias, head_sink, k_scale, v_scale and the two reserved QK-Norm slots
        for (size_t index = 0; index < 7; ++index) {
            inputs.push_back(makeEmptyPlaceholder());
        }
    } else if (optionals == Optionals::InteriorGap) {
        inputs.push_back(makeEmptyPlaceholder());  // position_ids
        inputs.push_back(makeEmptyPlaceholder());  // attention_bias
        inputs.push_back(
            v0::Constant::create(ov::element::f16, ov::Shape{NUM_HEADS}, std::vector<float>{0.f}));  // head_sink
    }

    const auto gqa = std::make_shared<internal::GroupQueryAttention>(inputs,
                                                                     NUM_HEADS,
                                                                     KV_NUM_HEADS,
                                                                     0.f,
                                                                     true,   // do_rotary
                                                                     false,  // rotary_interleaved
                                                                     0,      // kv_cache_bit_width
                                                                     internal::GroupQueryAttentionQuantType::NONE,
                                                                     internal::GroupQueryAttentionQuantType::NONE,
                                                                     localWindowSize,
                                                                     false,  // sliding_window_cache
                                                                     false   // smooth_softmax
    );

    return std::make_shared<ov::Model>(
        gqa->outputs(),
        ov::ParameterVector{query, key, value, pastKey, pastValue, seqLensK, totalSeqLen});
}

TEST(GQACompilerCompatibility, TrailingPlaceholdersAreDecomposed) {
    const auto model = makeGQAModel(Optionals::Trailing);
    ASSERT_EQ(countGQA(model), 1u);

    applyCompatibilityPasses(model);
    EXPECT_EQ(countGQA(model), 0u);
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(GQACompilerCompatibility, InteriorPlaceholderIsDecomposed) {
    const auto model = makeGQAModel(Optionals::InteriorGap);
    ASSERT_EQ(countGQA(model), 1u);

    applyCompatibilityPasses(model);
    EXPECT_EQ(countGQA(model), 0u);
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

// No placeholders here (e.g. a sliding-window export) - still must decompose, since a stale compiler would just
// ignore local_window_size and silently compute full causal attention instead of failing to compile.
TEST(GQACompilerCompatibility, MinimalArityWithNoPlaceholdersIsAlsoDecomposed) {
    const auto model = makeGQAModel(Optionals::None, /*localWindowSize=*/32);
    ASSERT_EQ(countGQA(model), 1u);

    applyCompatibilityPasses(model);
    EXPECT_EQ(countGQA(model), 0u);
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(GQACompilerCompatibility, GroupQueryAttentionInsideSubGraphIsDecomposed) {
    const auto thenBody = makeGQAModel(Optionals::None);
    const auto elseBody = makeGQAModel(Optionals::None);

    ov::ParameterVector outerParameters;
    for (const auto& parameter : thenBody->get_parameters()) {
        outerParameters.push_back(
            std::make_shared<v0::Parameter>(parameter->get_element_type(), parameter->get_partial_shape()));
    }

    const auto condition = v0::Constant::create(ov::element::boolean, ov::Shape{}, {true});
    const auto ifOp = std::make_shared<v8::If>(condition);
    ifOp->set_then_body(thenBody);
    ifOp->set_else_body(elseBody);
    for (size_t index = 0; index < outerParameters.size(); ++index) {
        ifOp->set_input(outerParameters[index], thenBody->get_parameters()[index], elseBody->get_parameters()[index]);
    }
    for (size_t index = 0; index < thenBody->get_results().size(); ++index) {
        ifOp->set_output(thenBody->get_results()[index], elseBody->get_results()[index]);
    }

    const auto model = std::make_shared<ov::Model>(ifOp->outputs(), outerParameters);
    ASSERT_EQ(countGQA(model), 0u) << "the operator lives in the sub-graphs, not in the top-level model";
    ASSERT_EQ(countGQA(thenBody), 1u);
    ASSERT_EQ(countGQA(elseBody), 1u);

    applyCompatibilityPasses(model);
    EXPECT_EQ(countGQA(thenBody), 0u);
    EXPECT_EQ(countGQA(elseBody), 0u);
}

TEST(GQACompilerCompatibility, ModelWithoutGroupQueryAttentionIsLeftAlone) {
    const auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{1, 3, 8, 8});
    const auto weights = v0::Constant::create(ov::element::f32, ov::Shape{3, 3, 1, 1}, std::vector<float>(9, 1.f));
    const auto convolution = std::make_shared<v1::Convolution>(input,
                                                               weights,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::Strides{1, 1});
    const auto model = std::make_shared<ov::Model>(convolution->outputs(), ov::ParameterVector{input});

    applyCompatibilityPasses(model);

    const auto ops = model->get_ordered_ops();
    EXPECT_TRUE(std::any_of(ops.begin(), ops.end(), [](const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<v1::Convolution>(node);
    }));
}

}  // namespace
