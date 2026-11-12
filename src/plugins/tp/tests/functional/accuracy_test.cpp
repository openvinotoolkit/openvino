// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Accuracy of the whole TP pipeline.
//
// Each case compiles one synthetic transformer block twice -- once on a single
// rank device, once on TP with the model split across several of them -- feeds both
// the same input and compares the outputs. Tensor parallelism is a pure
// refactoring of the computation, so the two must agree up to the reordering of
// the AllReduce summation.
//
// What varies between cases is the sharding: MLP shape, bias, weight layout,
// quantization group alignment, head split. Attention is here to make the
// block realistic, not as the subject -- these blocks use SDPA, and the
// PagedAttention formulation has its own suite in paged_attention_test.cpp.
//
// Needs two or more rank devices (rank_devices.hpp); the suite skips otherwise.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "rank_devices.hpp"
#include "tp_test_models.hpp"

namespace ov::tp::tests {
namespace {

constexpr size_t sequence_length = 16;

using Param = std::tuple<uint32_t, ov::element::Type>;

const std::vector<uint32_t> world_sizes = {2, 3, 4};
const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

class TPAccuracyTest : public ::testing::TestWithParam<Param> {
protected:
    static uint32_t world_size() {
        return std::get<0>(GetParam());
    }

    static ov::element::Type inference_precision() {
        return std::get<1>(GetParam());
    }

    void SetUp() override {
        if (const auto missing = missing_rank_devices(world_size()); !missing.empty())
            GTEST_SKIP() << missing;
    }

    static double tolerance() {
        return inference_precision() == ov::element::f16 ? 1e-2 : 1e-5;
    }

    /// The single-device reference and the TP model over the parameter's world size.
    std::pair<ov::InferRequest, ov::InferRequest> compile_both(const std::shared_ptr<ov::Model>& model) {
        return {core.compile_model(model, rank_device(0), precision_config()).create_infer_request(),
                core.compile_model(model, tp_device(world_size()), tp_config()).create_infer_request()};
    }

    /// Compiles `model` on a single rank device and on TP over the parameter's
    /// world size, runs both on `input` and returns the two outputs.
    ///
    /// The outputs are copied to host memory: an output tensor is owned by the
    /// infer request that produced it and does not outlive the pipelines torn
    /// down when this returns.
    std::pair<ov::Tensor, ov::Tensor> run_both(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input) {
        auto [reference, parallel] = compile_both(model);

        reference.set_input_tensor(input);
        parallel.set_input_tensor(input);
        reference.infer();
        parallel.infer();

        return {to_host(reference.get_output_tensor(0)), to_host(parallel.get_output_tensor(0))};
    }

    /// Compares two outputs at the tolerance the current precision warrants.
    static void expect_close(const ov::Tensor& reference, const ov::Tensor& parallel) {
        ov::test::utils::compare(reference, parallel, tolerance(), tolerance());
    }

    static ov::Tensor to_host(const ov::Tensor& tensor) {
        ov::Tensor host(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(host);
        return host;
    }

    /// Drops the trailing `tokens` positions from every KV cache state, the
    /// same read-slice-write GenAI performs between chat turns. KV cache
    /// states are [batch, kv_heads, seq, head_dim].
    static void trim_kv_cache(ov::InferRequest& request, size_t tokens) {
        constexpr size_t seq_axis = 2;
        for (auto&& state : request.query_state()) {
            auto cached = state.get_state();
            auto shape = cached.get_shape();
            ASSERT_GE(shape[seq_axis], tokens) << state.get_name();
            shape[seq_axis] -= tokens;

            ov::Tensor trimmed(cached.get_element_type(), shape);
            ov::Tensor(cached, ov::Coordinate(shape.size(), 0), ov::Coordinate(shape)).copy_to(trimmed);
            state.set_state(trimmed);
        }
    }

    static ov::Tensor make_input(const BlockConfig& config, size_t length = sequence_length, int32_t seed = 7) {
        // A narrow range keeps the u4 weights from saturating the activations,
        // so a real mismatch is not masked by both sides producing garbage.
        ov::test::utils::InputGenerateData data(-1, 2, 32, seed);
        return ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                       ov::Shape{1, length, config.hidden},
                                                       data);
    }

    static ov::AnyMap precision_config() {
        return {ov::hint::inference_precision(inference_precision())};
    }

    static ov::AnyMap tp_config() {
        return precision_config();
    }

    ov::Core core;
};

TEST_P(TPAccuracyTest, MatchesSingleDeviceOnGatedMlpBlock) {
    BlockConfig config;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

TEST_P(TPAccuracyTest, MatchesSingleDeviceOnSingleBranchMlpBlock) {
    BlockConfig config;
    config.gated_mlp = false;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

TEST_P(TPAccuracyTest, MatchesSingleDeviceOnBiasedProjections) {
    BlockConfig config;
    config.with_bias = true;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// A row-parallel projection sums partial products across ranks, so its bias
// must be added exactly once. Adding it on every rank would scale the bias by
// the world size, which only shows up once the collective has run.
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnUntransposedWeights) {
    BlockConfig config;
    config.transpose_b = false;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// Two generation steps over a stateful KV cache: the second one reads back what
// the first wrote, so a cache that was sharded inconsistently with the query
// heads shows up as a mismatch only on the second call.
TEST_P(TPAccuracyTest, MatchesSingleDeviceAcrossStatefulSteps) {
    BlockConfig config;
    config.stateful = true;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = compile_both(model);

    for (size_t step = 0; step < 2; ++step) {
        SCOPED_TRACE("step " + std::to_string(step));
        // Prefill, then a single-token generation step reusing the cache.
        auto input = make_input(config, step == 0 ? sequence_length : 1, static_cast<int32_t>(step) + 1);

        reference.set_input_tensor(input);
        parallel.set_input_tensor(input);
        reference.infer();
        parallel.infer();

        expect_close(reference.get_output_tensor(0), parallel.get_output_tensor(0));
    }
}

// GenAI drops the tail of the KV cache between chat turns by reading each
// state, slicing off the trailing tokens and writing it back. On TP every
// rank holds a slice of the kv heads, so the read has to gather and the write
// has to scatter; getting that wrong overwrites the other ranks' heads with
// rank 0's and the next step silently produces a different answer.
TEST_P(TPAccuracyTest, MatchesSingleDeviceAfterKvCacheTrim) {
    constexpr size_t trimmed_tokens = 4;

    BlockConfig config;
    config.stateful = true;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = compile_both(model);

    auto prefill = make_input(config, sequence_length, 1);
    reference.set_input_tensor(prefill);
    parallel.set_input_tensor(prefill);
    reference.infer();
    parallel.infer();

    trim_kv_cache(reference, trimmed_tokens);
    trim_kv_cache(parallel, trimmed_tokens);

    auto next = make_input(config, 1, 2);
    reference.set_input_tensor(next);
    parallel.set_input_tensor(next);
    reference.infer();
    parallel.infer();

    expect_close(reference.get_output_tensor(0), parallel.get_output_tensor(0));
}

// 448 is 14 quantization groups: three and four ranks get unequal shares of the
// MLP, the remainder on the lower ranks.
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnUnevenMlpSplit) {
    BlockConfig config;
    config.intermediate = 448;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// Five KV heads divide by none of the world sizes under test, so the ranks end
// up with different numbers of attention heads -- and with the remainder handed
// to the lower ranks, the last rank is the one that gets shorted.
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnUnevenHeadSplit) {
    BlockConfig config;
    config.num_kv_heads = 5;
    config.num_heads = 10;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// Three groups of 64 per rank in the MLP: the packed u4 zero points of the down projection are cut
// inside a byte on every world size (TPRankPassesNibbleTest).
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnZeroPointsCutInsideAByte) {
    BlockConfig config;
    config.group_size = 64;
    config.intermediate = 3 * 64 * world_size();
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// On four ranks a boundary of the O projection falls inside a quantization group, so its weight is
// cut by a Slice at run time (TPRankPassesRuntimeSliceTest); two and three ranks cut it whole.
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnAWeightSlicedAtRunTime) {
    BlockConfig config;
    config.group_size = 128;
    config.intermediate = 1536;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// Fewer KV heads than ranks: every rank keeps the one KV head and its share of the query heads,
// and the KV-cache state read and written whole takes one copy of each head.
BlockConfig one_kv_head() {
    BlockConfig config;
    config.num_kv_heads = 1;
    config.num_heads = 12;  // divides by every world size under test
    return config;
}

TEST_P(TPAccuracyTest, MatchesSingleDeviceWithOneKvHead) {
    auto config = one_kv_head();
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// Two KV heads on four ranks: ranks 0-1 keep head 0, ranks 2-3 head 1.
TEST_P(TPAccuracyTest, MatchesSingleDeviceWithTwoKvHeadsOnMoreRanks) {
    if (world_size() % 2 != 0)
        GTEST_SKIP() << "2 KV heads do not divide " << world_size() << " ranks";
    auto config = one_kv_head();
    config.num_kv_heads = 2;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

TEST_P(TPAccuracyTest, MatchesSingleDeviceWithOneKvHeadAfterKvCacheTrim) {
    auto config = one_kv_head();
    config.stateful = true;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = compile_both(model);

    auto prefill = make_input(config, sequence_length, 1);
    reference.set_input_tensor(prefill);
    parallel.set_input_tensor(prefill);
    reference.infer();
    parallel.infer();
    expect_close(reference.get_output_tensor(0), parallel.get_output_tensor(0));

    trim_kv_cache(reference, 4);
    trim_kv_cache(parallel, 4);
    for (auto&& state : parallel.query_state())
        EXPECT_EQ(state.get_state().get_shape()[1], 1u) << state.get_name();

    auto next = make_input(config, 1, 2);
    reference.set_input_tensor(next);
    parallel.set_input_tensor(next);
    reference.infer();
    parallel.infer();
    expect_close(reference.get_output_tensor(0), parallel.get_output_tensor(0));
}

// Fused qkv_proj / gate_up_proj (Phi-3): one weight per fused projection, cut per part, and the
// split after it taking the rank's parts. A contiguous cut gives rank 0 only gate features.
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnFusedProjections) {
    BlockConfig config;
    config.qkv = Fused::SLICES;
    config.gate_up = Fused::SPLIT;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

TEST_P(TPAccuracyTest, MatchesSingleDeviceOnFusedProjectionsWithVariadicSplitsAndBias) {
    BlockConfig config;
    config.qkv = Fused::VARIADIC_SPLIT;
    config.gate_up = Fused::VARIADIC_SPLIT;
    config.with_bias = true;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

TEST_P(TPAccuracyTest, MatchesSingleDeviceOnAFusedQkvWithOneKvHeadAfterKvCacheTrim) {
    BlockConfig config;
    config.qkv = Fused::SLICES;
    config.gate_up = Fused::SPLIT;
    config.num_heads = 24;
    config.num_kv_heads = 1;
    config.stateful = true;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = compile_both(model);

    auto prefill = make_input(config, sequence_length, 1);
    reference.set_input_tensor(prefill);
    parallel.set_input_tensor(prefill);
    reference.infer();
    parallel.infer();
    expect_close(reference.get_output_tensor(0), parallel.get_output_tensor(0));

    trim_kv_cache(reference, 4);
    trim_kv_cache(parallel, 4);
    auto next = make_input(config, 1, 2);
    reference.set_input_tensor(next);
    parallel.set_input_tensor(next);
    reference.infer();
    parallel.infer();
    expect_close(reference.get_output_tensor(0), parallel.get_output_tensor(0));
}

// The vocabulary projection is split by output feature and the bands are
// copied back into rank 0 -- the only rank whose outputs anyone reads. A band
// written at the wrong offset, or a rank whose slice never arrives, shows up
// here as logits that stop matching partway along the vocabulary.
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnAVocabularyProjection) {
    BlockConfig config;
    auto model = make_block_with_lm_head(config, 768);  // divides by every world size under test

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// A vocabulary no world size divides has to stay whole on every rank, with no
// gather inserted; the logits must come out right all the same.
TEST_P(TPAccuracyTest, MatchesSingleDeviceOnAnIndivisibleVocabulary) {
    BlockConfig config;
    auto model = make_block_with_lm_head(config, 769);  // prime

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

INSTANTIATE_TEST_SUITE_P(TP,
                         TPAccuracyTest,
                         ::testing::Combine(::testing::ValuesIn(world_sizes),
                                            ::testing::ValuesIn(precisions)),
                         [](const ::testing::TestParamInfo<Param>& info) {
                             return "world" + std::to_string(std::get<0>(info.param)) + "_" +
                                    std::get<1>(info.param).get_type_name();
                         });

}  // namespace
}  // namespace ov::tp::tests
