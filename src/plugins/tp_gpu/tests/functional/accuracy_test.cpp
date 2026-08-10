// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Accuracy of the whole TP_GPU pipeline.
//
// Each case compiles one synthetic transformer block twice -- once on a single
// GPU, once on TP_GPU with the model split across several of them -- feeds both
// the same input and compares the outputs.  Tensor parallelism is a pure
// refactoring of the computation, so the two must agree up to the reordering of
// the AllReduce summation.
//
// Needs two or more GPUs; the suite skips otherwise.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "tp_gpu/properties.hpp"
#include "tp_test_models.hpp"

namespace ov::tp_gpu::tests {
namespace {

/// Sequence length used for the comparison: long enough for the attention to
/// mix all heads, short enough to keep the test quick.
constexpr size_t sequence_length = 16;

/// World size and the arithmetic both sides are pinned to.  Pinning matters:
/// left to itself the GPU picks f16, and a single ULP there is already ~1e-3 --
/// the same magnitude as a genuine sharding error, which would go unnoticed.
using Param = std::tuple<uint32_t, ov::element::Type>;

const std::vector<uint32_t> world_sizes = {2, 3, 4};
const std::vector<ov::element::Type> precisions = {ov::element::f32, ov::element::f16};

class TPGpuAccuracyTest : public ::testing::TestWithParam<Param> {
protected:
    static uint32_t world_size() {
        return std::get<0>(GetParam());
    }

    static ov::element::Type inference_precision() {
        return std::get<1>(GetParam());
    }

    static void SetUpTestSuite() {
        try {
            gpu_count = ov::Core{}.get_property("GPU", ov::available_devices).size();
        } catch (const std::exception& e) {
            // No GPU plugin, or it failed to initialize.  Keep the reason: a
            // broken environment otherwise looks like a machine without GPUs.
            gpu_error = e.what();
        }
    }

    void SetUp() override {
        if (gpu_count < world_size())
            GTEST_SKIP() << "need >= " << world_size() << " GPUs, found " << gpu_count << gpu_error;
    }

    /// Splitting a reduction across devices reorders the additions, so the two
    /// results differ in the last bits.  One ULP of the inference precision is
    /// the floor; these leave room for a few accumulated roundings on top.
    static double tolerance() {
        return inference_precision() == ov::element::f16 ? 1e-2 : 1e-5;
    }

    /// Compiles `model` on a single GPU and on TP_GPU over the parameter's
    /// world size, runs both on `input` and returns the two outputs.
    ///
    /// The outputs are copied to host memory: an output tensor is owned by the
    /// infer request that produced it and does not outlive the pipelines torn
    /// down when this returns.
    std::pair<ov::Tensor, ov::Tensor> run_both(const std::shared_ptr<ov::Model>& model, const ov::Tensor& input) {
        auto reference = core.compile_model(model, "GPU", precision_config()).create_infer_request();
        auto parallel = core.compile_model(model, "TP_GPU", tp_config()).create_infer_request();

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
    /// same read-slice-write GenAI performs between chat turns.  KV cache
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
        ov::AnyMap config = precision_config();
        config.emplace(ov::tp_gpu::tp_size(world_size()));
        return config;
    }

    ov::Core core;

    static size_t gpu_count;
    static std::string gpu_error;
};

size_t TPGpuAccuracyTest::gpu_count = 0;
std::string TPGpuAccuracyTest::gpu_error;

TEST_P(TPGpuAccuracyTest, MatchesSingleGpuOnGatedMlpBlock) {
    BlockConfig config;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

TEST_P(TPGpuAccuracyTest, MatchesSingleGpuOnSingleBranchMlpBlock) {
    BlockConfig config;
    config.gated_mlp = false;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

TEST_P(TPGpuAccuracyTest, MatchesSingleGpuOnBiasedProjections) {
    BlockConfig config;
    config.with_bias = true;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// A row-parallel projection sums partial products across ranks, so its bias
// must be added exactly once.  Adding it on every rank would scale the bias by
// the world size, which only shows up once the collective has run.
TEST_P(TPGpuAccuracyTest, MatchesSingleGpuOnUntransposedWeights) {
    BlockConfig config;
    config.transpose_b = false;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// Two generation steps over a stateful KV cache: the second one reads back what
// the first wrote, so a cache that was sharded inconsistently with the query
// heads shows up as a mismatch only on the second call.
TEST_P(TPGpuAccuracyTest, MatchesSingleGpuAcrossStatefulSteps) {
    BlockConfig config;
    config.stateful = true;
    auto model = make_transformer_block(config);

    auto reference = core.compile_model(model, "GPU", precision_config()).create_infer_request();
    auto parallel = core.compile_model(model, "TP_GPU", tp_config()).create_infer_request();

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
// state, slicing off the trailing tokens and writing it back.  On TP every
// rank holds a slice of the kv heads, so the read has to gather and the write
// has to scatter; getting that wrong overwrites the other ranks' heads with
// rank 0's and the next step silently produces a different answer.
TEST_P(TPGpuAccuracyTest, MatchesSingleGpuAfterKvCacheTrim) {
    constexpr size_t trimmed_tokens = 4;

    BlockConfig config;
    config.stateful = true;
    auto model = make_transformer_block(config);

    auto reference = core.compile_model(model, "GPU", precision_config()).create_infer_request();
    auto parallel = core.compile_model(model, "TP_GPU", tp_config()).create_infer_request();

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

// Every other case leaves `intermediate` at 448 -- 14 quantization groups, so
// three and four ranks cannot divide it evenly and the split falls back on
// whole groups.  512 is 16 groups, which two and four ranks do divide evenly;
// running both tells an error in that fallback apart from one in the split
// itself.
TEST_P(TPGpuAccuracyTest, MatchesSingleGpuOnEvenlyDivisibleMlp) {
    BlockConfig config;
    config.intermediate = 512;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

// The default 4 KV heads divide evenly over two and four ranks.  Five do not,
// for any world size the tests use, so the ranks end up with different numbers
// of attention heads -- and with the remainder handed to the lower ranks, the
// last rank is the one that gets shorted.
TEST_P(TPGpuAccuracyTest, MatchesSingleGpuOnUnevenHeadSplit) {
    BlockConfig config;
    config.num_kv_heads = 5;
    config.num_heads = 10;
    auto model = make_transformer_block(config);

    auto [reference, parallel] = run_both(model, make_input(config));
    expect_close(reference, parallel);
}

INSTANTIATE_TEST_SUITE_P(TPGpu,
                         TPGpuAccuracyTest,
                         ::testing::Combine(::testing::ValuesIn(world_sizes),
                                            ::testing::ValuesIn(precisions)),
                         [](const ::testing::TestParamInfo<Param>& info) {
                             return "world" + std::to_string(std::get<0>(info.param)) + "_" +
                                    std::get<1>(info.param).get_type_name();
                         });

}  // namespace
}  // namespace ov::tp_gpu::tests
