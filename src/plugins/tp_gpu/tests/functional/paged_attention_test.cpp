// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// PagedAttention on TP_GPU: the cache the plugin owns, and the numbers it
// produces.
//
// Under tensor parallelism the KV cache is split by head across the rank
// devices, so the pipeline cannot allocate it through one remote context. The
// plugin hands out a controller instead, and these cases pin down both halves
// of that contract: what the controller reports and does, and that a model
// running on the resulting cache computes what a single GPU computes.
//
// Needs two or more GPUs; the suite skips otherwise.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tp_gpu/paged_attention_cache_controller.hpp"
#include "tp_gpu/properties.hpp"
#include "tp_test_models.hpp"

namespace ov::tp_gpu::tests {
namespace {

constexpr uint32_t world_size = 2;

/// Enough tokens to fill more than one block, so the block table is not
/// degenerate and a wrong stride shows up.
constexpr size_t prompt_tokens = 40;

class TPGpuPagedAttentionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        try {
            gpu_count = ov::Core{}.get_property("GPU", ov::available_devices).size();
        } catch (const std::exception& e) {
            gpu_error = e.what();
        }
    }

    void SetUp() override {
        if (gpu_count < world_size)
            GTEST_SKIP() << "need >= " << world_size << " GPUs, found " << gpu_count << gpu_error;
    }

    /// The cache precision is pinned to f16 rather than left to the plugin,
    /// which would compress it to int8. Quantization is a step function, so the
    /// last-bit difference between a sharded and an unsharded reduction lands
    /// on a different level and the two runs diverge for a reason that has
    /// nothing to do with sharding.
    static ov::AnyMap gpu_config() {
        return {ov::hint::kv_cache_precision(ov::element::f16)};
    }

    static ov::AnyMap tp_config() {
        ov::AnyMap config = gpu_config();
        config.emplace(ov::tp_gpu::tp_size(world_size));
        return config;
    }

    static ov::tp_gpu::PagedAttentionCacheControllerPtr controller_of(const ov::CompiledModel& compiled) {
        return compiled.get_property(ov::tp_gpu::paged_attention_cache_controller);
    }

    static bool offers_controller(const ov::CompiledModel& compiled) {
        const auto supported = compiled.get_property(ov::supported_properties);
        return std::find(supported.begin(),
                         supported.end(),
                         ov::tp_gpu::paged_attention_cache_controller.name()) != supported.end();
    }

    /// Bytes one block takes across every cache port of a compiled model --
    /// the same sum a paged-attention scheduler budgets against.
    static size_t block_bytes_of(const ov::CompiledModel& compiled) {
        size_t bytes = 0;
        for (const auto& port : compiled.inputs()) {
            bool is_cache = false;
            for (const auto& name : port.get_names())
                is_cache |= name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0;
            if (!is_cache)
                continue;

            const auto& shape = port.get_partial_shape();
            bytes += static_cast<size_t>(shape[1].get_length() * shape[2].get_length() * shape[3].get_length()) *
                     port.get_element_type().size();
        }
        return bytes;
    }

    static ov::Tensor make_input(const BlockConfig& config, size_t tokens, int32_t seed = 7) {
        // A narrow range keeps the u4 weights from saturating the activations,
        // so a real mismatch is not masked by both sides producing garbage.
        ov::test::utils::InputGenerateData data(-1, 2, 32, seed);
        return ov::test::utils::create_and_fill_tensor(ov::element::f32,
                                                       ov::Shape{tokens, 1, config.hidden},
                                                       data);
    }

    /// Fills the scheduler metadata of a single sequence occupying the first
    /// blocks of an otherwise unused cache.
    static void set_prefill_metadata(ov::InferRequest& request, size_t tokens, size_t block_size) {
        const size_t blocks = (tokens + block_size - 1) / block_size;

        auto scalar = [&](const char* name, int32_t value) {
            ov::Tensor tensor(ov::element::i32, ov::Shape{});
            tensor.data<int32_t>()[0] = value;
            request.set_tensor(name, tensor);
        };
        auto vector = [&](const char* name, const std::vector<int32_t>& values) {
            ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
            std::copy(values.begin(), values.end(), tensor.data<int32_t>());
            request.set_tensor(name, tensor);
        };

        std::vector<int32_t> indices(blocks);
        std::iota(indices.begin(), indices.end(), 0);

        // One sequence, nothing cached yet: it starts at token 0 and spans the
        // whole run of tokens.
        vector("past_lens", {0});
        vector("subsequence_begins", {0, static_cast<int32_t>(tokens)});
        vector("block_indices", indices);
        vector("block_indices_begins", {0, static_cast<int32_t>(blocks)});
        scalar("max_context_len", static_cast<int32_t>(tokens));
    }

    /// Allocates the cache of a plain GPU model the way a paged-attention
    /// pipeline does, so the reference run has something to write into.
    static std::vector<ov::Tensor> allocate_reference_cache(ov::CompiledModel& compiled,
                                                            ov::InferRequest& request,
                                                            size_t blocks) {
        auto context = compiled.get_context();
        std::vector<ov::Tensor> cache;
        for (const auto& port : compiled.inputs()) {
            std::string cache_name;
            for (const auto& name : port.get_names()) {
                if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0)
                    cache_name = name;
            }
            if (cache_name.empty())
                continue;

            auto shape = port.get_partial_shape().get_max_shape();
            shape[0] = blocks;
            cache.push_back(context.create_tensor(port.get_element_type(), shape));
            request.set_tensor(cache_name, cache.back());
        }
        return cache;
    }

    static ov::Tensor to_host(const ov::Tensor& tensor) {
        ov::Tensor host(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(host);
        return host;
    }

    ov::Core core;

    static size_t gpu_count;
    static std::string gpu_error;
};

size_t TPGpuPagedAttentionTest::gpu_count = 0;
std::string TPGpuPagedAttentionTest::gpu_error;

// ---------------------------------------------------------------------------
// The controller contract
// ---------------------------------------------------------------------------

TEST_F(TPGpuPagedAttentionTest, OffersAControllerOnlyForPagedAttentionModels) {
    auto paged = core.compile_model(make_paged_attention_block(BlockConfig{}), "TP_GPU", tp_config());
    EXPECT_TRUE(offers_controller(paged));
    EXPECT_NE(controller_of(paged), nullptr);

    // A stateful model keeps its cache in variables, which the pipeline never
    // allocates; advertising a controller there would send it down the wrong
    // path.
    BlockConfig stateful;
    stateful.stateful = true;
    auto sdpa = core.compile_model(make_transformer_block(stateful), "TP_GPU", tp_config());
    EXPECT_FALSE(offers_controller(sdpa));
}

TEST_F(TPGpuPagedAttentionTest, ReportsTheCacheGeometry) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), "TP_GPU", tp_config());
    auto controller = controller_of(compiled);

    EXPECT_EQ(controller->get_num_layers(), 1u);
    // Key and value of every layer, on every rank.
    EXPECT_EQ(controller->get_num_cache_tensors(), 2u * world_size);
    EXPECT_GT(controller->get_block_size(), 0u);
    EXPECT_EQ(controller->get_num_allocated_blocks(), 0u);
}

TEST_F(TPGpuPagedAttentionTest, ReportsTheWholeBlockNotOneRanksShare) {
    // The scheduler budgets memory against the size of a block. Reporting one
    // rank's slice would let it believe the cache is `world_size` times
    // cheaper than it is.
    auto reference = core.compile_model(make_paged_attention_block(BlockConfig{}), "GPU", gpu_config());
    auto parallel = core.compile_model(make_paged_attention_block(BlockConfig{}), "TP_GPU", tp_config());

    EXPECT_EQ(controller_of(parallel)->get_block_size_in_bytes(), block_bytes_of(reference));
}

TEST_F(TPGpuPagedAttentionTest, AllocatesGrowsAndClears) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), "TP_GPU", tp_config());
    auto controller = controller_of(compiled);

    controller->allocate_cache_if_needed(4);
    EXPECT_EQ(controller->get_num_allocated_blocks(), 4u);

    // Asking for less than what is there is a no-op: the scheduler calls this
    // on every step and must not thrash the allocation.
    controller->allocate_cache_if_needed(2);
    EXPECT_EQ(controller->get_num_allocated_blocks(), 4u);

    controller->allocate_cache_if_needed(9);
    EXPECT_EQ(controller->get_num_allocated_blocks(), 9u);

    controller->clear();
    EXPECT_EQ(controller->get_num_allocated_blocks(), 0u);

    // Usable again after a clear -- a pipeline reuses the model across chats.
    controller->allocate_cache_if_needed(3);
    EXPECT_EQ(controller->get_num_allocated_blocks(), 3u);
}

TEST_F(TPGpuPagedAttentionTest, CopiesBlocksOnEveryRank) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), "TP_GPU", tp_config());
    auto controller = controller_of(compiled);

    controller->allocate_cache_if_needed(8);
    // Copy-on-write duplicates one block into several; the ranks hold
    // different heads of the same block indices, so each has to do it.
    EXPECT_NO_THROW(controller->copy_blocks({{0, {1, 2}}, {3, {4}}}));

    EXPECT_THROW(controller->copy_blocks({{8, {0}}}), ov::Exception);
    EXPECT_THROW(controller->copy_blocks({{0, {8}}}), ov::Exception);
}

TEST_F(TPGpuPagedAttentionTest, RefusesToInferWithoutACache) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), "TP_GPU", tp_config());
    auto request = compiled.create_infer_request();

    BlockConfig config;
    request.set_tensor("input", make_input(config, prompt_tokens));
    set_prefill_metadata(request, prompt_tokens, controller_of(compiled)->get_block_size());

    // Nothing allocated the cache, so there is nothing to attend over. Saying
    // so beats reading whatever the device memory held.
    EXPECT_THROW(request.infer(), ov::Exception);
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

TEST_F(TPGpuPagedAttentionTest, MatchesSingleGpuOnAPrefill) {
    BlockConfig config;
    auto model = make_paged_attention_block(config);

    auto reference_model = core.compile_model(model, "GPU", gpu_config());
    auto reference = reference_model.create_infer_request();

    auto parallel_model = core.compile_model(model, "TP_GPU", tp_config());
    auto parallel = parallel_model.create_infer_request();

    auto controller = controller_of(parallel_model);
    const size_t block_size = controller->get_block_size();
    const size_t blocks = (prompt_tokens + block_size - 1) / block_size;

    auto reference_cache = allocate_reference_cache(reference_model, reference, blocks);
    controller->allocate_cache_if_needed(blocks);

    auto input = make_input(config, prompt_tokens);
    reference.set_tensor("input", input);
    parallel.set_tensor("input", input);
    set_prefill_metadata(reference, prompt_tokens, block_size);
    set_prefill_metadata(parallel, prompt_tokens, block_size);

    reference.infer();
    parallel.infer();

    // Splitting a reduction across devices reorders the additions, so the two
    // results differ in the last bits; f16 activations set the floor.
    ov::test::utils::compare(to_host(reference.get_output_tensor(0)),
                             to_host(parallel.get_output_tensor(0)),
                             1e-2,
                             1e-2);
}

TEST_F(TPGpuPagedAttentionTest, MatchesSingleGpuAcrossTwoSteps) {
    BlockConfig config;
    auto model = make_paged_attention_block(config);

    auto reference_model = core.compile_model(model, "GPU", gpu_config());
    auto reference = reference_model.create_infer_request();

    auto parallel_model = core.compile_model(model, "TP_GPU", tp_config());
    auto parallel = parallel_model.create_infer_request();

    auto controller = controller_of(parallel_model);
    const size_t block_size = controller->get_block_size();
    const size_t blocks = (prompt_tokens + block_size) / block_size + 1;

    auto reference_cache = allocate_reference_cache(reference_model, reference, blocks);
    controller->allocate_cache_if_needed(blocks);

    // Prefill, then one generated token attending over what the prefill wrote.
    // A cache sharded inconsistently with the query heads is still fine on the
    // first step and wrong on the second.
    auto prefill = make_input(config, prompt_tokens, 1);
    reference.set_tensor("input", prefill);
    parallel.set_tensor("input", prefill);
    set_prefill_metadata(reference, prompt_tokens, block_size);
    set_prefill_metadata(parallel, prompt_tokens, block_size);
    reference.infer();
    parallel.infer();

    auto step = make_input(config, 1, 2);
    const size_t used_blocks = (prompt_tokens + block_size) / block_size;
    for (ov::InferRequest* request : {&reference, &parallel}) {
        request->set_tensor("input", step);

        auto past_lens = ov::Tensor(ov::element::i32, ov::Shape{1});
        past_lens.data<int32_t>()[0] = static_cast<int32_t>(prompt_tokens);
        request->set_tensor("past_lens", past_lens);

        auto begins = ov::Tensor(ov::element::i32, ov::Shape{2});
        begins.data<int32_t>()[0] = 0;
        begins.data<int32_t>()[1] = 1;
        request->set_tensor("subsequence_begins", begins);

        auto indices = ov::Tensor(ov::element::i32, ov::Shape{used_blocks});
        std::iota(indices.data<int32_t>(), indices.data<int32_t>() + used_blocks, 0);
        request->set_tensor("block_indices", indices);

        auto index_begins = ov::Tensor(ov::element::i32, ov::Shape{2});
        index_begins.data<int32_t>()[0] = 0;
        index_begins.data<int32_t>()[1] = static_cast<int32_t>(used_blocks);
        request->set_tensor("block_indices_begins", index_begins);

        auto context_len = ov::Tensor(ov::element::i32, ov::Shape{});
        context_len.data<int32_t>()[0] = static_cast<int32_t>(prompt_tokens + 1);
        request->set_tensor("max_context_len", context_len);
    }

    reference.infer();
    parallel.infer();

    ov::test::utils::compare(to_host(reference.get_output_tensor(0)),
                             to_host(parallel.get_output_tensor(0)),
                             1e-2,
                             1e-2);
}

}  // namespace
}  // namespace ov::tp_gpu::tests
