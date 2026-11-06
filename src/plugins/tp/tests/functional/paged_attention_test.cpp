// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// PagedAttention on TP: the cache the plugin owns, and the numbers it
// produces.
//
// Under tensor parallelism the KV cache is split by head across the rank
// devices, so the pipeline cannot allocate it through one remote context. The
// plugin manages it instead (ov::plugin_managed_cache), and these cases pin
// down both halves of that contract: what the facade reports and does, and
// that a model running on the resulting cache computes what a single rank
// device computes.
//
// Needs two or more rank devices (rank_devices.hpp); the suite skips otherwise.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/plugin_managed_cache.hpp"
#include "rank_devices.hpp"
#include "tp_test_models.hpp"

namespace ov::tp::tests {
namespace {

constexpr uint32_t world_size = 2;

/// Enough tokens to fill more than one block, so the block table is not
/// degenerate and a wrong stride shows up.
constexpr size_t prompt_tokens = 40;

/// The only group a paged-attention model without linear attention has.
constexpr size_t kv_group = 0;

std::vector<int32_t> block_range(size_t first, size_t count) {
    std::vector<int32_t> indices(count);
    std::iota(indices.begin(), indices.end(), static_cast<int32_t>(first));
    return indices;
}

class TPPagedAttentionTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (const auto missing = missing_rank_devices(world_size); !missing.empty())
            GTEST_SKIP() << missing;
    }

    /// The cache precision is pinned to f16 rather than left to the plugin,
    /// which would compress it to int8. Quantization is a step function, so the
    /// last-bit difference between a sharded and an unsharded reduction lands
    /// on a different level and the two runs diverge for a reason that has
    /// nothing to do with sharding.
    static ov::AnyMap reference_config() {
        return {ov::hint::kv_cache_precision(ov::element::f16)};
    }

    static ov::AnyMap tp_config() {
        return reference_config();
    }

    ov::CompiledModel compile_reference(const std::shared_ptr<ov::Model>& model) {
        return core.compile_model(model, rank_device(0), reference_config());
    }

    static ov::PluginManagedCache cache_of(const ov::CompiledModel& compiled) {
        return compiled.get_property(ov::plugin_managed_cache);
    }

    static bool offers_cache(const ov::CompiledModel& compiled) {
        const auto supported = compiled.get_property(ov::supported_properties);
        return std::find(supported.begin(), supported.end(), ov::plugin_managed_cache.name()) != supported.end();
    }

    static size_t block_size_of(const ov::PluginManagedCache& cache) {
        return cache.get_groups().at(kv_group).block_size;
    }

    static bool is_cache_port(const ov::Output<const ov::Node>& port) {
        for (const auto& name : port.get_names()) {
            if (name.rfind("key_cache.", 0) == 0 || name.rfind("value_cache.", 0) == 0)
                return true;
        }
        return false;
    }

    /// Bytes one block takes across every cache port of a compiled model --
    /// the same sum a paged-attention scheduler budgets against.
    static size_t block_bytes_of(const ov::CompiledModel& compiled) {
        size_t bytes = 0;
        for (const auto& port : compiled.inputs()) {
            if (!is_cache_port(port))
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

    /// Fills the scheduler metadata of a single sequence with `past` tokens
    /// cached and `tokens` new ones, its blocks listed in `blocks`.
    static void set_metadata(ov::InferRequest& request,
                             size_t past,
                             size_t tokens,
                             const std::vector<int32_t>& blocks) {
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

        vector("past_lens", {static_cast<int32_t>(past)});
        vector("subsequence_begins", {0, static_cast<int32_t>(tokens)});
        vector("block_indices", blocks);
        vector("block_indices_begins", {0, static_cast<int32_t>(blocks.size())});
        scalar("max_context_len", static_cast<int32_t>(past + tokens));
    }

    /// One sequence occupying the first blocks of an otherwise unused cache.
    static void set_prefill_metadata(ov::InferRequest& request, size_t tokens, size_t block_size) {
        set_metadata(request, 0, tokens, block_range(0, (tokens + block_size - 1) / block_size));
    }

    /// Allocates the cache of a single-device model the way a paged-attention
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

    static float max_abs_diff(const ov::Tensor& a, const ov::Tensor& b) {
        const auto host_a = to_host(a);
        const auto host_b = to_host(b);
        OPENVINO_ASSERT(host_a.get_element_type() == ov::element::f32 && host_a.get_shape() == host_b.get_shape());
        float diff = 0.0f;
        for (size_t i = 0; i < host_a.get_size(); ++i)
            diff = std::max(diff, std::abs(host_a.data<float>()[i] - host_b.data<float>()[i]));
        return diff;
    }

    /// The decode step following a prefill of `prompt_tokens`, writing to and
    /// attending over `blocks`.
    static ov::Tensor decode(ov::InferRequest& request, const BlockConfig& config, const std::vector<int32_t>& blocks) {
        request.set_tensor("input", make_input(config, 1, 2));
        set_metadata(request, prompt_tokens, 1, blocks);
        request.infer();
        return to_host(request.get_output_tensor(0));
    }

    static void prefill(ov::InferRequest& request, const BlockConfig& config, size_t block_size) {
        request.set_tensor("input", make_input(config, prompt_tokens, 1));
        set_prefill_metadata(request, prompt_tokens, block_size);
        request.infer();
    }

    ov::Core core;
};

// ---------------------------------------------------------------------------
// The controller contract
// ---------------------------------------------------------------------------

TEST_F(TPPagedAttentionTest, OffersTheCacheOnlyForPagedAttentionModels) {
    auto paged = core.compile_model(make_paged_attention_block(BlockConfig{}), tp_device(world_size), tp_config());
    EXPECT_TRUE(offers_cache(paged));
    EXPECT_TRUE(static_cast<bool>(cache_of(paged)));

    // A stateful model keeps its cache in variables, which the pipeline never
    // allocates; advertising a managed cache there would send it down the wrong
    // path.
    BlockConfig stateful;
    stateful.stateful = true;
    auto sdpa = core.compile_model(make_transformer_block(stateful), tp_device(world_size), tp_config());
    EXPECT_FALSE(offers_cache(sdpa));
}

TEST_F(TPPagedAttentionTest, ReportsTheModelLevelGroup) {
    // Everything the facade reports is about the model as if it were not
    // split: the shapes and the block size are those a single device reports, so
    // the scheduler budgets the whole block and not one rank's share. One KV
    // head is kept by both ranks and still counted once.
    BlockConfig one_kv_head;
    one_kv_head.num_kv_heads = 1;
    for (const auto& config : {BlockConfig{}, one_kv_head}) {
        SCOPED_TRACE(std::to_string(config.num_kv_heads) + " KV heads");
        auto reference = compile_reference(make_paged_attention_block(config));
        auto parallel = core.compile_model(make_paged_attention_block(config), tp_device(world_size), tp_config());

        const auto groups = cache_of(parallel).get_groups();
        ASSERT_EQ(groups.size(), 1u);
        const auto& group = groups[kv_group];
        EXPECT_EQ(group.kind, ov::CacheKind::KEY_VALUE);
        EXPECT_EQ(group.block_size, 16u);
        EXPECT_EQ(group.block_size_in_bytes, block_bytes_of(reference));

        std::vector<std::string> names;
        for (const auto& port : reference.inputs()) {
            if (!is_cache_port(port))
                continue;
            const auto it = std::find_if(group.tensors.begin(), group.tensors.end(), [&](const ov::CacheTensorInfo& t) {
                return port.get_names().count(t.name) != 0;
            });
            ASSERT_NE(it, group.tensors.end()) << port;
            EXPECT_EQ(it->layer, 0u);
            EXPECT_EQ(it->element_type, port.get_element_type());
            EXPECT_EQ(it->shape, port.get_partial_shape());
            names.push_back(it->name);
        }
        EXPECT_EQ(names.size(), group.tensors.size());
        EXPECT_EQ(cache_of(parallel).get_num_allocated_blocks(kv_group), 0u);
    }
}

TEST_F(TPPagedAttentionTest, AllocatesReallocatesAndDeallocates) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), tp_device(world_size), tp_config());
    auto cache = cache_of(compiled);

    cache.allocate(kv_group, 4, false);
    EXPECT_EQ(cache.get_num_allocated_blocks(kv_group), 4u);
    EXPECT_THROW(cache.allocate(kv_group, 8, false), ov::Exception);

    cache.reallocate(kv_group, 9, true);
    EXPECT_EQ(cache.get_num_allocated_blocks(kv_group), 9u);
    cache.reallocate(kv_group, 2, false);
    EXPECT_EQ(cache.get_num_allocated_blocks(kv_group), 2u);

    cache.deallocate(kv_group);
    EXPECT_EQ(cache.get_num_allocated_blocks(kv_group), 0u);
    EXPECT_NO_THROW(cache.deallocate(kv_group));
    EXPECT_THROW(cache.reallocate(kv_group, 4, false), ov::Exception);

    // Usable again after a deallocation -- a pipeline reuses the model across chats.
    cache.allocate(kv_group, 3, true);
    EXPECT_EQ(cache.get_num_allocated_blocks(kv_group), 3u);

    EXPECT_THROW(cache.allocate(kv_group + 1, 3, false), ov::Exception);
    EXPECT_THROW(cache.reallocate(kv_group, 0, false), ov::Exception);
}

TEST_F(TPPagedAttentionTest, AvailableMemoryFollowsTheAllocation) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), tp_device(world_size), tp_config());
    // A request holds the cache too: freeing must reach it as well.
    auto request = compiled.create_infer_request();
    auto cache = cache_of(compiled);
    const size_t block_bytes = cache.get_groups()[kv_group].block_size_in_bytes;

    const size_t before = cache.get_available_memory();
    EXPECT_GT(before, 0u);
    EXPECT_EQ(before % block_bytes, 0u);

    constexpr size_t blocks = 4096;
    cache.allocate(kv_group, blocks, false);
    const size_t allocated = cache.get_available_memory();
    EXPECT_LE(allocated + blocks * block_bytes, before);

    cache.deallocate(kv_group);
    EXPECT_EQ(cache.get_available_memory(), before);
}

TEST_F(TPPagedAttentionTest, FailedReallocationKeepsTheCache) {
    BlockConfig config;
    auto model = make_paged_attention_block(config);
    auto reference_model = compile_reference(model);
    auto reference = reference_model.create_infer_request();
    auto parallel_model = core.compile_model(model, tp_device(world_size), tp_config());
    auto parallel = parallel_model.create_infer_request();

    auto cache = cache_of(parallel_model);
    const size_t block_size = block_size_of(cache);
    const size_t blocks = (prompt_tokens + block_size) / block_size;
    auto reference_cache = allocate_reference_cache(reference_model, reference, blocks);
    cache.allocate(kv_group, blocks, true);

    prefill(reference, config, block_size);
    prefill(parallel, config, block_size);

    // Far more than any device holds: the allocation is refused before anything is bound.
    const size_t too_many = cache.get_available_memory() / cache.get_groups()[kv_group].block_size_in_bytes * 16 + 1;
    EXPECT_THROW(cache.reallocate(kv_group, too_many, false), ov::Exception);
    EXPECT_EQ(cache.get_num_allocated_blocks(kv_group), blocks);

    // The prefill's keys and values are still there.
    ov::test::utils::compare(decode(reference, config, block_range(0, blocks)),
                             decode(parallel, config, block_range(0, blocks)),
                             1e-2,
                             1e-2);
}

TEST_F(TPPagedAttentionTest, RejectsTensorsOnCachePorts) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), tp_device(world_size), tp_config());
    auto request = compiled.create_infer_request();
    const auto tensor = cache_of(compiled).get_groups()[kv_group].tensors.front();

    ov::Tensor host(tensor.element_type, tensor.shape.get_min_shape());
    EXPECT_THROW(request.set_tensor(tensor.name, host), ov::Exception);
    EXPECT_THROW(request.get_tensor(tensor.name), ov::Exception);
}

TEST_F(TPPagedAttentionTest, RefusesToInferWithoutACache) {
    auto compiled = core.compile_model(make_paged_attention_block(BlockConfig{}), tp_device(world_size), tp_config());
    auto request = compiled.create_infer_request();
    auto cache = cache_of(compiled);

    BlockConfig config;
    request.set_tensor("input", make_input(config, prompt_tokens));
    set_prefill_metadata(request, prompt_tokens, block_size_of(cache));

    // Nothing allocated the cache, so there is nothing to attend over. Saying
    // so beats reading whatever the device memory held.
    EXPECT_THROW(request.infer(), ov::Exception);

    // Nor after it was freed again.
    cache.allocate(kv_group, 4, false);
    cache.deallocate(kv_group);
    EXPECT_THROW(request.infer(), ov::Exception);
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

TEST_F(TPPagedAttentionTest, MatchesSingleDeviceOnAPrefill) {
    BlockConfig config;
    auto model = make_paged_attention_block(config);

    auto reference_model = compile_reference(model);
    auto reference = reference_model.create_infer_request();

    auto parallel_model = core.compile_model(model, tp_device(world_size), tp_config());
    auto cache = cache_of(parallel_model);
    const size_t block_size = block_size_of(cache);
    const size_t blocks = (prompt_tokens + block_size - 1) / block_size;

    auto reference_cache = allocate_reference_cache(reference_model, reference, blocks);
    cache.allocate(kv_group, blocks, false);
    // Created after the allocation: the cache is bound to it on creation.
    auto parallel = parallel_model.create_infer_request();

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

TEST_F(TPPagedAttentionTest, MatchesSingleDeviceAcrossTwoSteps) {
    // One KV head: both ranks keep it and half of its query heads each. Fused projections: one
    // weight per fused projection, cut per part.
    BlockConfig one_kv_head;
    one_kv_head.num_kv_heads = 1;
    BlockConfig fused;
    fused.qkv = Fused::VARIADIC_SPLIT;
    fused.gate_up = Fused::SPLIT;
    for (const auto& config : {BlockConfig{}, one_kv_head, fused}) {
        SCOPED_TRACE(std::to_string(config.num_kv_heads) + " KV heads, fused " + std::to_string(config.qkv != Fused::NONE));
        auto model = make_paged_attention_block(config);

        auto reference_model = compile_reference(model);
        auto reference = reference_model.create_infer_request();

        auto parallel_model = core.compile_model(model, tp_device(world_size), tp_config());
        // Created before the allocation: the cache is bound to it when allocated.
        auto parallel = parallel_model.create_infer_request();

        auto cache = cache_of(parallel_model);
        const size_t block_size = block_size_of(cache);
        const size_t blocks = (prompt_tokens + block_size) / block_size;

        auto reference_cache = allocate_reference_cache(reference_model, reference, blocks);
        cache.allocate(kv_group, blocks, false);

        // Prefill, then one generated token attending over what the prefill wrote.
        // A cache sharded inconsistently with the query heads is still fine on the
        // first step and wrong on the second.
        prefill(reference, config, block_size);
        prefill(parallel, config, block_size);

        // Growing the cache in between keeps what the prefill wrote and rebinds the request.
        cache.reallocate(kv_group, blocks + 5, true);

        ov::test::utils::compare(decode(reference, config, block_range(0, blocks)),
                                 decode(parallel, config, block_range(0, blocks)),
                                 1e-2,
                                 1e-2);
    }
}

TEST_F(TPPagedAttentionTest, CopiedBlocksHoldTheSameKeysAndValues) {
    // Copy-on-write duplicates blocks; the ranks hold different heads of the
    // same block indices, so each has to copy.
    BlockConfig config;
    auto model = make_paged_attention_block(config);
    auto reference_model = compile_reference(model);
    auto reference = reference_model.create_infer_request();
    auto parallel_model = core.compile_model(model, tp_device(world_size), tp_config());
    auto parallel = parallel_model.create_infer_request();

    auto cache = cache_of(parallel_model);
    const size_t block_size = block_size_of(cache);
    const size_t blocks = (prompt_tokens + block_size) / block_size;
    auto reference_cache = allocate_reference_cache(reference_model, reference, blocks);
    cache.allocate(kv_group, 2 * blocks + 1, false);

    prefill(reference, config, block_size);
    prefill(parallel, config, block_size);

    std::map<size_t, std::list<size_t>> copies;
    for (size_t block = 0; block < blocks; ++block)
        copies[block] = {blocks + 1 + block};
    cache.copy_blocks(kv_group, copies);

    ov::test::utils::compare(decode(reference, config, block_range(0, blocks)),
                             decode(parallel, config, block_range(blocks + 1, blocks)),
                             1e-2,
                             1e-2);

    EXPECT_THROW(cache.copy_blocks(kv_group, {{2 * blocks + 1, {0}}}), ov::Exception);
    EXPECT_THROW(cache.copy_blocks(kv_group, {{0, {2 * blocks + 1}}}), ov::Exception);
}

TEST_F(TPPagedAttentionTest, ZeroedBlocksMatchAZeroInitializedCache) {
    BlockConfig config;
    auto compiled = core.compile_model(make_paged_attention_block(config), tp_device(world_size), tp_config());
    auto request = compiled.create_infer_request();
    auto cache = cache_of(compiled);
    const size_t block_size = block_size_of(cache);
    const size_t blocks = (prompt_tokens + block_size) / block_size;
    const auto used = block_range(0, blocks);

    // A decode over a zero-initialized cache, as if the prefill had written zeros.
    cache.allocate(kv_group, blocks, true);
    const auto over_zeros = decode(request, config, used);

    prefill(request, config, block_size);
    const auto over_prefill = decode(request, config, used);
    // The cache matters, or the comparison below proves nothing.
    EXPECT_GT(max_abs_diff(over_zeros, over_prefill), 1e-2f);

    cache.zero_blocks(kv_group, {used.begin(), used.end()});
    EXPECT_EQ(max_abs_diff(over_zeros, decode(request, config, used)), 0.0f);

    EXPECT_THROW(cache.zero_blocks(kv_group, {blocks}), ov::Exception);
}

}  // namespace
}  // namespace ov::tp::tests
