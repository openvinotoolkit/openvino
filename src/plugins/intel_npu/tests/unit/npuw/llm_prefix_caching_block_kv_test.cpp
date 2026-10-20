// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Request-level tests of prefix caching on top of the block KV cache.
//
// A cached block keeps references to the pool's own block tensors instead of copies, so
// the pool has to treat those tensors as read-only for as long as the cache holds them:
// never hand them out again as scratch memory, and never rewrite them in place. The
// interaction with the LongRoPE re-rotation, which rewrites keys in place, is covered in
// llm_longrope_mode_test.cpp.

#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <vector>

#include "kv_cache_block_manager_test_access.hpp"
#include "llm_block_kv_test_harness.hpp"

namespace {

using ov::test::npuw::build_llm_test_model;
using ov::test::npuw::KVCacheBlockManagerTestAccess;
using ov::test::npuw::LLMBlockKvTestAccess;
using ov::test::npuw::LLMKvTestPipeline;
using ov::test::npuw::to_floats;

class PrefixCacheBlockKv : public ov::test::npuw::LLMBlockKvTest {
protected:
    // The memory behind the key block at `seq_block_idx` of every layer.
    static std::map<uint32_t, const void*> key_block_data(ov::npuw::LLMInferRequest& req, size_t seq_block_idx) {
        std::map<uint32_t, const void*> out;
        for (const auto& [layer_idx, layer] : managers(req)) {
            const auto id = layer.key_manager->get_allocated_blocks().at(seq_block_idx);
            out[layer_idx] = layer.key_manager->get_block_tensor(id)->data();
        }
        return out;
    }

    static size_t allocated_blocks(ov::npuw::LLMInferRequest& req) {
        size_t total = 0;
        for (const auto& [layer_idx, layer] : managers(req)) {
            total += layer.key_manager->get_allocated_blocks().size();
            total += layer.value_manager->get_allocated_blocks().size();
        }
        return total;
    }
};

TEST_F(PrefixCacheBlockKv, BlockSizeFollowsThePrefillChunkSize) {
    LLMKvTestPipeline pipeline(build_llm_test_model(), block_props());
    EXPECT_EQ(LLMBlockKvTestAccess::block_size(LLMBlockKvTestAccess::block_strategy(pipeline.request())), kBlock);
}

TEST_F(PrefixCacheBlockKv, StoredBlockReferencesThePoolTensorsAndSharesThem) {
    LLMKvTestPipeline pipeline(build_llm_test_model(), block_props());
    pipeline.prefill(kTwoBlocks, 0);

    const auto block = store_block(pipeline.request(), 0u);
    ASSERT_NE(block, nullptr);
    EXPECT_EQ(block->get_token_start(), 0u);
    const auto& refs = as_pool_block(block).get_layer_refs();
    ASSERT_EQ(refs.size(), managers(pipeline.request()).size());

    for (const auto& [layer_idx, ref] : refs) {
        const auto& layer = managers(pipeline.request()).at(layer_idx);
        ASSERT_EQ(layer.key_manager->get_allocated_blocks().size(), 2u) << "layer " << layer_idx;
        const auto key_id = layer.key_manager->get_allocated_blocks().at(0);
        const auto value_id = layer.value_manager->get_allocated_blocks().at(0);
        EXPECT_EQ(ref.num_tokens, kBlock);
        // No copy: the cache points at the memory the model wrote into.
        EXPECT_EQ(ref.key_tensor->data(), layer.key_manager->get_block_tensor(key_id)->data());
        EXPECT_EQ(ref.value_tensor->data(), layer.value_manager->get_block_tensor(value_id)->data());
        EXPECT_TRUE(KVCacheBlockManagerTestAccess::is_shared(*layer.key_manager, key_id));
        EXPECT_TRUE(KVCacheBlockManagerTestAccess::is_shared(*layer.value_manager, value_id));
        // The second block is not part of this cache block.
        EXPECT_FALSE(KVCacheBlockManagerTestAccess::is_shared(*layer.key_manager,
                                                              layer.key_manager->get_allocated_blocks().at(1)));
    }
}

// The cache is an optimization, so a block the pool does not hold completely is skipped
// rather than failing the prefill. Declining it shares nothing, and the complete block that
// was cached before stays shared.
TEST_F(PrefixCacheBlockKv, IncompleteBlockIsDeclinedWhileTheCompleteOneStaysShared) {
    LLMKvTestPipeline pipeline(build_llm_test_model(), block_props());
    pipeline.prefill(kBlock + 8, 0);  // one full block and one holding only the last 8 tokens

    const auto& first_layer = managers(pipeline.request()).begin()->second;
    const auto allocated = first_layer.key_manager->get_allocated_blocks();
    ASSERT_EQ(allocated.size(), 2u);
    size_t partial = allocated.size();
    for (size_t i = 0; i < allocated.size(); ++i) {
        if (first_layer.key_manager->get_block_tokens(allocated[i]) != kBlock) {
            partial = i;
        }
    }
    ASSERT_LT(partial, allocated.size()) << "the prompt must leave one block partially filled";
    const size_t full = 1 - partial;

    ASSERT_NE(store_block(pipeline.request(), full * kBlock), nullptr);
    EXPECT_EQ(store_block(pipeline.request(), partial * kBlock), nullptr);
    EXPECT_EQ(store_block(pipeline.request(), allocated.size() * kBlock), nullptr);

    for (const auto& [layer_idx, layer] : managers(pipeline.request())) {
        for (const auto* manager : {layer.key_manager.get(), layer.value_manager.get()}) {
            const auto ids = manager->get_allocated_blocks();
            ASSERT_EQ(ids.size(), 2u);
            EXPECT_TRUE(KVCacheBlockManagerTestAccess::is_shared(*manager, ids[full])) << "layer " << layer_idx;
            EXPECT_FALSE(KVCacheBlockManagerTestAccess::is_shared(*manager, ids[partial])) << "layer " << layer_idx;
        }
    }
}

TEST_F(PrefixCacheBlockKv, RestoreInjectsTheCachedTensorsIntoAnEmptyPool) {
    LLMKvTestPipeline source(build_llm_test_model(), block_props());
    source.prefill(kTwoBlocks, 0);
    const auto block = store_block(source.request(), 0u);
    ASSERT_NE(block, nullptr);

    LLMKvTestPipeline target(build_llm_test_model(), block_props());
    LLMBlockKvTestAccess::block_strategy(target.request()).apply_cached_prefix_blocks({block});

    for (const auto& [layer_idx, ref] : as_pool_block(block).get_layer_refs()) {
        const auto& layer = managers(target.request()).at(layer_idx);
        ASSERT_EQ(layer.key_manager->get_allocated_blocks().size(), 1u);
        ASSERT_EQ(layer.value_manager->get_allocated_blocks().size(), 1u);
        const auto key_id = layer.key_manager->get_allocated_blocks().front();
        EXPECT_EQ(layer.key_manager->get_block_tensor(key_id)->data(), ref.key_tensor->data());
        EXPECT_EQ(layer.key_manager->get_block_tokens(key_id), kBlock);
        EXPECT_TRUE(KVCacheBlockManagerTestAccess::is_shared(*layer.key_manager, key_id));
    }
}

// A restore that cannot be completed must not leave the pool half restored.
TEST_F(PrefixCacheBlockKv, RejectedRestoreLeavesThePoolUntouched) {
    LLMKvTestPipeline source(build_llm_test_model(), block_props());
    source.prefill(kTwoBlocks, 0);
    const auto block = store_block(source.request(), 0u);
    ASSERT_NE(block, nullptr);

    LLMKvTestPipeline target(build_llm_test_model(), block_props());
    auto& strategy = LLMBlockKvTestAccess::block_strategy(target.request());

    // The valid block comes first, so a restore that injected as it went would already have
    // changed the pool when the second block is found to carry no KV.
    const auto without_kv = std::make_shared<ov::npuw::PrefixBlockBlockKV>(kBlock);
    EXPECT_THROW(strategy.apply_cached_prefix_blocks({block, without_kv}), ov::Exception);
    EXPECT_EQ(allocated_blocks(target.request()), 0u);

    // A block another strategy produced is not understood.
    const auto foreign = std::make_shared<ov::npuw::KVBlock>(kBlock);
    EXPECT_THROW(strategy.apply_cached_prefix_blocks({foreign}), ov::Exception);
    EXPECT_EQ(allocated_blocks(target.request()), 0u);

    // The restore has to precede the first prefill chunk.
    const size_t before = allocated_blocks(source.request());
    ASSERT_GT(before, 0u);
    EXPECT_THROW(LLMBlockKvTestAccess::block_strategy(source.request()).apply_cached_prefix_blocks({block}),
                 ov::Exception);
    EXPECT_EQ(allocated_blocks(source.request()), before);
}

TEST_F(PrefixCacheBlockKv, RepeatedPromptRestoresTheCachedBlockWithoutCopying) {
    LLMKvTestPipeline pipeline(build_llm_test_model(), prefix_props());
    pipeline.prefill(kTwoBlocks, 0);
    const auto first_run = key_block_data(pipeline.request(), 0);

    pipeline.prefill(kTwoBlocks, 0);

    // The first block comes straight out of the cache; the last chunk is always recomputed.
    EXPECT_EQ(key_block_data(pipeline.request(), 0), first_run);
    for (const auto& [layer_idx, layer] : managers(pipeline.request())) {
        EXPECT_EQ(layer.key_manager->get_allocated_blocks().size(), 2u);
        EXPECT_TRUE(KVCacheBlockManagerTestAccess::is_shared(*layer.key_manager,
                                                             layer.key_manager->get_allocated_blocks().at(0)));

        // The fake ignores its inputs, so check the binding: the recomputed chunk has to read
        // the restored block rather than a dummy.
        const std::string prefix = "past_key_values." + std::to_string(layer_idx);
        const auto key_tensor = layer.key_manager->get_block_tensor(layer.key_manager->get_allocated_blocks().at(0));
        const auto value_tensor =
            layer.value_manager->get_block_tensor(layer.value_manager->get_allocated_blocks().at(0));
        EXPECT_EQ(LLMBlockKvTestAccess::prefill_past(pipeline.request(), prefix + ".key_block_0")->data(),
                  key_tensor->data())
            << "layer " << layer_idx;
        EXPECT_EQ(LLMBlockKvTestAccess::prefill_past(pipeline.request(), prefix + ".value_block_0")->data(),
                  value_tensor->data())
            << "layer " << layer_idx;
    }
}

// The pool keeps blocks warm across conversations, which is only safe for tensors nobody else
// holds: a different prompt must not write over what the cache still serves.
TEST_F(PrefixCacheBlockKv, CachedTensorIsNotReusedAsScratchByAnotherPrompt) {
    LLMKvTestPipeline pipeline(build_llm_test_model(), prefix_props());
    pipeline.prefill(kTwoBlocks, 0, 1);
    const auto cached = key_block_data(pipeline.request(), 0);

    pipeline.prefill(kTwoBlocks, 0, 2);

    const auto current = key_block_data(pipeline.request(), 0);
    for (const auto& [layer_idx, data] : cached) {
        EXPECT_NE(current.at(layer_idx), data) << "layer " << layer_idx;
    }
}

// A generate step that rewinds into a cached block writes new KV there, which must go to a
// private copy: the cache keeps serving the original prompt.
TEST_F(PrefixCacheBlockKv, GenerateRewoundIntoCachedBlockLeavesItUntouched) {
    LLMKvTestPipeline pipeline(build_llm_test_model(), prefix_props());
    pipeline.prefill(kTwoBlocks, 0);

    std::map<uint32_t, ov::SoPtr<ov::ITensor>> cached;
    std::map<uint32_t, std::vector<float>> cached_before;
    for (const auto& [layer_idx, layer] : managers(pipeline.request())) {
        cached[layer_idx] = layer.key_manager->get_block_tensor(layer.key_manager->get_allocated_blocks().at(0));
        // The fake writes the same value every time, so zero the block to make a write visible.
        ov::npuw::util::fill_tensor_bytes(cached[layer_idx], 0u);
        cached_before[layer_idx] = to_floats(cached[layer_idx]);
    }

    pipeline.generate_step(kBlock - 1, kBlock - 1);  // position 31 is the last row of block 0

    for (const auto& [layer_idx, layer] : managers(pipeline.request())) {
        const auto id = layer.key_manager->get_allocated_blocks().at(0);
        const auto pooled = layer.key_manager->get_block_tensor(id);
        EXPECT_EQ(to_floats(cached.at(layer_idx)), cached_before.at(layer_idx))
            << "cached keys changed, layer " << layer_idx;
        EXPECT_NE(pooled->data(), cached.at(layer_idx)->data()) << "layer " << layer_idx;
        EXPECT_NE(to_floats(pooled), cached_before.at(layer_idx)) << "the step did not write, layer " << layer_idx;
        EXPECT_FALSE(KVCacheBlockManagerTestAccess::is_shared(*layer.key_manager, id)) << "layer " << layer_idx;

        // The generate request has to read the copy that now holds the new row.
        const std::string port = "past_key_values." + std::to_string(layer_idx) + ".key_block_0";
        EXPECT_EQ(LLMBlockKvTestAccess::generate_past(pipeline.request(), port)->data(), pooled->data())
            << "layer " << layer_idx;
    }

    // The same prompt again hits the cache and gets the original block back.
    pipeline.prefill(kTwoBlocks, 0);
    for (const auto& [layer_idx, layer] : managers(pipeline.request())) {
        const auto id = layer.key_manager->get_allocated_blocks().at(0);
        const auto restored = layer.key_manager->get_block_tensor(id);
        EXPECT_EQ(restored->data(), cached.at(layer_idx)->data()) << "layer " << layer_idx;
        EXPECT_EQ(to_floats(restored), cached_before.at(layer_idx)) << "layer " << layer_idx;
    }
}

}  // namespace
