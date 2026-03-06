// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/paged_cache_manager.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "openvino/reference/paged_attention.hpp"

using namespace ov::reference::paged_attention_cache;

namespace {

// helper to build a zero-filled cache shape [num_blocks, kv_heads, block_size, head_size]
struct CacheLayout {
    std::size_t num_blocks;
    std::size_t kv_heads;
    std::size_t block_size;
    std::size_t head_size;

    ov::Shape shape() const {
        return {num_blocks, kv_heads, block_size, head_size};
    }
    std::size_t elems() const {
        return num_blocks * kv_heads * block_size * head_size;
    }
};

// shorthand node key
// it is a random value that pretends to be a PagedAttention operator ID
constexpr std::uintptr_t NODE = 67;

// register a simple operator state and return the manager
std::unique_ptr<PagedCacheManager> make_manager(CacheLayout layout,
                                                EvictionPolicy policy,
                                                std::vector<float>& key_data,
                                                std::vector<float>& val_data,
                                                std::size_t seq_count = 1) {
    key_data.assign(layout.elems(), 0.f);
    val_data.assign(layout.elems(), 0.f);

    auto mgr = std::make_unique<PagedCacheManager>(ov::element::f32, policy, /*max_cache_bytes=*/0);

    // initial past_lens = 0 for each sequence
    std::vector<std::int32_t> past(seq_count, 0);
    // no initial block indices
    mgr->ensure_operator(NODE,
                         key_data.data(),
                         val_data.data(),
                         layout.shape(),
                         layout.shape(),
                         nullptr,
                         0,
                         nullptr,
                         0,
                         past.data(),
                         seq_count);
    return mgr;
}

}  // namespace

// Basic lifecycle: register, begin_step, ensure_token, resolve_token
TEST(PagedCacheManagerTest, BasicAllocateAndResolve) {
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    // write a token at position 0
    std::vector<float> krow(layout.kv_heads * layout.head_size, 1.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 2.f);
    mgr->write_token_kv<float>(NODE, 0, 0, krow.data(), vrow.data());

    // resolve it
    PagedCacheManager::TokenAddress addr;
    ASSERT_TRUE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_GE(addr.block, 0);

    // read back key data
    const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
    ASSERT_NE(kp, nullptr);
    EXPECT_FLOAT_EQ(kp[0], 1.f);

    // read back value data
    const float* vp = mgr->value_ptr<float>(NODE, addr, 0);
    ASSERT_NE(vp, nullptr);
    EXPECT_FLOAT_EQ(vp[0], 2.f);
}

// FIFO eviction: fills all blocks then forces eviction from oldest
TEST(PagedCacheManagerTest, FIFOEvictionWorks) {
    // 4 blocks, block_size=2 -> can hold 8 tokens total
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd, 2);

    EXPECT_EQ(mgr->eviction_policy(), EvictionPolicy::FIFO);

    // fill seq 0 with 6 tokens (3 blocks)
    std::int32_t past0 = 0;
    std::int32_t past1 = 0;
    std::int32_t pasts[2] = {past0, past1};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 6; t++) {
        krow[0] = static_cast<float>(t + 100);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // fill seq 1 with 2 tokens (1 block) -> total 4 blocks used up
    for (int t = 0; t < 2; t++) {
        krow[0] = static_cast<float>(t + 200);
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // now all 4 blocks are used, write one more token for seq 1 which needs a new block
    // this should evict the oldest block from seq 0 (the longest)
    krow[0] = 999.f;
    mgr->write_token_kv<float>(NODE, 1, 2, krow.data(), vrow.data());

    // seq 0 tokens 0 and 1 (the first block) should be evicted
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));

    // seq 0 tokens 2-5 should still be accessible
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 5, addr));

    // seq 1 token 2 should be accessible (the new one)
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
    const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
    ASSERT_NE(kp, nullptr);
    EXPECT_FLOAT_EQ(kp[0], 999.f);
}

// SCORE eviction: picks the sequence whose front block has lowest score
TEST(PagedCacheManagerTest, ScoreEvictionPicksLowestScoreFrontBlock) {
    // 4 blocks, block_size=2, 1 head, head_size=4
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 2);

    EXPECT_EQ(mgr->eviction_policy(), EvictionPolicy::SCORE);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 4 tokens -> 2 blocks
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 4 tokens -> 2 blocks, now all 4 blocks used
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t + 10);
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // feed attention scores so that seq 0 front block has LOW score and seq 1 front block has HIGH score
    // layout: [seq0_4_tokens, seq1_4_tokens]
    pasts[0] = 4;
    pasts[1] = 4;
    // seq 0 block 0 tokens: 0.1 each -> front block score 0.2
    // seq 0 block 1 tokens: 5.0 each -> block score 10.0
    // seq 1 block 0 tokens: 3.0 each -> front block score 6.0
    // seq 1 block 1 tokens: 3.0 each -> block score 6.0
    std::vector<float> scores = {0.1f, 0.1f, 5.0f, 5.0f, 3.f, 3.f, 3.f, 3.f};
    mgr->update_attention_scores(NODE, scores.data(), scores.size(), pasts, 2);

    // force a new allocation for seq 1 which will trigger eviction
    // should evict seq 0 front block (score 0.2) not seq 1 front block (score 6.0)
    krow[0] = 777.f;
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    // seq 0 tokens 0,1 should be evicted (front block of seq 0 had lowest score)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));

    // seq 0 tokens 2,3 should survive
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 3, addr));

    // all seq 1 tokens should be intact
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 3, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));
}

// SCORE eviction falls back to FIFO when no scores are recorded
TEST(PagedCacheManagerTest, ScoreEvictionFallsBackToFifo) {
    // 4 blocks, block_size=2, 2 sequences
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 6 tokens -> 3 blocks
    for (int t = 0; t < 6; t++) {
        krow[0] = static_cast<float>(t);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 2 tokens -> 1 block, total 4 blocks used
    for (int t = 0; t < 2; t++) {
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // no scores fed, force eviction by adding a 5th block for seq 1
    krow[0] = 888.f;
    mgr->write_token_kv<float>(NODE, 1, 2, krow.data(), vrow.data());

    // FIFO fallback should evict oldest block from seq 0 (longest)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
}

// ADAPTIVE_RKV eviction: evicts from the sequence whose front block is least diverse
TEST(PagedCacheManagerTest, AdaptiveRKVEvictionPicksLeastDiverse) {
    // 4 blocks, block_size=2, eviction_size=4 tokens
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    EXPECT_EQ(mgr->eviction_policy(), EvictionPolicy::ADAPTIVE_RKV);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 4 tokens -> 2 blocks
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 4 tokens -> 2 blocks, all 4 blocks used
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t + 10);
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // Feed attention scores: seq 0 block 1 is important, block 0 is not.
    // seq 1 both blocks are equally important.
    // scores layout: [seq0: 4 tokens][seq1: 4 tokens]
    std::int32_t pasts_for_scores[2] = {4, 4};
    //                   seq0: tok0  tok1  tok2  tok3    seq1: tok0  tok1  tok2  tok3
    float scores[8] = {0.1f, 0.1f, 5.0f, 5.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    mgr->update_attention_scores(NODE, scores, 8, pasts_for_scores, 2);

    // Feed 2D diversity matrices [2 blocks, 4 tokens].
    // seq 0: front block row has low diversity to retained set
    // seq 1: front block row has high diversity
    // eviction_size = 4 (2 blocks * 2 tokens/block)
    float div0[8] = {// block 0 row: diversity to each of the 4 tokens
                     0.1f,
                     0.1f,
                     0.1f,
                     0.1f,
                     // block 1 row:
                     5.0f,
                     5.0f,
                     5.0f,
                     5.0f};
    float div1[8] = {// block 0 row:
                     3.0f,
                     3.0f,
                     3.0f,
                     3.0f,
                     // block 1 row:
                     3.0f,
                     3.0f,
                     3.0f,
                     3.0f};
    mgr->update_diversity_scores(NODE, 0, div0, 2, 4, 0);
    mgr->update_diversity_scores(NODE, 1, div1, 2, 4, 0);

    // force eviction by writing token 4 for seq 1
    krow[0] = 666.f;
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    // seq 0 front block (tokens 0,1) should be evicted:
    // - seq 0 block 0 has low attention (0.2 total), low diversity (0.1)
    // - seq 1 block 0 has moderate attention (4.0), moderate diversity (3.0)
    // With attention-mass gating (p=0.9), seq 0's block 1 (score 10.0) covers 98%+ of
    // attention mass, so block 0 is NOT retained -> eligible for eviction.
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));  // block 1 survived
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));  // seq 1 intact
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));  // new token accessible
}

// ADAPTIVE_RKV falls back to SCORE then FIFO if no diversity data
TEST(PagedCacheManagerTest, AdaptiveRKVFallsBackToScoreThenFifo) {
    // 4 blocks, 2 sequences
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 6 tokens -> 3 blocks
    for (int t = 0; t < 6; t++) {
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }
    // seq 1: 2 tokens -> 1 block, total 4 blocks
    for (int t = 0; t < 2; t++) {
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // no diversity, no scores -> should fall all the way back to FIFO
    krow[0] = 111.f;
    mgr->write_token_kv<float>(NODE, 1, 2, krow.data(), vrow.data());

    // FIFO should evict from seq 0 (longest)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
}

// Sequence reset clears blocks and score/diversity state
TEST(PagedCacheManagerTest, SequenceResetClearsState) {
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 1);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    std::vector<float> krow(4, 1.f);
    std::vector<float> vrow(4, 1.f);
    for (int t = 0; t < 4; t++) {
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // feed some scores
    past = 4;
    std::vector<float> scores = {1.f, 2.f, 3.f, 4.f};
    mgr->update_attention_scores(NODE, scores.data(), scores.size(), &past, 1);

    // reset the sequence by setting past_lens to 0
    past = 0;
    mgr->begin_step(NODE, &past, 1);

    // all old tokens should be gone
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 3, addr));

    // should be able to write new tokens without issues (blocks were freed)
    mgr->write_token_kv<float>(NODE, 0, 0, krow.data(), vrow.data());
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 0, addr));
}

// Multiple evictions in a row dont corrupt memory
TEST(PagedCacheManagerTest, RepeatedEvictionsNoCorruption) {
    // small pool: 3 blocks of size 2
    CacheLayout layout{3, 2, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd, 1);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // write 20 tokens, which forces many evictions with only 3 blocks (6 token slots)
    for (int t = 0; t < 20; t++) {
        for (std::size_t j = 0; j < krow.size(); j++) {
            krow[j] = static_cast<float>(t * 100 + j);
        }
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // only the last ~6 tokens should be resolvable (3 blocks * 2 tokens/block)
    // earlier ones were evicted
    PagedCacheManager::TokenAddress addr;
    int resolvable = 0;
    for (int t = 0; t < 20; t++) {
        if (mgr->resolve_token(NODE, 0, t, addr)) {
            resolvable++;
            // check that we can read back the key data without crashing
            const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
            ASSERT_NE(kp, nullptr);
            EXPECT_FLOAT_EQ(kp[0], static_cast<float>(t * 100));
        }
    }

    // we expect roughly 6 tokens to be resolvable (3 blocks * 2)
    EXPECT_GE(resolvable, 4);
    EXPECT_LE(resolvable, 8);  // some slack for boundary effects
}

// max_cache_bytes triggers eviction before physical blocks run out
TEST(PagedCacheManagerTest, MaxCacheBytesTriggersEviction) {
    // 8 physical blocks, block_size=2, 1 kv head, head_size=4 (float=4B)
    // key_block_bytes = value_block_bytes = 1*2*4*4 = 32 bytes
    // bytes_per_block = 64 -> max_cache_bytes=192 allows 3 active blocks
    CacheLayout layout{8, 1, 2, 4};
    std::vector<float> kd(layout.elems(), 0.f);
    std::vector<float> vd(layout.elems(), 0.f);

    auto mgr = std::make_unique<PagedCacheManager>(ov::element::f32,
                                                   EvictionPolicy::FIFO,
                                                   /*max_cache_bytes=*/192);

    std::vector<std::int32_t> past(2, 0);
    mgr->ensure_operator(NODE,
                         kd.data(),
                         vd.data(),
                         layout.shape(),
                         layout.shape(),
                         nullptr,
                         0,
                         nullptr,
                         0,
                         past.data(),
                         2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 4 tokens -> 2 blocks
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t + 100);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // seq 1: 2 tokens -> 1 block.  Total 3 active blocks = budget limit
    for (int t = 0; t < 2; t++) {
        krow[0] = static_cast<float>(t + 200);
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());
    }

    // Writing one more token for seq 1 needs a 4th block, but budget allows only 3,
    // so eviction fires even though 5 physical blocks are still free
    krow[0] = 999.f;
    mgr->write_token_kv<float>(NODE, 1, 2, krow.data(), vrow.data());

    // FIFO evicts oldest block from longest sequence 
    // (seq 0 front block -> tokens 0,1)
    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 3, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
    const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
    ASSERT_NE(kp, nullptr);
    EXPECT_FLOAT_EQ(kp[0], 999.f);
}

// Default constructor uses SCORE policy
TEST(PagedCacheManagerTest, DefaultConstructorUsesScore) {
    PagedCacheManager mgr(ov::element::f32);
    EXPECT_EQ(mgr.eviction_policy(), EvictionPolicy::SCORE);
}

// Reference PA kernel with tiny cache forces eviction, no corruption
TEST(PagedCacheManagerTest, ReferenceKernelEvictionNoCorruption) {
    // minimal config: 1 kv head, head_size=4, block_size=2, only 3 blocks
    // with 3 blocks we can hold 6 tokens, so a sequence of 8 forces eviction
    const std::size_t num_blocks = 3;
    const std::size_t kv_heads = 1;
    const std::size_t blk_size = 2;
    const std::size_t head_size = 4;
    const std::size_t q_heads = 1;
    const std::size_t q_features = q_heads * head_size;
    const std::size_t kv_features = kv_heads * head_size;

    // initial (empty) key and value caches
    const std::size_t cache_elems = num_blocks * kv_heads * blk_size * head_size;
    std::vector<float> key_cache(cache_elems, 0.f);
    std::vector<float> val_cache(cache_elems, 0.f);
    ov::Shape cache_shape = {num_blocks, kv_heads, blk_size, head_size};

    // create a manager with SCORE policy
    auto mgr = std::make_unique<PagedCacheManager>(ov::element::f32, EvictionPolicy::SCORE, 0);

    // empty block indices for initial registration
    std::vector<std::int32_t> block_idx = {0, 1, 2};
    std::vector<std::int32_t> block_begins = {0, 3};

    // run two steps: first step with 6 tokens (fills cache), second step with 2 more (forces eviction)
    constexpr std::uintptr_t nk = 99;

    // --- step 1: prompt with 6 tokens ---
    const std::size_t step1_tokens = 6;
    std::vector<float> q1(step1_tokens * q_features);
    std::vector<float> k1(step1_tokens * kv_features);
    std::vector<float> v1(step1_tokens * kv_features);
    for (std::size_t i = 0; i < q1.size(); i++)
        q1[i] = 0.1f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < k1.size(); i++)
        k1[i] = 0.2f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < v1.size(); i++)
        v1[i] = 0.3f * static_cast<float>(i + 1);

    std::int32_t past1 = 0;
    std::int32_t subseq1[2] = {0, static_cast<std::int32_t>(step1_tokens)};
    float scale_val = 1.0f / std::sqrt(static_cast<float>(head_size));
    std::int32_t score_window = 0;

    std::vector<float> out1(step1_tokens * q_features, -999.f);

    ov::reference::paged_attention<float>(nk,
                                          mgr.get(),
                                          out1.data(),
                                          nullptr,
                                          nullptr,
                                          q1.data(),
                                          k1.data(),
                                          v1.data(),
                                          key_cache.data(),
                                          val_cache.data(),
                                          &past1,
                                          subseq1,
                                          block_idx.data(),
                                          block_idx.size(),
                                          block_begins.data(),
                                          block_begins.size(),
                                          &scale_val,
                                          ov::element::f32,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          &score_window,
                                          1,
                                          nullptr,
                                          0,
                                          nullptr,
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          {step1_tokens, q_features},
                                          {step1_tokens, kv_features},
                                          {step1_tokens, kv_features},
                                          cache_shape,
                                          cache_shape,
                                          {1},
                                          {2});

    // check no NaN or inf in output
    for (std::size_t i = 0; i < out1.size(); i++) {
        EXPECT_TRUE(std::isfinite(out1[i])) << "step 1 output[" << i << "] is not finite";
    }

    // --- step 2: decode with 2 more tokens, forces eviction ---
    const std::size_t step2_tokens = 2;
    std::vector<float> q2(step2_tokens * q_features);
    std::vector<float> k2(step2_tokens * kv_features);
    std::vector<float> v2(step2_tokens * kv_features);
    for (std::size_t i = 0; i < q2.size(); i++)
        q2[i] = 0.4f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < k2.size(); i++)
        k2[i] = 0.5f * static_cast<float>(i + 1);
    for (std::size_t i = 0; i < v2.size(); i++)
        v2[i] = 0.6f * static_cast<float>(i + 1);

    std::int32_t past2 = static_cast<std::int32_t>(step1_tokens);
    std::int32_t subseq2[2] = {0, static_cast<std::int32_t>(step2_tokens)};

    std::vector<float> out2(step2_tokens * q_features, -999.f);

    ov::reference::paged_attention<float>(nk,
                                          mgr.get(),
                                          out2.data(),
                                          nullptr,
                                          nullptr,
                                          q2.data(),
                                          k2.data(),
                                          v2.data(),
                                          key_cache.data(),
                                          val_cache.data(),
                                          &past2,
                                          subseq2,
                                          block_idx.data(),
                                          block_idx.size(),
                                          block_begins.data(),
                                          block_begins.size(),
                                          &scale_val,
                                          ov::element::f32,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          &score_window,
                                          1,
                                          nullptr,
                                          0,
                                          nullptr,
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          {},
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          ov::element::Type{},
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          {step2_tokens, q_features},
                                          {step2_tokens, kv_features},
                                          {step2_tokens, kv_features},
                                          cache_shape,
                                          cache_shape,
                                          {1},
                                          {2});

    // check no NaN, inf, or uninitialized values in output
    for (std::size_t i = 0; i < out2.size(); i++) {
        EXPECT_TRUE(std::isfinite(out2[i])) << "step 2 output[" << i << "] is not finite";
        EXPECT_NE(out2[i], -999.f) << "step 2 output[" << i << "] was never written";
    }
}

// ===================================================================
// FIFO: additional coverage
// ===================================================================

// Single sequence forces self-eviction when no other victim exists
TEST(PagedCacheManagerTest, FIFOStealsFromSelfAsLastResort) {
    // 2 blocks, block_size=2, single sequence
    CacheLayout layout{2, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd, 1);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // fill 2 blocks (4 tokens)
    for (int t = 0; t < 4; t++) {
        krow[0] = static_cast<float>(t + 1);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // writing token 4 forces self-eviction of front block
    krow[0] = 50.f;
    mgr->write_token_kv<float>(NODE, 0, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 4, addr));
    const float* kp = mgr->key_ptr<float>(NODE, addr, 0);
    ASSERT_NE(kp, nullptr);
    EXPECT_FLOAT_EQ(kp[0], 50.f);
}

// Data integrity: surviving blocks retain correct key/value data after eviction
TEST(PagedCacheManagerTest, FIFOEvictionPreservesSurvivingData) {
    // 3 blocks, 2 kv_heads, block_size=2, head_size=4
    CacheLayout layout{3, 2, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd, 1);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    // write 6 tokens (fills 3 blocks), each with distinct per-head key values
    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 6; t++) {
        for (std::size_t j = 0; j < krow.size(); j++) {
            krow[j] = static_cast<float>(t * 100 + j);
            vrow[j] = static_cast<float>(t * 1000 + j);
        }
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // force eviction (front block tokens 0,1 evicted)
    for (std::size_t j = 0; j < krow.size(); j++) {
        krow[j] = static_cast<float>(6 * 100 + j);
        vrow[j] = static_cast<float>(6 * 1000 + j);
    }
    mgr->write_token_kv<float>(NODE, 0, 6, krow.data(), vrow.data());

    // verify surviving tokens 2-6 have correct data in both kv heads
    for (int t = 2; t <= 6; t++) {
        PagedCacheManager::TokenAddress addr;
        ASSERT_TRUE(mgr->resolve_token(NODE, 0, t, addr)) << "token " << t;
        for (std::size_t h = 0; h < 2; h++) {
            const float* kp = mgr->key_ptr<float>(NODE, addr, h);
            ASSERT_NE(kp, nullptr);
            EXPECT_FLOAT_EQ(kp[0], static_cast<float>(t * 100 + h * layout.head_size))
                << "key head " << h << " token " << t;

            const float* vp = mgr->value_ptr<float>(NODE, addr, h);
            ASSERT_NE(vp, nullptr);
            EXPECT_FLOAT_EQ(vp[0], static_cast<float>(t * 1000 + h * layout.head_size))
                << "value head " << h << " token " << t;
        }
    }
}

// FIFO with equal-length sequences picks non-requester
TEST(PagedCacheManagerTest, FIFOEqualLengthPrefersNonRequester) {
    // 4 blocks, block_size=2, 2 sequences each with 2 blocks
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::FIFO, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // seq 0: 4 tokens -> 2 blocks
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    // seq 1: 4 tokens -> 2 blocks
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    // seq 1 requests a new block -> evicts from seq 0 (non-requester)
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));  // seq 0 front evicted
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));   // seq 0 back survived
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));   // seq 1 intact
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));   // new token ok
}

// Score eviction penalizes the requester sequence (prefers non-requester)
TEST(PagedCacheManagerTest, ScoreEvictionPrefersNonRequester) {
    // 4 blocks, 2 sequences.  Seq 1 (the requester) has a much lower front-block score
    // than seq 0, but the penalty should prevent self-eviction.
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    // seq 0 front block has medium score, seq 1 front block has very low score
    pasts[0] = 4;
    pasts[1] = 4;
    float scores[8] = {5.f, 5.f, 5.f, 5.f, 0.01f, 0.01f, 5.f, 5.f};
    mgr->update_attention_scores(NODE, scores, 8, pasts, 2);

    // seq 1 requests a new block.  Without the penalty, seq 1's front block (0.02)
    // would be evicted, but the 1e12 penalty makes seq 0 (10.0) the victim instead.
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));  // seq 0 front evicted
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));   // seq 1 front preserved
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));
}

// Score updates accumulate across multiple calls
TEST(PagedCacheManagerTest, ScoreEvictionAccumulatesScores) {
    // 4 blocks, 2 sequences.
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    pasts[0] = 4;
    pasts[1] = 4;

    // First update: seq 0 front = 1.0, seq 1 front = 2.0
    float scores1[8] = {0.5f, 0.5f, 1.f, 1.f, 1.0f, 1.0f, 1.f, 1.f};
    mgr->update_attention_scores(NODE, scores1, 8, pasts, 2);
    // Second update: add more to seq 0 front, making it 1.0 + 10.0 = 11.0
    float scores2[8] = {5.0f, 5.0f, 0.f, 0.f, 0.0f, 0.0f, 0.f, 0.f};
    mgr->update_attention_scores(NODE, scores2, 8, pasts, 2);

    // Now seq 0 front = 11.0, seq 1 front = 2.0.
    // Seq 0 requests the new block.  SCORE penalizes requester (seq 0) by +1e12,
    // so seq 1's front (2.0) < seq 0's penalized front (11 + 1e12) -> seq 1 evicted.
    mgr->write_token_kv<float>(NODE, 0, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 0, addr));   // seq 0 front survived (requester, penalized)
    EXPECT_FALSE(mgr->resolve_token(NODE, 1, 0, addr));  // seq 1 front evicted (2.0)
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 4, addr));
}

// Score eviction with single sequence (forced self-eviction via score)
TEST(PagedCacheManagerTest, ScoreEvictionSingleSequenceSelfEvicts) {
    CacheLayout layout{3, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::SCORE, kd, vd, 1);

    std::int32_t past = 0;
    mgr->begin_step(NODE, &past, 1);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 6; t++) {
        krow[0] = static_cast<float>(t);
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    }

    // feed scores: front block (tokens 0,1) has low score
    past = 6;
    float scores[6] = {0.1f, 0.1f, 5.0f, 5.0f, 5.0f, 5.0f};
    mgr->update_attention_scores(NODE, scores, 6, &past, 1);

    // force self-eviction
    krow[0] = 99.f;
    mgr->write_token_kv<float>(NODE, 0, 6, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 1, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 6, addr));
}

// Attention-mass gating protects a front block that is highly attended
TEST(PagedCacheManagerTest, AdaptiveRKVGatingProtectsRetainedFrontBlock) {
    // 6 blocks, block_size=2, 3 sequences.
    // Seq 0: front block has HIGH attention -> retained -> protected from eviction
    // Seq 1: front block has LOW attention -> not retained -> eligible
    // Seq 2: front block has LOW attention -> not retained -> eligible
    // Expect seq 1 or 2 to be evicted (depending on diversity), not seq 0
    CacheLayout layout{6, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 3);

    std::int32_t pasts[3] = {0, 0, 0};
    mgr->begin_step(NODE, pasts, 3);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);

    // each seq: 4 tokens -> 2 blocks, total 6 blocks
    for (int s = 0; s < 3; s++)
        for (int t = 0; t < 4; t++)
            mgr->write_token_kv<float>(NODE, static_cast<std::size_t>(s), t, krow.data(), vrow.data());

    // Attention scores: seq 0 front block is dominant (90% of mass)
    std::int32_t pasts_s[3] = {4, 4, 4};
    // seq 0: block0=9.0 block1=1.0 -> p=0.9, target=9.0, block0 alone covers it -> retained
    // seq 1: block0=0.1 block1=9.9 -> target=9.0, block1 covers it -> block0 NOT retained
    // seq 2: block0=0.1 block1=9.9 -> same as seq 1
    float scores[12] = {4.5f, 4.5f, 0.5f, 0.5f, 0.05f, 0.05f, 4.95f, 4.95f, 0.05f, 0.05f, 4.95f, 4.95f};
    mgr->update_attention_scores(NODE, scores, 12, pasts_s, 3);

    // Diversity: seq 1 front has lower diversity than seq 2 front
    float div0[8] = {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f};
    float div1[8] = {0.1f, 0.1f, 0.1f, 0.1f, 5.f, 5.f, 5.f, 5.f};
    float div2[8] = {3.f, 3.f, 3.f, 3.f, 5.f, 5.f, 5.f, 5.f};
    mgr->update_diversity_scores(NODE, 0, div0, 2, 4, 0);
    mgr->update_diversity_scores(NODE, 1, div1, 2, 4, 0);
    mgr->update_diversity_scores(NODE, 2, div2, 2, 4, 0);

    // force eviction from seq 2
    mgr->write_token_kv<float>(NODE, 2, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    // seq 0 must survive entirely (front block is retained by gating)
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 0, addr)) << "seq 0 front block should be protected";
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));

    // seq 1 front block should be evicted (lowest diversity among non-retained fronts)
    EXPECT_FALSE(mgr->resolve_token(NODE, 1, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));

    // seq 2 survived + new token
    EXPECT_TRUE(mgr->resolve_token(NODE, 2, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 2, 4, addr));
}

// When all front blocks are retained by gating, fall back to SCORE eviction
TEST(PagedCacheManagerTest, AdaptiveRKVFallsBackToScoreWhenAllRetained) {
    // 4 blocks, 2 sequences.   Both front blocks have dominant attention -> both retained
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    // Both front blocks carry the majority of attention mass
    std::int32_t pasts_s[2] = {4, 4};
    float scores[8] = {9.0f, 9.0f, 1.0f, 1.0f, 9.0f, 9.0f, 1.0f, 1.0f};
    mgr->update_attention_scores(NODE, scores, 8, pasts_s, 2);

    // Diversity data present, but both front blocks are retained by attention-mass gating
    float div[8] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    mgr->update_diversity_scores(NODE, 0, div, 2, 4, 0);
    mgr->update_diversity_scores(NODE, 1, div, 2, 4, 0);

    // force eviction from seq 1     (both candidates are empty -> falls back to SCORE)
    // SCORE: seq 0 front=18.0, seq 1 front=18.0 (tied), but requester (seq 1) gets penalty
    // -> seq 0 front block evicted
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));  // evicted via SCORE fallback
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));
}

// Filtered column mean: only columns of retained blocks contribute to diversity score
TEST(PagedCacheManagerTest, AdaptiveRKVFilteredColumnMean) {
    // 6 blocks, block_size=2, 2 sequences
    // 3 blocks per seq (6 tokens each), eviction_size = 6, eviction zone = 3 blocks
    CacheLayout layout{6, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 6; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    for (int t = 0; t < 6; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    // Attention scores: both seqs have low front block, high blocks 1,2
    std::int32_t pasts_s[2] = {6, 6};
    // seq 0: block0=0.2, block1=5.0, block2=5.0 -> target=9.18, blocks 1+2 (10.0) cover it
    // seq 1: block0=0.2, block1=5.0, block2=5.0 -> same
    float scores[12] = {0.1f, 0.1f, 2.5f, 2.5f, 2.5f, 2.5f, 0.1f, 0.1f, 2.5f, 2.5f, 2.5f, 2.5f};
    mgr->update_attention_scores(NODE, scores, 12, pasts_s, 2);

    // Diversity matrices [3 blocks, 6 tokens]:
    // Seq 0 front-block row: high in retained columns (blocks 1,2), low in non-retained (block 0)
    //   tokens: [blk0:t0 blk0:t1 | blk1:t2 blk1:t3 | blk2:t4 blk2:t5]
    //   row 0:  [0.0  0.0  |  10.0  10.0  |  10.0  10.0 ]
    //   filtered mean (only retained blk1,blk2 columns): mean(10,10,10,10) = 10.0
    float div0[18] = {
        0.0f,
        0.0f,
        10.0f,
        10.0f,
        10.0f,
        10.0f,  // block 0 row
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,  // block 1 row
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,  // block 2 row
    };

    // Seq 1 front-block row: low everywhere
    //   row 0:  [0.0  0.0  |  1.0  1.0  |  1.0  1.0 ]
    //   filtered mean (retained blk1,blk2): mean(1,1,1,1) = 1.0
    float div1[18] = {
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f,
        1.0f,  // block 0 row
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,  // block 1 row
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,
        5.0f,  // block 2 row
    };

    mgr->update_diversity_scores(NODE, 0, div0, 3, 6, 0);
    mgr->update_diversity_scores(NODE, 1, div1, 3, 6, 0);

    // eviction via seq 0 requesting a new block
    mgr->write_token_kv<float>(NODE, 0, 6, krow.data(), vrow.data());

    // Seq 1 front block has filtered diversity 1.0 < seq 0's 10.0 -> seq 1 is evicted
    PagedCacheManager::TokenAddress addr;
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 0, addr));   // seq 0 front survived (div=10.0)
    EXPECT_FALSE(mgr->resolve_token(NODE, 1, 0, addr));  // seq 1 front evicted (div=1.0)
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));   // seq 1 back survived
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 6, addr));   // new token ok
}

// start_block_offset > 0: front block is before the eviction zone (in start area)
TEST(PagedCacheManagerTest, AdaptiveRKVStartBlockOffset) {
    // 6 blocks, block_size=2, 2 sequences × 3 blocks each.
    // start_block_offset=1 means the eviction zone begins at block index 1, so the
    // front block of each sequence (index 0) is in the "start area" and has no
    // diversity row.  It gets div_score=0 and is always evictable (not retained by
    // the diversity data check)
    CacheLayout layout{6, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 6; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    for (int t = 0; t < 6; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    // Feed attention scores
    std::int32_t pasts_s[2] = {6, 6};
    // seq 0: low front
    // seq 1: also low front
    float scores[12] = {0.1f, 0.1f, 5.f, 5.f, 5.f, 5.f, 0.2f, 0.2f, 5.f, 5.f, 5.f, 5.f};
    mgr->update_attention_scores(NODE, scores, 12, pasts_s, 2);

    // Diversity covering blocks 1,2 (zone): 2 zone blocks, eviction_size=4, start_block_offset=1
    // Since front block (deque index 0) is outside the zone (start_blk=1),
    // start_blk > 0 -> front_is_retained=false and div_score defaults to 0
    float div0[8] = {8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f};
    float div1[8] = {8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f, 8.f};
    mgr->update_diversity_scores(NODE, 0, div0, 2, 4, /*start_block_offset=*/1);
    mgr->update_diversity_scores(NODE, 1, div1, 2, 4, /*start_block_offset=*/1);

    // Both front blocks get div_score=0, penalty differentiates.
    // Requester (seq 0) gets +1e12 penalty -> seq 1 front block evicted
    mgr->write_token_kv<float>(NODE, 0, 6, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 0, addr));
    EXPECT_FALSE(mgr->resolve_token(NODE, 1, 0, addr));  // seq 1 front evicted
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 6, addr));
}

// ADAPTIVE_RKV with scores but no diversity -> falls back to SCORE (not FIFO)
TEST(PagedCacheManagerTest, AdaptiveRKVFallsBackToScoreWithScoresNoDiversity) {
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    // Feed scores only (no diversity).  Seq 0 front = 0.2, seq 1 front = 8.0
    pasts[0] = 4;
    pasts[1] = 4;
    float scores[8] = {0.1f, 0.1f, 5.f, 5.f, 4.f, 4.f, 4.f, 4.f};
    mgr->update_attention_scores(NODE, scores, 8, pasts, 2);

    // diversity chain: no diversity data -> candidates empty -> fallback to steal_block_by_score
    // SCORE picks seq 0 front (0.2) over seq 1 front (8.0)
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));  // seq 0 evicted by SCORE fallback
    EXPECT_TRUE(mgr->resolve_token(NODE, 0, 2, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 0, addr));
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 4, addr));
}

// Diversity matrix is cleared after eviction, next eviction must use fresh data or fallback
TEST(PagedCacheManagerTest, AdaptiveRKVDiversityClearedAfterEviction) {
    CacheLayout layout{4, 1, 2, 4};
    std::vector<float> kd, vd;
    auto mgr = make_manager(layout, EvictionPolicy::ADAPTIVE_RKV, kd, vd, 2);

    std::int32_t pasts[2] = {0, 0};
    mgr->begin_step(NODE, pasts, 2);

    std::vector<float> krow(layout.kv_heads * layout.head_size, 0.f);
    std::vector<float> vrow(layout.kv_heads * layout.head_size, 0.f);
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 0, t, krow.data(), vrow.data());
    for (int t = 0; t < 4; t++)
        mgr->write_token_kv<float>(NODE, 1, t, krow.data(), vrow.data());

    std::int32_t pasts_s[2] = {4, 4};
    float scores[8] = {0.1f, 0.1f, 5.f, 5.f, 2.f, 2.f, 2.f, 2.f};
    mgr->update_attention_scores(NODE, scores, 8, pasts_s, 2);

    float div0[8] = {0.1f, 0.1f, 0.1f, 0.1f, 5.f, 5.f, 5.f, 5.f};
    float div1[8] = {3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f, 3.f};
    mgr->update_diversity_scores(NODE, 0, div0, 2, 4, 0);
    mgr->update_diversity_scores(NODE, 1, div1, 2, 4, 0);

    // First eviction: seq 0 front evicted (low diversity)
    mgr->write_token_kv<float>(NODE, 1, 4, krow.data(), vrow.data());

    PagedCacheManager::TokenAddress addr;
    EXPECT_FALSE(mgr->resolve_token(NODE, 0, 0, addr));

    // Diversity was cleared for the evicted seq.  Now seq 0 has 1 block left, seq 1 has 3.
    // Force another eviction without feeding new diversity.
    // Since diversity_matrix was cleared, fallback to SCORE should kick in.
    // Need to also update scores for the new arrangement
    pasts_s[0] = 4;  // logical length still 4
    pasts_s[1] = 5;
    // seq 0's remaining block (tokens 2,3) still has score 10.0 from before
    // seq 1 has 3 blocks with scores.  No new diversity -> SCORE fallback
    mgr->write_token_kv<float>(NODE, 1, 5, krow.data(), vrow.data());

    // Should not crash; some eviction should occur
    EXPECT_TRUE(mgr->resolve_token(NODE, 1, 5, addr));
}
