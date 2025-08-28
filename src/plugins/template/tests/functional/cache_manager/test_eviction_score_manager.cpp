#include <gtest/gtest.h>
#include "openvino/runtime/tensor.hpp"
#include "cache_eviction.hpp"

using namespace ov::cache;

TEST(CacheManagerEviction, ScoreRegisterRemoveSmoke) {
    const size_t block = 16;
    const size_t layers = 2;
    const size_t pool = 8;
    EvictionScoreManager mgr(block, layers, pool, AggregationMode::SUM, 0);

    AttentionScoresForEachDecoderLayer scores;
    {
        ov::Tensor t0(ov::element::f32, {32});
        auto* p0 = t0.data<float>();
        for (size_t i=0;i<32;++i) p0[i] = static_cast<float>(i+1);
        scores.push_back(t0);

        ov::Tensor t1(ov::element::f32, {32});
        auto* p1 = t1.data<float>();
        for (size_t i=0;i<32;++i) p1[i] = static_cast<float>(32 - i);
        scores.push_back(t1);
    }
    std::set<size_t> skipped{};
    mgr.register_new_token_scores(scores, skipped);

    ASSERT_EQ(mgr.get_scores().size(), layers);
    EXPECT_EQ(mgr.get_scores()[0].size(), 32);
    EXPECT_EQ(mgr.get_scores()[1].size(), 32);

    mgr.remove_scores({0}, /*layer=*/0);
    EXPECT_EQ(mgr.get_scores()[0].size(), 16);
}

TEST(CacheManagerEviction, MaxCacheCeilProperty) {
    CacheEvictionConfig cfg(/*start*/32, /*recent*/64, /*max*/256, AggregationMode::SUM, false);
    CacheEvictionAlgorithm algo(cfg, /*block_size*/16, /*num_layers*/2, /*pool*/8);
    EXPECT_EQ(algo.get_max_cache_size_after_eviction(), 256 + 16 - 1);
}
