// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "custom/subgraph_tests/src/classes/concat_sdp.hpp"

namespace ov {
namespace test {
namespace {

const std::vector<std::vector<InputShape>> inputShapes = {
    // greedy search
    {
        {{1, 8, -1, 64}, {{1, 8, 10, 64}, {1, 8, 1, 64}, {1, 8, 1, 64}, {1, 8, 20, 64}, {1, 8, 1, 64}}},
        {{1, 8, -1, 64}, {{1, 8, 0, 64}, {1, 8, 10, 64}, {1, 8, 11, 64}, {1, 8, 12, 64}, {1, 8, 32, 64}}},
    },
    // beam search
    {
        {{-1, 8, -1, 64}, {{4, 8, 10, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}, {4, 8, 1, 64}}},
        {{-1, 8, -1, 64}, {{4, 8, 0, 64}, {4, 8, 10, 64}, {4, 8, 11, 64}, {4, 8, 12, 64}, {4, 8, 13, 64}}},
    },
    // big batch to check cvt_copy fast-path inside mha_single_token_kernel
    {
        {{-1, 8, -1, 64}, {{129, 8, 10, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}, {129, 8, 1, 64}}},
        {{-1, 8, -1, 64}, {{129, 8, 0, 64}, {129, 8, 10, 64}, {129, 8, 11, 64}, {129, 8, 12, 64}, {129, 8, 13, 64}}},
    },
};

// K×V codec matrix: {none, u8, u4 scalar, TBQ4 (u4+TURBO)}.
enum class KvCodec { NONE, U8, U4, TBQ4 };
static ov::AnyMap kv_cfg(KvCodec k, KvCodec v) {
    ov::AnyMap m;
    auto set = [&](KvCodec c, const char* prec_key, const char* alg_key) {
        switch (c) {
            case KvCodec::NONE: break;
            case KvCodec::U8:   m[prec_key] = "u8"; break;
            case KvCodec::U4:   m[prec_key] = "u4"; break;
            case KvCodec::TBQ4: m[prec_key] = "u4"; m[alg_key] = "TURBO"; break;
        }
    };
    set(k, "KEY_CACHE_PRECISION", "KEY_CACHE_QUANT_ALG");
    set(v, "VALUE_CACHE_PRECISION", "VALUE_CACHE_QUANT_ALG");
    return m;
}
static std::vector<ov::AnyMap> all_kv_cfgs() {
    std::vector<ov::AnyMap> out;
    for (auto k : {KvCodec::NONE, KvCodec::U8, KvCodec::U4, KvCodec::TBQ4}) {
        for (auto v : {KvCodec::NONE, KvCodec::U8, KvCodec::U4, KvCodec::TBQ4}) {
            out.push_back(kv_cfg(k, v));
        }
    }
    return out;
}

INSTANTIATE_TEST_SUITE_P(smoke_ConcatSDPTest,
                         ConcatSDPTest,
                         ::testing::Combine(::testing::Values(ElementType::f32, ElementType::bf16, ElementType::f16),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(all_kv_cfgs()),
                                            ::testing::Values(true, false),
                                            ::testing::Values<int64_t>(8),
                                            ::testing::Values<int64_t>(8, 2, 1)),
                         ConcatSDPTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
