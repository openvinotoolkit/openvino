// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Request-level tests for the LongRoPE short/long mode state machine.
//
// llm_longrope_kv_test.cpp proves the coefficient math against an independent
// reference; what is left, and what these tests cover, is everything around it: when a
// crossing is detected, that the rewrite runs after the KV of the previous stage has
// been migrated into the request being rewritten, which rows it touches, and that a
// transition which cannot be completed leaves the recorded mode describing the cache
// as it really is.
//
// The observation trick is a second, "control" model that is identical except for a
// context limit no position in the test can reach. Both models are driven with the same
// inputs through fake sub-requests whose outputs depend only on the port name, so their
// KV caches agree byte for byte up to the re-rotation - which is then the only thing the
// comparison can be measuring.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "kv_cache_block_manager_test_access.hpp"
#include "llm_block_kv_test_harness.hpp"
#include "llm_longrope_kv.hpp"

namespace ov::test::npuw {

struct LLMLongRopeModeTestAccess {
    using Request = ov::npuw::LLMInferRequest;

    static bool long_mode(Request& req) {
        return req.m_longrope_long_mode;
    }

    static const std::vector<std::string>& past_names(Request& req) {
        return req.m_kvcache_past_names;
    }

    static ov::SoPtr<ov::ITensor> generate_past(Request& req, const std::string& name) {
        return req.m_kvcache_request->get_tensor(req.m_kvcache_in_ports.at(name));
    }

    static uint32_t stored_tokens(Request& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc.num_stored_tokens;
    }

    static uint32_t seq_dim(Request& req) {
        return req.m_npuw_llm_compiled_model->m_kvcache_desc.dim;
    }

    static ov::npuw::patterns::pre_compute::LongRopeCosSin& tables(Request& req) {
        return req.m_npuw_llm_compiled_model->m_longrope_tables;
    }

    static bool has_long(const ov::npuw::LLMCompiledModel& compiled) {
        return compiled.m_longrope_tables.has_long;
    }

    // Drives the transition directly, which is the only way to reach the rejection
    // paths - a real request always hands over its own, well-formed port map.
    static void sync_mode(Request& req,
                          const ov::npuw::LLMInferRequest::PortsMap& in_ports,
                          const ov::SoPtr<ov::ITensor>& position_ids,
                          uint32_t num_cached_tokens) {
        req.sync_longrope_mode(req.m_kvcache_request, in_ports, position_ids, num_cached_tokens);
    }

    static const ov::npuw::LLMInferRequest::PortsMap& generate_in_ports(Request& req) {
        return req.m_kvcache_in_ports;
    }
};

}  // namespace ov::test::npuw

namespace {

using ov::test::npuw::build_longrope_llm_test_model;
using ov::test::npuw::KVCacheBlockManagerTestAccess;
using ov::test::npuw::LLMBlockKvTestAccess;
using ov::test::npuw::LLMKvTestPipeline;
using ov::test::npuw::LLMLongRopeModeTestAccess;
using ov::test::npuw::make_i64;
using ov::test::npuw::make_patterned_factory;
using ov::test::npuw::NullPlugin;
using ov::test::npuw::to_floats;

// A position no prompt in this file reaches, so the model never leaves the short
// mode and its KV cache is never re-rotated.
constexpr int64_t kUnreachableLimit = 4096;
// Crossed by generated token 16 of a 16-token prompt.
constexpr int64_t kCrossingLimit = 16;

// The shared pipeline over a LongRoPE model with the given context limit.
class LongRopePipeline : public LLMKvTestPipeline {
public:
    explicit LongRopePipeline(int64_t context_limit, const ov::AnyMap& extra_props = {})
        : LLMKvTestPipeline(build_longrope_llm_test_model(context_limit), with_cache_rope(extra_props)) {}

    // The past keys of every layer, flattened, in m_kvcache_past_names order.
    std::vector<std::vector<float>> generate_keys() {
        std::vector<std::vector<float>> out;
        for (const auto& name : LLMLongRopeModeTestAccess::past_names(request())) {
            if (ov::npuw::util::isPastKeyParam(name)) {
                out.push_back(to_floats(LLMLongRopeModeTestAccess::generate_past(request(), name)));
            }
        }
        return out;
    }

    ov::Shape generate_key_shape() {
        for (const auto& name : LLMLongRopeModeTestAccess::past_names(request())) {
            if (ov::npuw::util::isPastKeyParam(name)) {
                return LLMLongRopeModeTestAccess::generate_past(request(), name)->get_shape();
            }
        }
        return {};
    }

private:
    static ov::AnyMap with_cache_rope(const ov::AnyMap& extra_props) {
        ov::AnyMap props{{"NPUW_LLM_CACHE_ROPE", "YES"}};
        for (const auto& [key, value] : extra_props) {
            props[key] = value;
        }
        return props;
    }
};

// Applies the same turn the request is expected to have applied, to a copy of the
// control run's keys. The coefficient math itself is validated independently in
// llm_longrope_kv_test.cpp; here it is the reference for placement and row range.
std::vector<std::vector<float>> expected_turn(std::vector<std::vector<float>> keys,
                                              const ov::Shape& shape,
                                              uint32_t seq_dim,
                                              ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
                                              uint32_t num_tokens,
                                              bool to_long) {
    const auto delta = ov::npuw::longrope::make_mode_delta(tables, 0, num_tokens, to_long);
    for (auto& layer : keys) {
        auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f32, shape, layer.data()));
        ov::npuw::longrope::rerotate_keys(tensor, seq_dim, num_tokens, delta);
    }
    return keys;
}

void expect_keys_near(const std::vector<std::vector<float>>& actual,
                      const std::vector<std::vector<float>>& expected,
                      float tol) {
    ASSERT_EQ(actual.size(), expected.size());
    ASSERT_FALSE(actual.empty());
    for (size_t layer = 0; layer < actual.size(); ++layer) {
        ASSERT_EQ(actual[layer].size(), expected[layer].size());
        for (size_t i = 0; i < actual[layer].size(); ++i) {
            ASSERT_NEAR(actual[layer][i], expected[layer][i], tol) << "layer " << layer << " element " << i;
        }
    }
}

// The models this file relies on must actually reach the LongRoPE path, otherwise every
// comparison below would pass on two untransformed graphs.
TEST(LLMLongRopeMode, ModelsReachTheLongRopePath) {
    LongRopePipeline crossing(kCrossingLimit);
    LongRopePipeline control(kUnreachableLimit);

    EXPECT_TRUE(LLMLongRopeModeTestAccess::has_long(crossing.compiled()));
    // Its limit sits past the whole context, so it can only ever run short.
    EXPECT_FALSE(LLMLongRopeModeTestAccess::has_long(control.compiled()));
    EXPECT_TRUE(LLMLongRopeModeTestAccess::tables(crossing.request()).is_valid());
}

// The first generate step past the limit turns exactly the cached rows, and it does so
// after the prefill KV has been migrated into the generate request - had it run before,
// the migration would have overwritten the turned keys with the untouched ones and the
// two runs would agree.
TEST(LLMLongRopeMode, CrossingTurnsTheMigratedCache) {
    constexpr size_t kPrompt = 16;

    LongRopePipeline control(kUnreachableLimit);
    control.prefill(kPrompt, 0);
    control.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    EXPECT_FALSE(LLMLongRopeModeTestAccess::long_mode(control.request()));
    const auto control_keys = control.generate_keys();

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    EXPECT_FALSE(LLMLongRopeModeTestAccess::long_mode(crossing.request()));
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    EXPECT_TRUE(LLMLongRopeModeTestAccess::long_mode(crossing.request()));

    const auto expected = expected_turn(control_keys,
                                        crossing.generate_key_shape(),
                                        LLMLongRopeModeTestAccess::seq_dim(crossing.request()),
                                        LLMLongRopeModeTestAccess::tables(crossing.request()),
                                        kPrompt,
                                        true);
    expect_keys_near(crossing.generate_keys(), expected, 2e-3f);
    // A turn that did nothing would trivially satisfy the comparison above.
    EXPECT_NE(control_keys, crossing.generate_keys());
}

// A prompt that is already past the limit finds an empty cache: the mode is recorded,
// nothing is rewritten, and the following steps do not cross again.
TEST(LLMLongRopeMode, FreshLongPromptRecordsTheModeWithoutTouchingTheCache) {
    constexpr size_t kPrompt = 20;  // positions 0..19, so the prompt itself is long

    LongRopePipeline control(kUnreachableLimit);
    control.prefill(kPrompt, 0);
    control.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    const auto control_keys = control.generate_keys();

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    EXPECT_TRUE(LLMLongRopeModeTestAccess::long_mode(crossing.request()));
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    EXPECT_TRUE(LLMLongRopeModeTestAccess::long_mode(crossing.request()));

    expect_keys_near(crossing.generate_keys(), control_keys, 0.0f);
}

// Trimming the cache back below the limit - what speculative decoding does when its
// proposal is rejected - flips the mode back and returns the keys to where they were.
TEST(LLMLongRopeMode, TrimmingBackBelowTheLimitTurnsTheCacheBack) {
    constexpr size_t kPrompt = 16;
    constexpr uint32_t kTrimTo = 10;

    LongRopePipeline control(kUnreachableLimit);
    control.prefill(kPrompt, 0);
    control.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    const auto control_keys = control.generate_keys();

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));
    ASSERT_TRUE(LLMLongRopeModeTestAccess::long_mode(crossing.request()));

    crossing.generate_step(kTrimTo, kTrimTo);
    EXPECT_FALSE(LLMLongRopeModeTestAccess::long_mode(crossing.request()));
    EXPECT_EQ(LLMLongRopeModeTestAccess::stored_tokens(crossing.request()), kTrimTo + 1u);

    // Only the rows that survived the trim were turned twice; compare just those.
    const auto keys = crossing.generate_keys();
    const auto shape = crossing.generate_key_shape();
    const size_t head_dim = shape.back();
    const size_t row_stride = head_dim;  // [batch, kv_heads, seq, head_dim]
    const size_t seq_len = shape[LLMLongRopeModeTestAccess::seq_dim(crossing.request())];
    ASSERT_EQ(keys.size(), control_keys.size());
    for (size_t layer = 0; layer < keys.size(); ++layer) {
        for (size_t plane = 0; plane * seq_len * row_stride < keys[layer].size(); ++plane) {
            for (size_t token = 0; token < kTrimTo; ++token) {
                for (size_t ch = 0; ch < head_dim; ++ch) {
                    const size_t idx = (plane * seq_len + token) * row_stride + ch;
                    ASSERT_NEAR(keys[layer][idx], control_keys[layer][idx], 4e-3f)
                        << "layer " << layer << " token " << token << " channel " << ch;
                }
            }
        }
    }
}

// A transition that cannot be carried out must fail before it writes anything and must
// leave the recorded mode describing the cache as it really is - otherwise the next
// call would see no flip and run against keys from the other rotation frame.
TEST(LLMLongRopeMode, RejectedTransitionLeavesTheCacheAndTheModeAlone) {
    constexpr size_t kPrompt = 8;
    constexpr uint32_t kCached = kPrompt + 1;  // the prompt plus the token just generated

    LongRopePipeline crossing(kCrossingLimit);
    crossing.prefill(kPrompt, 0);
    crossing.generate_step(kPrompt, static_cast<int64_t>(kPrompt));  // position 8, still short
    auto& req = crossing.request();
    ASSERT_FALSE(LLMLongRopeModeTestAccess::long_mode(req));
    ASSERT_EQ(LLMLongRopeModeTestAccess::stored_tokens(req), kCached);
    const auto before = crossing.generate_keys();

    // A port map without the past-key inputs: the same shape of failure as a KV layout
    // the rewrite does not support.
    ov::npuw::LLMInferRequest::PortsMap broken;
    for (const auto& [name, port] : LLMLongRopeModeTestAccess::generate_in_ports(req)) {
        if (!ov::npuw::util::isPastKeyParam(name)) {
            broken.emplace(name, port);
        }
    }

    // A position past the limit, so the transition to long is the one being attempted.
    auto long_position = make_i64({1, 1}, kCrossingLimit);
    const auto position_ids = ov::get_tensor_impl(long_position);

    EXPECT_THROW(LLMLongRopeModeTestAccess::sync_mode(req, broken, position_ids, kCached), ov::Exception);
    EXPECT_FALSE(LLMLongRopeModeTestAccess::long_mode(req));
    expect_keys_near(crossing.generate_keys(), before, 0.0f);

    // The flip is still pending, so a well-formed retry performs it in full.
    LLMLongRopeModeTestAccess::sync_mode(req, LLMLongRopeModeTestAccess::generate_in_ports(req), position_ids, kCached);
    EXPECT_TRUE(LLMLongRopeModeTestAccess::long_mode(req));
    EXPECT_NE(crossing.generate_keys(), before);
}

// A block-based cache holds the same keys in a pool of fixed-size blocks, so the turn has
// to find them there instead of in one tensor per layer. Same observation trick as above,
// but read out of the pool: a request port is either a view of a block, a copy of one, or
// a dummy tensor shared by every port that currently backs no token.
class LongRopeBlockCache : public ov::test::npuw::LLMBlockKvTest {
protected:
    struct PooledKeys {
        ov::Shape block_shape;
        std::vector<uint32_t> layer;
        std::vector<uint32_t> first_token;  // where each block's row 0 sits in the cache
        std::vector<uint32_t> live_tokens;
        std::vector<std::vector<float>> data;
    };

    // Every live key block of every layer, layer-major and in block order.
    static PooledKeys pooled_keys(ov::npuw::LLMInferRequest& req, uint32_t num_cached) {
        auto& strategy = LLMBlockKvTestAccess::block_strategy(req);
        const auto& managers = LLMBlockKvTestAccess::block_managers(strategy);
        const uint32_t block_size = LLMBlockKvTestAccess::block_size(strategy);

        std::vector<uint32_t> layers;
        for (const auto& [layer_idx, unused] : managers) {
            layers.push_back(layer_idx);
        }
        std::sort(layers.begin(), layers.end());

        PooledKeys out;
        for (const uint32_t layer_idx : layers) {
            auto* manager = managers.at(layer_idx).key_manager.get();
            EXPECT_NE(manager, nullptr);
            const auto allocated = manager->get_allocated_blocks();
            for (uint32_t first = 0u; first < num_cached; first += block_size) {
                EXPECT_LT(first / block_size, allocated.size());
                const auto tensor = manager->get_block_tensor(allocated[first / block_size]);
                out.block_shape = tensor->get_shape();
                out.layer.push_back(layer_idx);
                out.first_token.push_back(first);
                out.live_tokens.push_back(std::min(num_cached - first, block_size));
                out.data.push_back(to_floats(tensor));
            }
        }
        return out;
    }

    // The turn the request is expected to have applied, block by block: each block reads
    // the delta rows its own positions land on.
    static PooledKeys expected_turn(PooledKeys keys,
                                    uint32_t seq_dim,
                                    ov::npuw::patterns::pre_compute::LongRopeCosSin& tables,
                                    uint32_t num_cached,
                                    bool to_long) {
        const auto delta = ov::npuw::longrope::make_mode_delta(tables, 0, num_cached, to_long);
        for (size_t i = 0; i < keys.data.size(); ++i) {
            auto tensor = ov::get_tensor_impl(ov::Tensor(ov::element::f32, keys.block_shape, keys.data[i].data()));
            const auto layout = ov::npuw::longrope::check_key_tensor(tensor, seq_dim, keys.live_tokens[i], delta);
            ov::npuw::longrope::rerotate_keys(layout, keys.live_tokens[i], delta, keys.first_token[i]);
        }
        return keys;
    }

    static void run(LongRopePipeline& pipeline, size_t prompt, int64_t last_position) {
        pipeline.prefill(prompt, 0);
        for (int64_t position = static_cast<int64_t>(prompt); position <= last_position; ++position) {
            pipeline.generate_step(static_cast<size_t>(position), position);
        }
    }

    // Flat element indices of the rows a block actually holds a key in. The rows past
    // them are never written, so two runs need not agree there.
    static std::vector<size_t> live_elements(const ov::Shape& shape, uint32_t seq_dim, uint32_t live_tokens) {
        size_t seq_stride = 1;
        for (size_t d = seq_dim + 1; d < shape.size(); ++d) {
            seq_stride *= shape[d];
        }
        const size_t seq_len = shape[seq_dim];
        const size_t outer = ov::shape_size(shape) / (seq_len * seq_stride);
        std::vector<size_t> indices;
        indices.reserve(outer * live_tokens * seq_stride);
        for (size_t o = 0; o < outer; ++o) {
            for (size_t t = 0; t < live_tokens; ++t) {
                for (size_t k = 0; k < seq_stride; ++k) {
                    indices.push_back((o * seq_len + t) * seq_stride + k);
                }
            }
        }
        return indices;
    }
};

// The crossing turns every cached key held in the pool, across both the numbered blocks
// and the tail one, and leaves the tail port - which holds a copy of its block rather than
// a view of it - agreeing with the block it was copied from.
TEST_F(LongRopeBlockCache, CrossingTurnsEveryBlockOfThePool) {
    constexpr size_t kPrompt = 64;     // two full blocks
    constexpr int64_t kLimit = 100;    // crossed by the generate step at position 100
    constexpr uint32_t kCached = 100;  // rows in the cache when that step re-rotates, three
                                       // full blocks plus four rows of the tail one

    LongRopePipeline control(kUnreachableLimit, block_props());
    run(control, kPrompt, kLimit);
    ASSERT_FALSE(LLMLongRopeModeTestAccess::long_mode(control.request()));
    const auto control_keys = pooled_keys(control.request(), kCached);
    // The prompt must have reached past the numbered blocks, or the tail path goes untested.
    ASSERT_GT(control_keys.first_token.back(), 0u);

    LongRopePipeline crossing(kLimit, block_props());
    run(crossing, kPrompt, kLimit);
    EXPECT_TRUE(LLMLongRopeModeTestAccess::long_mode(crossing.request()));

    const auto expected = expected_turn(control_keys,
                                        LLMLongRopeModeTestAccess::seq_dim(crossing.request()),
                                        LLMLongRopeModeTestAccess::tables(crossing.request()),
                                        kCached,
                                        true);
    const auto actual = pooled_keys(crossing.request(), kCached);
    const uint32_t seq_dim = LLMLongRopeModeTestAccess::seq_dim(crossing.request());
    ASSERT_EQ(actual.data.size(), expected.data.size());
    bool any_difference = false;
    for (size_t i = 0; i < actual.data.size(); ++i) {
        ASSERT_EQ(actual.data[i].size(), expected.data[i].size());
        for (const size_t e : live_elements(actual.block_shape, seq_dim, actual.live_tokens[i])) {
            ASSERT_NEAR(actual.data[i][e], expected.data[i][e], 2e-3f) << "block " << i << " element " << e;
            any_difference = any_difference || actual.data[i][e] != control_keys.data[i][e];
        }
    }
    // A turn that did nothing would trivially satisfy the comparison above.
    EXPECT_TRUE(any_difference);

    // The tail port is a copy, so an in-place turn of the pool alone would leave it stale.
    size_t tails_checked = 0;
    for (const auto& entry : LLMLongRopeModeTestAccess::generate_in_ports(crossing.request())) {
        const auto& name = entry.first;
        if (!ov::npuw::util::isPastKeyParam(name) || name.find("block_tail") == std::string::npos) {
            continue;
        }
        const uint32_t layer = static_cast<uint32_t>(std::stoi(name.substr(name.find('.') + 1)));
        // The tail port of a layer holds that layer's last block, the one whose rows sit
        // past the numbered ports.
        size_t block = actual.data.size();
        for (size_t i = 0; i < actual.data.size(); ++i) {
            if (actual.layer[i] != layer) {
                continue;
            }
            if (block == actual.data.size() || actual.first_token[i] > actual.first_token[block]) {
                block = i;
            }
        }
        ASSERT_LT(block, actual.data.size()) << "no pooled block for " << name;

        const auto tail_tensor = LLMLongRopeModeTestAccess::generate_past(crossing.request(), name);
        const auto tail = to_floats(tail_tensor);
        // The tail is a shorter tensor than a block, so each side is indexed through its
        // own shape.
        const auto tail_idx = live_elements(tail_tensor->get_shape(), seq_dim, actual.live_tokens[block]);
        const auto block_idx = live_elements(actual.block_shape, seq_dim, actual.live_tokens[block]);
        ASSERT_EQ(tail_idx.size(), block_idx.size());
        for (size_t i = 0; i < tail_idx.size(); ++i) {
            ASSERT_FLOAT_EQ(tail[tail_idx[i]], actual.data[block][block_idx[i]]) << "tail " << name << " row " << i;
        }
        ++tails_checked;
    }
    EXPECT_GT(tails_checked, 0u);
}

// Prefix caching on top of the block KV cache; the tests that do not involve LongRoPE are
// in llm_prefix_caching_block_kv_test.cpp. A cached block keeps references to the pool's
// own block tensors instead of copies, so a re-rotation, which rewrites keys in place, must
// not reach them: the pool turns a private copy instead and rebinds the request to it.
TEST_F(LongRopeBlockCache, LongRopeTurnLeavesTheCachedBlockUntouched) {
    constexpr size_t kPrompt = kTwoBlocks;
    constexpr int64_t kLimit = 100;

    LongRopePipeline pipeline(kLimit, block_props());
    pipeline.prefill(kPrompt, 0);
    const auto block = store_block(pipeline.request(), 0u);
    ASSERT_NE(block, nullptr);
    const auto& refs = as_pool_block(block).get_layer_refs();

    std::map<uint32_t, std::vector<float>> cached_before;
    for (const auto& [layer_idx, ref] : refs) {
        cached_before[layer_idx] = to_floats(ref.key_tensor);
    }

    for (int64_t position = static_cast<int64_t>(kPrompt); position <= kLimit; ++position) {
        pipeline.generate_step(static_cast<size_t>(position), position);
    }
    ASSERT_TRUE(LLMLongRopeModeTestAccess::long_mode(pipeline.request()));

    for (const auto& [layer_idx, ref] : refs) {
        const auto& layer = managers(pipeline.request()).at(layer_idx);
        const auto key_id = layer.key_manager->get_allocated_blocks().at(0);
        const auto pooled = layer.key_manager->get_block_tensor(key_id);

        EXPECT_EQ(to_floats(ref.key_tensor), cached_before.at(layer_idx)) << "cached keys changed, layer " << layer_idx;
        EXPECT_NE(pooled->data(), ref.key_tensor->data()) << "layer " << layer_idx;
        EXPECT_NE(to_floats(pooled), cached_before.at(layer_idx)) << "pool keys were not turned, layer " << layer_idx;
        EXPECT_FALSE(KVCacheBlockManagerTestAccess::is_shared(*layer.key_manager, key_id));
    }
}

// A KV layout the rewrite cannot turn is refused when the model is compiled, not at the
// crossing token: by then there is no correct thing left to do.
class LongRopeCompileRejection : public ::testing::Test {
protected:
    static ov::AnyMap base_props() {
        return {{"NPUW_LLM", "YES"},
                {"NPUW_DEVICES", "CPU"},
                {"NPUW_LLM_MAX_PROMPT_LEN", "64"},
                {"NPUW_LLM_MIN_RESPONSE_LEN", "32"},
                {"NPUW_LLM_CACHE_ROPE", "YES"}};
    }

    void compile(int64_t context_limit, const ov::AnyMap& extra) {
        auto props = base_props();
        for (const auto& [key, value] : extra) {
            props[key] = value;
        }
        ov::npuw::LLMCompiledModel(build_longrope_llm_test_model(context_limit),
                                   std::make_shared<NullPlugin>(),
                                   props,
                                   make_patterned_factory());
    }

    void expect_longrope_rejection(int64_t context_limit, const ov::AnyMap& extra) {
        try {
            compile(context_limit, extra);
            FAIL() << "compilation was expected to be rejected";
        } catch (const ov::Exception& e) {
            EXPECT_NE(std::string(e.what()).find("LongRoPE mode changes"), std::string::npos) << e.what();
        }
    }
};

TEST_F(LongRopeCompileRejection, QuantizedKvCache) {
    expect_longrope_rejection(kCrossingLimit, {{ov::hint::kv_cache_precision.name(), ov::element::i8}});
}

}  // anonymous namespace
