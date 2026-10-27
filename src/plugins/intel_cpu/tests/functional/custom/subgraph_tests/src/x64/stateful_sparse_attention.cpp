// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Stateful SparseAttention: ReadValue -> Gather(beam_idx) -> Concat -> SparseAttention (+ Assign) is fused into
// SparseAttentionWithKVCache by the CPU plugin, which keeps the K/V cache in its KV cache states. The test runs the
// stateful model step by step and compares each step with a stateless SparseAttention model fed with the K/V history
// accumulated on the host.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"

namespace ov {
namespace test {

namespace {

constexpr size_t kHeads = 4;
constexpr size_t kKvHeads = 2;
constexpr size_t kTopK = 3;

struct Step {
    size_t len;         // L of the step
    bool reorder_beam;  // gather the cache rows with the reversed batch order before appending
};

// {inference precision, kv cache precision, key cache quantized by channel, batch, head size, key block size,
// number of selection heads}
using StatefulSparseAttentionParams =
    std::tuple<ov::element::Type, ov::element::Type, bool, size_t, size_t, size_t, size_t>;

// values in [-1, 1] with a step of 0.01, exactly representable in bf16 after the rounding done by the helper
ov::Tensor random_tensor(ov::element::Type prec, const ov::Shape& shape, int seed) {
    return ov::test::utils::create_and_fill_tensor(prec, shape, ov::test::utils::InputGenerateData(-1, 2, 100, seed));
}

void set_name(const std::shared_ptr<ov::op::v0::Parameter>& param, const std::string& name) {
    param->set_friendly_name(name);
    param->output(0).set_names({name});
}

// a tensor of the given floating point precision (f32 / bf16) with the given values
ov::Tensor to_tensor(const std::vector<float>& data,
                     const ov::Shape& shape,
                     ov::element::Type prec = ov::element::f32) {
    ov::Tensor t(prec, shape);
    if (prec == ov::element::bf16) {
        std::transform(data.begin(), data.end(), t.data<ov::bfloat16>(), [](float x) {
            return ov::bfloat16(x);
        });
    } else {
        std::copy(data.begin(), data.end(), t.data<float>());
    }
    return t;
}

std::vector<float> to_f32(const ov::Tensor& t) {
    std::vector<float> out(t.get_size());
    if (t.get_element_type() == ov::element::bf16) {
        std::transform(t.data<const ov::bfloat16>(),
                       t.data<const ov::bfloat16>() + t.get_size(),
                       out.begin(),
                       [](ov::bfloat16 x) {
                           return static_cast<float>(x);
                       });
    } else {
        std::copy_n(t.data<const float>(), t.get_size(), out.begin());
    }
    return out;
}

ov::Tensor to_tensor(const std::vector<int32_t>& data, const ov::Shape& shape) {
    ov::Tensor t(ov::element::i32, shape);
    std::copy(data.begin(), data.end(), t.data<int32_t>());
    return t;
}

// Host-side [N, Hkv, S, E] cache.
struct HostCache {
    size_t N = 0;
    size_t E = 0;
    size_t S = 0;
    std::vector<float> data;  // [N, Hkv, S, E]

    HostCache(size_t n, size_t e) : N(n), E(e) {}

    void reorder(const std::vector<int32_t>& beam) {
        std::vector<float> out(data.size());
        const size_t per_batch = kKvHeads * S * E;
        for (size_t b = 0; b < N; b++) {
            std::copy_n(data.begin() + beam[b] * per_batch, per_batch, out.begin() + b * per_batch);
        }
        data = std::move(out);
    }

    // append cur [N, Hkv, L, E]
    void append(const std::vector<float>& cur, size_t L) {
        std::vector<float> out(N * kKvHeads * (S + L) * E);
        for (size_t b = 0; b < N; b++) {
            for (size_t h = 0; h < kKvHeads; h++) {
                auto* dst = out.data() + ((b * kKvHeads + h) * (S + L)) * E;
                std::copy_n(data.data() + ((b * kKvHeads + h) * S) * E, S * E, dst);
                std::copy_n(cur.data() + ((b * kKvHeads + h) * L) * E, L * E, dst + S * E);
            }
        }
        data = std::move(out);
        S += L;
    }

    [[nodiscard]] ov::Tensor tensor(ov::element::Type prec) const {
        return to_tensor(data, ov::Shape{N, kKvHeads, S, E}, prec);
    }
};

}  // namespace

class StatefulSparseAttentionTest : public ::testing::TestWithParam<StatefulSparseAttentionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<StatefulSparseAttentionParams>& obj) {
        const auto& [prec, cache_prec, by_channel, batch, head_size, block_size, sel_heads] = obj.param;
        std::ostringstream result;
        result << "Prec=" << prec.get_type_name() << "_Cache=" << cache_prec.get_type_name()
               << (by_channel ? "_KeyByChannel" : "") << "_N=" << batch << "_E=" << head_size << "_Bk=" << block_size
               << "_Hs=" << sel_heads;
        return result.str();
    }

protected:
    static std::shared_ptr<ov::Model> make_stateful_model(ov::element::Type data_prec,
                                                          size_t head_size,
                                                          size_t block_size,
                                                          size_t sel_heads) {
        auto q = std::make_shared<ov::op::v0::Parameter>(data_prec, ov::PartialShape{-1, kHeads, -1, head_size});
        auto k = std::make_shared<ov::op::v0::Parameter>(data_prec, ov::PartialShape{-1, kKvHeads, -1, head_size});
        auto v = std::make_shared<ov::op::v0::Parameter>(data_prec, ov::PartialShape{-1, kKvHeads, -1, head_size});
        auto sel_idx =
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, sel_heads, -1, kTopK});
        auto sel_cnt = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, sel_heads, -1});
        auto beam_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
        auto past_init =
            std::make_shared<ov::op::v0::Parameter>(data_prec, ov::PartialShape{-1, kKvHeads, -1, head_size});
        set_name(q, "q");
        set_name(k, "k");
        set_name(v, "v");
        set_name(sel_idx, "block_indices");
        set_name(sel_cnt, "block_count");
        set_name(beam_idx, "beam_idx");
        set_name(past_init, "past_init");

        auto variable_k = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{-1, kKvHeads, -1, head_size}, data_prec, "pastk"});
        auto variable_v = std::make_shared<ov::op::util::Variable>(
            ov::op::util::VariableInfo{ov::PartialShape{-1, kKvHeads, -1, head_size}, data_prec, "pastv"});
        auto past_k = std::make_shared<ov::op::v6::ReadValue>(past_init, variable_k);
        auto past_v = std::make_shared<ov::op::v6::ReadValue>(past_init, variable_v);

        auto axis = ov::op::v0::Constant::create(ov::element::i32, {1}, {0});
        auto gather_k = std::make_shared<ov::op::v8::Gather>(past_k, beam_idx, axis);
        auto gather_v = std::make_shared<ov::op::v8::Gather>(past_v, beam_idx, axis);
        auto concat_k = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather_k, k}, 2);
        auto concat_v = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{gather_v, v}, 2);

        auto sparse = std::make_shared<ov::op::v17::SparseAttention>(q,
                                                                     concat_k,
                                                                     concat_v,
                                                                     sel_idx,
                                                                     sel_cnt,
                                                                     static_cast<int64_t>(block_size),
                                                                     /*query_block_size=*/1,
                                                                     /*causal=*/true);
        auto assign_k = std::make_shared<ov::op::v6::Assign>(concat_k, variable_k);
        auto assign_v = std::make_shared<ov::op::v6::Assign>(concat_v, variable_v);
        return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sparse)},
                                           ov::SinkVector{assign_k, assign_v},
                                           ov::ParameterVector{q, k, v, sel_idx, sel_cnt, beam_idx, past_init},
                                           "StatefulSparseAttention");
    }

    static std::shared_ptr<ov::Model> make_reference_model(ov::element::Type data_prec,
                                                           size_t head_size,
                                                           size_t block_size,
                                                           size_t sel_heads) {
        auto q = std::make_shared<ov::op::v0::Parameter>(data_prec, ov::PartialShape{-1, kHeads, -1, head_size});
        auto k = std::make_shared<ov::op::v0::Parameter>(data_prec, ov::PartialShape{-1, kKvHeads, -1, head_size});
        auto v = std::make_shared<ov::op::v0::Parameter>(data_prec, ov::PartialShape{-1, kKvHeads, -1, head_size});
        auto sel_idx =
            std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, sel_heads, -1, kTopK});
        auto sel_cnt = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, sel_heads, -1});
        set_name(q, "q");
        set_name(k, "k");
        set_name(v, "v");
        set_name(sel_idx, "block_indices");
        set_name(sel_cnt, "block_count");
        auto sparse = std::make_shared<ov::op::v17::SparseAttention>(q,
                                                                     k,
                                                                     v,
                                                                     sel_idx,
                                                                     sel_cnt,
                                                                     static_cast<int64_t>(block_size),
                                                                     /*query_block_size=*/1,
                                                                     /*causal=*/true);
        return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sparse)},
                                           ov::ParameterVector{q, k, v, sel_idx, sel_cnt},
                                           "SparseAttentionReference");
    }

    // sel_indices / sel_count for queries at the absolute positions [S - L, S): complete blocks only, ascending
    static void make_selection(std::mt19937& gen,
                               size_t N,
                               size_t sel_heads,
                               size_t L,
                               size_t S,
                               size_t block_size,
                               std::vector<int32_t>& indices,
                               std::vector<int32_t>& count) {
        indices.assign(N * sel_heads * L * kTopK, 0);
        count.assign(N * sel_heads * L, 0);
        for (size_t n = 0; n < N; n++) {
            for (size_t hs = 0; hs < sel_heads; hs++) {
                for (size_t l = 0; l < L; l++) {
                    const size_t token_pos = S - L + l;
                    const size_t complete = (token_pos + 1) / block_size;
                    std::vector<int32_t> ids(complete);
                    std::iota(ids.begin(), ids.end(), 0);
                    std::shuffle(ids.begin(), ids.end(), gen);
                    const size_t cnt = std::min(kTopK, complete);
                    ids.resize(cnt);
                    std::sort(ids.begin(), ids.end());
                    const size_t row = (n * sel_heads + hs) * L + l;
                    count[row] = static_cast<int32_t>(cnt);
                    std::copy(ids.begin(), ids.end(), indices.begin() + row * kTopK);
                }
            }
        }
    }

    static void compare(const ov::Tensor& expected,
                        const ov::Tensor& actual,
                        float threshold,
                        const std::string& tag,
                        float min_diff = 0.F) {
        ASSERT_EQ(expected.get_shape(), actual.get_shape()) << tag;
        const auto e = to_f32(expected);
        const auto a = to_f32(actual);
        float max_diff = 0.F;
        for (size_t i = 0; i < e.size(); i++) {
            max_diff = std::max(max_diff, std::abs(e[i] - a[i]));
        }
        EXPECT_LE(max_diff, threshold) << tag << ": max abs diff " << max_diff;
        EXPECT_GE(max_diff, min_diff) << tag << ": max abs diff " << max_diff << " is suspiciously small";
    }

    static void check_fused(const ov::CompiledModel& compiled) {
        bool has_sparse = false;
        for (const auto& op : compiled.get_runtime_model()->get_ops()) {
            const auto& info = op->get_rt_info();
            const auto it = info.find(ov::exec_model_info::LAYER_TYPE);
            if (it == info.end()) {
                continue;
            }
            const auto type = it->second.as<std::string>();
            has_sparse |= type == "SparseSDPA";
            EXPECT_NE(type, "Concat") << "the KV cache Concat must be fused into the sparse attention node";
        }
        EXPECT_TRUE(has_sparse) << "SparseSDPA node is expected in the runtime model";
    }
};

TEST_P(StatefulSparseAttentionTest, CompareWithStatelessReference) {
    const auto& [prec, cache_prec, key_by_channel, N, E, Bk, Hs] = GetParam();
    if (prec == ov::element::bf16 && !ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "bf16 is not supported on this machine";
    }
    auto core = ov::test::utils::PluginCache::get().core();
    // the reference keeps a plain (not compressed) cache: the stateful model is compared against it, a compressed cache
    // is only expected to be close (the groups of 32 elements are quantized with a {scale, zero point} each)
    ov::AnyMap ref_config{{ov::hint::inference_precision.name(), prec}};
    ov::AnyMap config = ref_config;
    config[ov::hint::kv_cache_precision.name()] = cache_prec;
    config[ov::key_cache_group_size.name()] = static_cast<uint64_t>(32);
    config[ov::value_cache_group_size.name()] = static_cast<uint64_t>(32);
    if (key_by_channel) {
        // the key groups are 32 tokens long: the prefill ends in a partial group that the decodes keep filling
        config[ov::internal::key_cache_quant_mode.name()] = ov::internal::CacheQuantMode::BY_CHANNEL;
    }
    // the random selection of blocks below; the K/V/Q values come from the common tensor helper
    std::mt19937 gen(1234);
    int seed = 1;
    // the prefill is not a multiple of the block size; the decode steps outgrow the capacity of the buffers twice
    std::vector<Step> steps{{19, false}, {1, false}, {1, false}, {5, false}};
    for (size_t i = 0; i < 24; i++) {
        steps.push_back({1, false});
    }
    if (N > 1) {
        steps.push_back({1, true});
        steps.push_back({1, false});
    }
    steps.push_back({3, false});

    auto compiled = core->compile_model(make_stateful_model(prec, E, Bk, Hs), ov::test::utils::DEVICE_CPU, config);
    auto compiled_ref =
        core->compile_model(make_reference_model(prec, E, Bk, Hs), ov::test::utils::DEVICE_CPU, ref_config);
    check_fused(compiled);

    // bf16 inference rounds the cache once, in both models
    float threshold = prec == ov::element::bf16 ? 2e-2F : 1e-4F;
    if (cache_prec == ov::element::u8) {
        threshold = 3e-2F;
    } else if (cache_prec == ov::element::u4) {
        threshold = 0.15F;
    }

    auto request = compiled.create_infer_request();
    auto request_ref = compiled_ref.create_infer_request();

    // runs one sequence of steps from the current (reset) state of the request
    auto run_sequence = [&](HostCache& k_cache, HostCache& v_cache, const std::vector<Step>& seq) {
        for (size_t step = 0; step < seq.size(); step++) {
            const auto L = seq[step].len;
            std::vector<int32_t> beam(N);
            std::iota(beam.begin(), beam.end(), 0);
            if (seq[step].reorder_beam) {
                std::reverse(beam.begin(), beam.end());
                k_cache.reorder(beam);
                v_cache.reorder(beam);
            }

            const auto q_t = random_tensor(prec, ov::Shape{N, kHeads, L, E}, seed++);
            const auto k_t = random_tensor(prec, ov::Shape{N, kKvHeads, L, E}, seed++);
            const auto v_t = random_tensor(prec, ov::Shape{N, kKvHeads, L, E}, seed++);
            k_cache.append(to_f32(k_t), L);
            v_cache.append(to_f32(v_t), L);
            std::vector<int32_t> indices;
            std::vector<int32_t> count;
            make_selection(gen, N, Hs, L, k_cache.S, Bk, indices, count);

            const auto idx_t = to_tensor(indices, ov::Shape{N, Hs, L, kTopK});
            const auto cnt_t = to_tensor(count, ov::Shape{N, Hs, L});

            request.set_tensor("q", q_t);
            request.set_tensor("k", k_t);
            request.set_tensor("v", v_t);
            request.set_tensor("block_indices", idx_t);
            request.set_tensor("block_count", cnt_t);
            request.set_tensor("beam_idx", to_tensor(beam, ov::Shape{N}));
            request.set_tensor("past_init", ov::Tensor(prec, ov::Shape{N, kKvHeads, 0, E}));
            request.infer();

            request_ref.set_tensor("q", q_t);
            request_ref.set_tensor("k", k_cache.tensor(prec));
            request_ref.set_tensor("v", v_cache.tensor(prec));
            request_ref.set_tensor("block_indices", idx_t);
            request_ref.set_tensor("block_count", cnt_t);
            request_ref.infer();

            compare(
                request_ref.get_output_tensor(0),
                request.get_output_tensor(0),
                threshold,
                "step " + std::to_string(step) + " (L=" + std::to_string(L) + ", S=" + std::to_string(k_cache.S) + ")");
            if (::testing::Test::HasFailure()) {
                return;
            }
        }
    };

    HostCache k_cache(N, E);
    HostCache v_cache(N, E);
    ASSERT_NO_FATAL_FAILURE(run_sequence(k_cache, v_cache, steps));

    // the state exposes the accumulated cache in the model layout and precision
    for (auto&& state : request.query_state()) {
        const auto& expected = state.get_name() == "pastk" ? k_cache : v_cache;
        // a compressed cache must not be exact: that proves the cache is really quantized
        const float min_diff = cache_prec == ov::element::f32 ? 0.F : 1e-4F;
        compare(expected.tensor(prec), state.get_state(), threshold, "get_state " + state.get_name(), min_diff);
    }

    // reset: a new sequence starts from the empty cache and reuses the buffers
    request.reset_state();
    HostCache k_cache2(N, E);
    HostCache v_cache2(N, E);
    ASSERT_NO_FATAL_FAILURE(run_sequence(k_cache2, v_cache2, {{9, false}, {1, false}, {1, false}}));

    // set_state: continue from a cache written from the outside
    {
        const size_t L_init = 11;
        HostCache k_init(N, E);
        HostCache v_init(N, E);
        k_init.append(to_f32(random_tensor(prec, ov::Shape{N, kKvHeads, L_init, E}, seed++)), L_init);
        v_init.append(to_f32(random_tensor(prec, ov::Shape{N, kKvHeads, L_init, E}, seed++)), L_init);
        for (auto&& state : request.query_state()) {
            state.set_state(state.get_name() == "pastk" ? k_init.tensor(prec) : v_init.tensor(prec));
        }
        ASSERT_NO_FATAL_FAILURE(run_sequence(k_init, v_init, {{1, false}, {1, false}, {2, false}}));
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_StatefulSparseAttention,
                         StatefulSparseAttentionTest,
                         ::testing::Combine(::testing::Values(ov::element::f32, ov::element::bf16),
                                            ::testing::Values(ov::element::f32, ov::element::u8, ov::element::u4),
                                            ::testing::Values(false),
                                            ::testing::Values(1, 2),
                                            ::testing::Values(32, 64),
                                            ::testing::Values(4, 16),
                                            ::testing::Values(1, 2)),
                         StatefulSparseAttentionTest::getTestCaseName);

// keys quantized by channel (u8 only); the values are quantized by token
INSTANTIATE_TEST_SUITE_P(smoke_StatefulSparseAttention_KeyByChannel,
                         StatefulSparseAttentionTest,
                         ::testing::Combine(::testing::Values(ov::element::f32, ov::element::bf16),
                                            ::testing::Values(ov::element::u8),
                                            ::testing::Values(true),
                                            ::testing::Values(1, 2),
                                            ::testing::Values(32, 64),
                                            ::testing::Values(4, 16),
                                            ::testing::Values(1, 2)),
                         StatefulSparseAttentionTest::getTestCaseName);

}  // namespace test
}  // namespace ov
