// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather_matmul.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <numeric>
#include <random>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

// Batched prefill (n_tokens > 16) of GatherMatmul: sort -> gather -> grouped GEMM [-> scatter].
// f16/i8 weights take the oneDNN grouped path (expert-major sort, gather, scatter); u4 takes the
// OCL batched GEMM path (slot-major sort, gather). The sort runs in local memory when
// n_experts * top_k <= 2048 groups and falls back to a sequential kernel otherwise.
struct gather_matmul_prefill_params {
    std::string name;
    std::vector<size_t> n_tokens;  // executed in order on one network
    size_t top_k;
    size_t n_experts;
    size_t n_activated;  // 1 broadcasts A over slots, top_k gives one A row block per slot
    size_t n;
    size_t k;
    data_types weights_type;
    size_t group_size;  // 0 for per-channel scales
    bool skewed;        // a quarter of the experts stay empty and expert 0 takes most tokens
    bool dynamic;
    bool zp = false;  // u8 zero point 8 per (expert, channel, group)
};

void PrintTo(const gather_matmul_prefill_params& p, std::ostream* os) {
    *os << p.name;
}

std::vector<int32_t> make_routing(size_t n_tokens, const gather_matmul_prefill_params& p, std::mt19937& rng) {
    std::vector<int32_t> experts(p.n_experts);
    std::iota(experts.begin(), experts.end(), 0);
    if (p.skewed) {
        experts.erase(std::remove_if(experts.begin(),
                                     experts.end(),
                                     [](int32_t e) {
                                         return e == 0 || e % 4 == 3;
                                     }),
                      experts.end());
    }
    std::vector<int32_t> indices(n_tokens * p.top_k);
    for (size_t t = 0; t < n_tokens; t++) {
        std::shuffle(experts.begin(), experts.end(), rng);
        for (size_t s = 0; s < p.top_k; s++)
            indices[t * p.top_k + s] = experts[s];
        if (p.skewed && rng() % 10 != 0)
            indices[t * p.top_k + t % p.top_k] = 0;
    }
    return indices;
}

class gather_matmul_prefill_gpu_test : public ::testing::TestWithParam<gather_matmul_prefill_params> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<gather_matmul_prefill_params>& info) {
        return info.param.name;
    }

    void run() {
        const auto& p = GetParam();
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            GTEST_SKIP() << "GatherMatmul requires a device with systolic (XMX) support";
        if (p.weights_type == data_types::u4 && engine.get_device_info().arch < gpu_arch::xe2)
            GTEST_SKIP() << "No u4 GatherMatmul micro-GEMM kernel before Xe2";

        const bool quantized = p.weights_type != data_types::f16;
        const bool is_u4 = p.weights_type == data_types::u4;
        const size_t n_groups = p.group_size ? p.k / p.group_size : 1;
        std::mt19937 rng(static_cast<uint32_t>(p.n_experts * 131 + p.top_k * 17 + p.k));
        std::uniform_real_distribution<float> unit(-1.f, 1.f);

        // Dequantized weights [E, N, K] for the reference.
        std::vector<float> w_ref(p.n_experts * p.n * p.k);
        std::vector<ov::float16> scales(p.n_experts * p.n * n_groups);
        for (auto& s : scales)
            s = ov::float16(is_u4 ? 0.01f + 0.01f * (unit(rng) + 1.f) : 0.005f + 0.005f * (unit(rng) + 1.f));
        auto scale_of = [&](size_t e, size_t n, size_t k) {
            return static_cast<float>(scales[(e * p.n + n) * n_groups + (p.group_size ? k / p.group_size : 0)]);
        };

        layout w_layout{ov::PartialShape{static_cast<int64_t>(p.n_experts), static_cast<int64_t>(p.n), static_cast<int64_t>(p.k)},
                        p.weights_type,
                        format::bfyx};
        auto w_mem = engine.allocate_memory(w_layout);
        if (p.weights_type == data_types::f16) {
            std::vector<ov::float16> w(w_ref.size());
            for (size_t i = 0; i < w.size(); i++) {
                w[i] = ov::float16(0.5f * unit(rng));
                w_ref[i] = static_cast<float>(w[i]);
            }
            set_values(w_mem, w);
        } else if (p.weights_type == data_types::i8) {
            std::vector<int8_t> w(w_ref.size());
            for (size_t i = 0; i < w.size(); i++) {
                w[i] = static_cast<int8_t>(static_cast<int>(rng() % 255) - 127);
                const size_t e = i / (p.n * p.k), n = (i / p.k) % p.n, k = i % p.k;
                w_ref[i] = (static_cast<float>(w[i]) - (p.zp ? 8.f : 0.f)) * scale_of(e, n, k);
            }
            set_values(w_mem, w);
        } else {
            std::vector<uint8_t> w(w_ref.size() / 2);
            for (size_t i = 0; i < w_ref.size(); i++) {
                const uint8_t q = static_cast<uint8_t>(rng() % 16);
                w[i / 2] |= (i % 2) ? static_cast<uint8_t>(q << 4) : q;
                const size_t e = i / (p.n * p.k), n = (i / p.k) % p.n, k = i % p.k;
                w_ref[i] = (static_cast<float>(q) - (p.zp ? 8.f : 0.f)) * scale_of(e, n, k);
            }
            set_values(w_mem, w);
        }

        const auto n_act = static_cast<int64_t>(p.n_activated);
        const auto top_k = static_cast<int64_t>(p.top_k);
        const auto k_dim = static_cast<int64_t>(p.k);
        auto in_pshape = p.dynamic ? ov::PartialShape{n_act, -1, k_dim} : ov::PartialShape{n_act, static_cast<int64_t>(p.n_tokens[0]), k_dim};
        auto idx_pshape = p.dynamic ? ov::PartialShape{-1, top_k} : ov::PartialShape{static_cast<int64_t>(p.n_tokens[0]), top_k};

        topology topology(input_layout("input", layout{in_pshape, data_types::f16, format::bfyx}),
                          input_layout("indices", layout{idx_pshape, data_types::i32, format::bfyx}),
                          data("weights", w_mem));
        auto placeholder = engine.allocate_memory(layout{ov::PartialShape{1}, data_types::f16, format::bfyx});
        set_values(placeholder, {ov::float16(0.f)});
        topology.add(data("bias", placeholder));
        std::vector<input_info> inputs{input_info("input"), input_info("weights"), input_info("indices"), input_info("bias")};
        if (quantized) {
            auto s_mem =
                engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(p.n_experts), static_cast<int64_t>(p.n), static_cast<int64_t>(n_groups)},
                                              data_types::f16,
                                              format::bfyx});
            set_values(s_mem, scales);
            auto zp_pshape =
                p.zp ? ov::PartialShape{static_cast<int64_t>(p.n_experts), static_cast<int64_t>(p.n), static_cast<int64_t>(n_groups)} : ov::PartialShape{1};
            auto zp_mem = engine.allocate_memory(layout{zp_pshape, data_types::u8, format::bfyx});
            set_values(zp_mem, std::vector<uint8_t>(ov::shape_size(zp_pshape.to_shape()), p.zp ? 8 : 0));
            topology.add(data("scales", s_mem), data("zp", zp_mem));
            inputs.emplace_back("scales");
            inputs.emplace_back("zp");
        }
        topology.add(gather_matmul("bgm", inputs, false, p.zp, static_cast<int32_t>(p.n_activated)));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topology, config);

        for (size_t n_tokens : p.n_tokens) {
            SCOPED_TRACE("n_tokens=" + std::to_string(n_tokens));
            std::vector<ov::float16> a(p.n_activated * n_tokens * p.k);
            for (auto& v : a)
                v = ov::float16(unit(rng));
            const auto indices = make_routing(n_tokens, p, rng);

            auto a_mem = engine.allocate_memory(layout{ov::PartialShape{n_act, static_cast<int64_t>(n_tokens), k_dim}, data_types::f16, format::bfyx});
            auto idx_mem = engine.allocate_memory(layout{ov::PartialShape{static_cast<int64_t>(n_tokens), top_k}, data_types::i32, format::bfyx});
            set_values(a_mem, a);
            set_values(idx_mem, indices);
            network.set_input_data("input", a_mem);
            network.set_input_data("indices", idx_mem);
            auto outputs = network.execute();
            auto out_mem = outputs.at("bgm").get_memory();
            ASSERT_EQ(out_mem->get_layout().get_shape(), (ov::Shape{p.top_k, n_tokens, p.n}));
            cldnn::mem_lock<ov::float16, mem_lock_type::read> out(out_mem, get_test_stream());

            size_t mismatches = 0;
            for (size_t s = 0; s < p.top_k; s++) {
                const size_t a_slot = std::min(s, p.n_activated - 1);
                for (size_t t = 0; t < n_tokens; t++) {
                    const size_t e = static_cast<size_t>(indices[t * p.top_k + s]);
                    const ov::float16* a_row = a.data() + (a_slot * n_tokens + t) * p.k;
                    for (size_t n = 0; n < p.n; n++) {
                        const float* w_row = w_ref.data() + (e * p.n + n) * p.k;
                        float ref = 0.f;
                        for (size_t k = 0; k < p.k; k++)
                            ref += static_cast<float>(a_row[k]) * w_row[k];
                        const float got = static_cast<float>(out[(s * n_tokens + t) * p.n + n]);
                        if (!(std::fabs(got - ref) <= 2e-2f + 1e-2f * std::fabs(ref)) && mismatches++ < 5) {
                            ADD_FAILURE() << "slot " << s << " token " << t << " expert " << e << " n " << n << ": got " << got << ", expected " << ref;
                        }
                    }
                }
            }
            ASSERT_EQ(mismatches, 0u);
        }
    }
};

TEST_P(gather_matmul_prefill_gpu_test, compare_with_reference) {
    run();
}

const auto f16 = data_types::f16;
const auto i8 = data_types::i8;
const auto u4 = data_types::u4;

// {name, n_tokens, top_k, n_experts, n_activated, N, K, weights, group_size, skewed, dynamic[, zp]}
INSTANTIATE_TEST_SUITE_P(smoke_gather_matmul_prefill_local_sort,
                         gather_matmul_prefill_gpu_test,
                         ::testing::Values(
                             // 256 groups with n_tokens * top_k > 2048 rows.
                             gather_matmul_prefill_params{"f16_e64_k4", {600, 33}, 4, 64, 1, 128, 256, f16, 0, true, true},
                             gather_matmul_prefill_params{"i8_e128_k8_up", {579, 17}, 8, 128, 1, 256, 256, i8, 0, true, true},
                             gather_matmul_prefill_params{"i8_e128_k8_down", {200}, 8, 128, 8, 128, 256, i8, 128, true, true},
                             gather_matmul_prefill_params{"i8_e128_k8_static", {300}, 8, 128, 1, 128, 256, i8, 0, false, false},
                             // Largest local group count.
                             gather_matmul_prefill_params{"i8_e256_k8_2048_groups", {400}, 8, 256, 1, 128, 256, i8, 0, true, true},
                             gather_matmul_prefill_params{"u4_e128_k8_up", {600, 20}, 8, 128, 1, 256, 256, u4, 128, true, true},
                             gather_matmul_prefill_params{"u4_zp_e128_k8_down", {100}, 8, 128, 8, 128, 256, u4, 128, true, true, true}),
                         gather_matmul_prefill_gpu_test::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_gather_matmul_prefill_fallback_sort,
                         gather_matmul_prefill_gpu_test,
                         ::testing::Values(
                             // Smallest group count above the local limit.
                             gather_matmul_prefill_params{"i8_e257_k8", {300}, 8, 257, 1, 128, 256, i8, 0, true, true},
                             gather_matmul_prefill_params{"f16_e512_k8", {150, 40}, 8, 512, 1, 128, 256, f16, 0, true, true},
                             gather_matmul_prefill_params{"i8_zp_e512_k8_down", {120}, 8, 512, 8, 128, 256, i8, 128, false, true, true},
                             gather_matmul_prefill_params{"u4_e512_k8_up", {300, 24}, 8, 512, 1, 128, 256, u4, 128, true, true}),
                         gather_matmul_prefill_gpu_test::get_test_case_name);

}  // namespace
