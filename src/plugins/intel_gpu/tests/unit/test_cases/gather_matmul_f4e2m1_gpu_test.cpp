// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// gather_matmul with f4e2m1 (MXFP4-style) weights vs. host f32 reference.

#ifdef ENABLE_ONEDNN_FOR_GPU

#include "test_utils.h"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather_matmul.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include <random>

using namespace cldnn;
using namespace ::tests;

namespace {

// E2M1 values doubled; the floating-point scales below are not encoded as E8M0.
constexpr int8_t kvalues_mxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

struct gm_f4_params {
    size_t n_tokens;
    bool has_bias;
    data_types scale_type;
};

std::string to_string(const ::testing::TestParamInfo<gm_f4_params>& info) {
    const auto& p = info.param;
    return "tokens" + std::to_string(p.n_tokens) + (p.has_bias ? "_bias" : "_nobias") +
           (p.scale_type == data_types::f32 ? "_scale_f32" : "_scale_f16");
}

class gather_matmul_f4e2m1_gpu : public ::testing::TestWithParam<gm_f4_params> {};

TEST_P(gather_matmul_f4e2m1_gpu, compare_with_host_reference) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        GTEST_SKIP() << "gather_matmul requires IMMAD";

    const auto p = GetParam();
    constexpr size_t E = 4, N = 64, K = 128, top_k = 2, group = 32, G = K / group;
    const size_t T = p.n_tokens;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_real_distribution<float> act(-1.f, 1.f);
    std::uniform_real_distribution<float> bdist(-0.5f, 0.5f);
    const float scale_choices[] = {0.0625f, 0.125f, 0.25f, 0.1f, 0.3f};
    std::uniform_int_distribution<int> sdist(0, 4);

    // Weights [E, N, K], two f4 values per byte, element 0 in the low nibble.
    std::vector<uint8_t> w_nibbles(E * N * K);
    for (auto& v : w_nibbles) v = static_cast<uint8_t>(nib(rng));
    std::vector<uint8_t> w_packed(E * N * K / 2);
    for (size_t i = 0; i < w_packed.size(); ++i)
        w_packed[i] = static_cast<uint8_t>(w_nibbles[2 * i] | (w_nibbles[2 * i + 1] << 4));

    // Scales logical [E, N, G]; reference uses values after rounding to the device type.
    std::vector<float> scales(E * N * G);
    for (auto& s : scales) {
        s = scale_choices[sdist(rng)];
        if (p.scale_type == data_types::f16) s = static_cast<float>(ov::float16(s));
    }

    // Activations [1, T, K] (n_activated_experts = 1 → broadcast over top_k slots).
    std::vector<ov::float16> a_f16(T * K);
    for (auto& v : a_f16) v = ov::float16(act(rng));

    // Indices [T, top_k], distinct experts per token.
    std::vector<int32_t> idx(T * top_k);
    for (size_t t = 0; t < T; ++t)
        for (size_t j = 0; j < top_k; ++j)
            idx[t * top_k + j] = static_cast<int32_t>((t + j * (E / top_k)) % E);

    // Bias [E, 1, N].
    std::vector<ov::float16> bias_f16(E * N);
    for (auto& v : bias_f16) v = ov::float16(bdist(rng));

    // ---- device memories ----
    auto a_mem = engine.allocate_memory({ov::PartialShape{1, (int64_t)T, (int64_t)K}, data_types::f16, format::bfyx});
    auto idx_mem = engine.allocate_memory({ov::PartialShape{(int64_t)T, (int64_t)top_k}, data_types::i32, format::bfyx});
    auto w_mem = engine.allocate_memory({ov::PartialShape{(int64_t)E, (int64_t)N, (int64_t)K}, data_types::f4e2m1, format::bfyx});
    auto s_mem = engine.allocate_memory({ov::PartialShape{(int64_t)E, (int64_t)N, (int64_t)G}, p.scale_type, format::bfyx});
    set_values(a_mem, a_f16);
    set_values(idx_mem, idx);
    set_values(w_mem, w_packed);
    if (p.scale_type == data_types::f16) {
        std::vector<ov::float16> s16(scales.begin(), scales.end());
        set_values(s_mem, s16);
    } else {
        set_values(s_mem, scales);
    }

    memory::ptr bias_mem;
    if (p.has_bias) {
        bias_mem = engine.allocate_memory({ov::PartialShape{(int64_t)E, 1, (int64_t)N}, data_types::f16, format::bfyx});
        set_values(bias_mem, bias_f16);
    } else {
        bias_mem = engine.allocate_memory({ov::PartialShape{1}, data_types::f16, format::bfyx});
        set_values(bias_mem, std::vector<ov::float16>{ov::float16(0.f)});
    }
    auto zp_mem = engine.allocate_memory({ov::PartialShape{1}, data_types::u8, format::bfyx});
    set_values(zp_mem, std::vector<uint8_t>{0});

    topology topo(input_layout("input", a_mem->get_layout()),
                  input_layout("indices", idx_mem->get_layout()),
                  data("weights", w_mem),
                  data("bias", bias_mem),
                  data("scales", s_mem),
                  data("zp", zp_mem),
                  gather_matmul("bgm",
                                {input_info("input"), input_info("weights"), input_info("indices"),
                                 input_info("bias"), input_info("scales"), input_info("zp")},
                                p.has_bias, /*has_zp=*/false, /*n_activated_experts=*/1));

    ExecutionConfig cfg = get_test_default_config(engine);
    cfg.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cfg.set_property(ov::intel_gpu::optimize_data(true));
    network net(engine, topo, cfg);
    net.set_input_data("input", a_mem);
    net.set_input_data("indices", idx_mem);
    auto outputs = net.execute();
    auto out_mem = outputs.at("bgm").get_memory();
    ASSERT_EQ(out_mem->get_layout().get_shape(), (ov::Shape{top_k, T, N}));

    // OpenVINO packs adjacent E2M1 values, unlike ggml's two halves of each 32-value block.
    std::vector<float> w_deq(E * N * K);
    for (size_t e = 0; e < E; ++e)
        for (size_t n = 0; n < N; ++n)
            for (size_t k = 0; k < K; ++k) {
                const size_t wi = (e * N + n) * K + k;
                const float d = scales[(e * N + n) * G + k / group];
                const uint8_t packed = w_packed[wi / 2];
                const uint8_t code = (wi % 2 == 0) ? (packed & 0x0f) : (packed >> 4);
                w_deq[wi] = static_cast<float>(kvalues_mxfp4[code]) * 0.5f * d;
            }

    cldnn::mem_lock<ov::float16, mem_lock_type::read> out(out_mem, get_test_stream());
    for (size_t j = 0; j < top_k; ++j)
        for (size_t t = 0; t < T; ++t) {
            const size_t e = static_cast<size_t>(idx[t * top_k + j]);
            for (size_t n = 0; n < N; ++n) {
                float ref = 0.f;
                for (size_t k = 0; k < K; ++k)
                    ref += static_cast<float>(a_f16[t * K + k]) * w_deq[(e * N + n) * K + k];
                if (p.has_bias) ref += static_cast<float>(bias_f16[e * N + n]);
                const float got = static_cast<float>(out[(j * T + t) * N + n]);
                ASSERT_NEAR(got, ref, 5e-2f + 1e-2f * std::fabs(ref))
                    << "slot=" << j << " token=" << t << " n=" << n << " expert=" << e;
            }
        }
}

INSTANTIATE_TEST_SUITE_P(smoke_gather_matmul_f4e2m1,
                         gather_matmul_f4e2m1_gpu,
                         ::testing::Values(gm_f4_params{1, false, data_types::f16},   // decode
                                           gm_f4_params{1, true, data_types::f16},
                                           gm_f4_params{8, false, data_types::f16},   // small prefill
                                           gm_f4_params{8, true, data_types::f16},
                                           gm_f4_params{33, false, data_types::f16},  // > BATCHED_PREFILL_THRESHOLD
                                           gm_f4_params{33, true, data_types::f16}),
                         to_string);

// Exposes scale-type handling: validate_impl has no f32 in supported_quant_param_types, and
// get_grouped_kernel hardcodes f16 in set_scales while scale_md uses the real type.
INSTANTIATE_TEST_SUITE_P(smoke_gather_matmul_f4e2m1_scale_f32,
                         gather_matmul_f4e2m1_gpu,
                         ::testing::Values(gm_f4_params{1, false, data_types::f32},
                                           gm_f4_params{33, true, data_types::f32}),
                         to_string);

}  // namespace

#endif  // ENABLE_ONEDNN_FOR_GPU