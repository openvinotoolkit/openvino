// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// GatherMatmul (MoE experts) with ternary (u2 codes {0, 1, 2}, zero point 1) and binary (u1, zero point 0.5) compressed
// weights: the GPU plugin runs them on the base-3 t1 / binary u1 kernels (dp4a GEMV for a few tokens, int8 DPAS prefill
// from 16 tokens). Each case is compared with the same model whose expert weights are given already decompressed,
// which takes the regular GatherMatmul path, and must allocate a fraction of its device memory.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "ov_ops/gather_matmul.hpp"

namespace {

struct LowBitCase {
    ov::element::Type weights_type;  // u2 (ternary) or u1 (binary)
    size_t tokens;
};

std::string case_name(const testing::TestParamInfo<LowBitCase>& info) {
    return std::string(info.param.weights_type == ov::element::u2 ? "ternary" : "binary") + "_T" + std::to_string(info.param.tokens);
}

class GatherMatmulLowBit : public testing::TestWithParam<LowBitCase> {
protected:
    static constexpr size_t E = 8, N = 256, K = 512, G = 128, TOP_K = 2;

    // Expert weight codes with one scale per (expert, row, group of G); also returns the decompressed values.
    static void make_weights(ov::element::Type wt, ov::Tensor& codes, ov::Tensor& scales, std::vector<float>& dense, float& zp) {
        std::mt19937 rng(7);
        const uint32_t levels = wt == ov::element::u2 ? 3 : 2;
        zp = wt == ov::element::u2 ? 1.0f : 0.5f;
        codes = ov::Tensor(wt, ov::Shape{E, N, K / G, G});
        scales = ov::Tensor(ov::element::f32, ov::Shape{E, N, K / G, 1});
        auto* c = static_cast<uint8_t*>(codes.data());
        std::fill_n(c, codes.get_byte_size(), uint8_t{0});
        dense.resize(E * N * K);
        std::uniform_int_distribution<uint32_t> code_dist(0, levels - 1);
        std::uniform_real_distribution<float> scale_dist(0.01f, 0.05f);
        auto* s = scales.data<float>();
        for (size_t g = 0; g < E * N * K / G; ++g) {
            s[g] = static_cast<float>(ov::float16(scale_dist(rng)));
        }
        for (size_t i = 0; i < E * N * K; ++i) {
            const uint32_t v = code_dist(rng);
            if (wt == ov::element::u2) {
                c[i / 4] |= static_cast<uint8_t>(v << ((i % 4) * 2));  // u2: least significant pair first
            } else {
                c[i / 8] |= static_cast<uint8_t>(v << (7 - i % 8));  // u1: most significant bit first
            }
            dense[i] = (static_cast<float>(v) - zp) * s[i / G];
        }
    }

    // A [1, tokens, K] (dynamic token count, as in a real model), expert ids [tokens, TOP_K]
    static std::shared_ptr<ov::Model> make_model(const ov::Output<ov::Node>& weights) {
        auto a = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, static_cast<int64_t>(K)});
        auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1, static_cast<int64_t>(TOP_K)});
        auto gm = std::make_shared<ov::op::internal::GatherMatmul>(a, weights, ids);
        return std::make_shared<ov::Model>(ov::OutputVector{gm}, ov::ParameterVector{a, ids});
    }

    // Runs the model on the GPU; also returns the device memory the compiled model holds.
    static ov::Tensor run(const std::shared_ptr<ov::Model>& model, const ov::Tensor& a, const ov::Tensor& ids, uint64_t& mem_bytes) {
        ov::Core core;
        auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(ov::element::f16));
        mem_bytes = 0;
        for (const auto& [type, bytes] : core.get_property("GPU", ov::intel_gpu::memory_statistics)) {
            mem_bytes += bytes;
        }
        auto req = compiled.create_infer_request();
        req.set_input_tensor(0, a);
        req.set_input_tensor(1, ids);
        req.infer();
        ov::Tensor out(req.get_output_tensor().get_element_type(), req.get_output_tensor().get_shape());
        req.get_output_tensor().copy_to(out);
        return out;
    }
};

TEST_P(GatherMatmulLowBit, MatchesDecompressedWeights) {
    const auto& p = GetParam();
    ov::Tensor codes, scales;
    std::vector<float> dense;
    float zp = 0;
    make_weights(p.weights_type, codes, scales, dense, zp);

    const auto final_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {int64_t(E), int64_t(N), int64_t(K)});
    const auto compressed = make_model(ov::decomposition::low_precision_dequantize(std::make_shared<ov::op::v0::Constant>(codes),
                                                                                   std::make_shared<ov::op::v0::Constant>(scales),
                                                                                   ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {zp}),
                                                                                   final_shape));
    ov::Tensor dense_t(ov::element::f32, ov::Shape{E, N, K});
    std::copy(dense.begin(), dense.end(), dense_t.data<float>());
    const auto reference = make_model(std::make_shared<ov::op::v0::Constant>(dense_t));

    std::mt19937 rng(11);
    std::normal_distribution<float> act(0.f, 1.f);
    ov::Tensor a(ov::element::f32, ov::Shape{1, p.tokens, K});
    for (size_t i = 0; i < a.get_size(); ++i) {
        a.data<float>()[i] = act(rng);
    }
    ov::Tensor ids(ov::element::i32, ov::Shape{p.tokens, TOP_K});
    for (size_t t = 0; t < p.tokens; ++t) {
        for (size_t k = 0; k < TOP_K; ++k) {
            ids.data<int32_t>()[t * TOP_K + k] = static_cast<int32_t>((t * 3 + k * 5) % E);
        }
    }

    uint64_t ref_mem = 0, got_mem = 0;
    const auto ref = run(reference, a, ids, ref_mem);
    const auto got = run(compressed, a, ids, got_mem);
    // The expert weights stay compressed on the device (1.625 or 1 bit per weight instead of 16).
    EXPECT_LT(got_mem * 4, ref_mem) << "compressed model holds " << got_mem << " bytes vs " << ref_mem << " decompressed";

    ASSERT_EQ(got.get_shape(), ref.get_shape());

    // Exact model of the kernels: the activations are rounded to f16, then each 128 values are quantized to int8 with
    // sc = max|x| / 127 and q = rte(x * (1 / sc)). Against this, only the f16 output rounding remains, so a repack or
    // decode error of even one weight per row shows up.
    std::vector<float> qa(p.tokens * K);
    for (size_t t = 0; t < p.tokens; ++t) {
        for (size_t g = 0; g < K / 128; ++g) {
            const float* x = a.data<float>() + t * K + g * 128;
            float m = 0.f;
            for (size_t i = 0; i < 128; ++i) {
                m = std::max(m, std::fabs(static_cast<float>(ov::float16(x[i]))));
            }
            const float sc = m > 0.f ? m / 127.f : 1.f;
            const float inv = 1.f / sc;
            for (size_t i = 0; i < 128; ++i) {
                qa[t * K + g * 128 + i] = std::nearbyint(static_cast<float>(ov::float16(x[i])) * inv) * sc;
            }
        }
    }
    float max_q = 0.f, max_qdiff = 0.f;
    for (size_t k = 0; k < TOP_K; ++k) {
        for (size_t t = 0; t < p.tokens; ++t) {
            const auto e = static_cast<size_t>(ids.data<int32_t>()[t * TOP_K + k]);
            for (size_t n = 0; n < N; ++n) {
                double acc = 0;
                for (size_t i = 0; i < K; ++i) {
                    acc += static_cast<double>(qa[t * K + i]) * dense[(e * N + n) * K + i];
                }
                const float g = got.data<float>()[(k * p.tokens + t) * N + n];
                max_q = std::max(max_q, std::fabs(static_cast<float>(acc)));
                max_qdiff = std::max(max_qdiff, std::fabs(static_cast<float>(acc) - g));
            }
        }
    }
    EXPECT_LE(max_qdiff, 1e-3f * max_q) << "vs the int8-activation model: max |diff| " << max_qdiff << " vs max " << max_q;

    // And against the decompressed f16 model, within the int8 activation error.
    float max_ref = 0.f, max_diff = 0.f;
    for (size_t i = 0; i < ref.get_size(); ++i) {
        max_ref = std::max(max_ref, std::fabs(ref.data<float>()[i]));
        max_diff = std::max(max_diff, std::fabs(ref.data<float>()[i] - got.data<float>()[i]));
    }
    EXPECT_LE(max_diff, 0.02f * max_ref) << "max |diff| " << max_diff << " vs max |ref| " << max_ref;
}

INSTANTIATE_TEST_SUITE_P(smoke_GatherMatmulLowBit,
                         GatherMatmulLowBit,
                         testing::Values(LowBitCase{ov::element::u2, 1},
                                         LowBitCase{ov::element::u2, 4},
                                         LowBitCase{ov::element::u2, 32},
                                         LowBitCase{ov::element::u1, 1},
                                         LowBitCase{ov::element::u1, 4},
                                         LowBitCase{ov::element::u1, 32}),
                         case_name);

}  // namespace
