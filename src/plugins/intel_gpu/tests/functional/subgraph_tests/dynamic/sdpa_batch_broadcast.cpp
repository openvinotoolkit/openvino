// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/runtime/core.hpp"

// ScaledDotProductAttention whose Q, K and V batches broadcast numpy-style (each is 1 or the output batch), executed
// end to end: shape inference, the transformations and the kernel, with dynamic batch and sequence lengths in one
// compiled model. Whisper beam search is the real case: in the cross-attention, Q batch = the number of beams and K/V
// batch = 1. sdpa_ocl read every input with the output batch index (past the end of a batch-1 input: NaN, wrong rows or
// a GPU page fault). The equal-batch step is the control. The oracle is an fp32 reference over the same f16 inputs: the
// TEMPLATE plugin computes in f16 (0.02 off over 1500 keys) and throws "Tile with non-positive repeats" on some of
// these shapes.
namespace {

enum class Variant { PLAIN, MASK, CAUSAL };

struct SDPABroadcastParams {
    size_t head_size;
    Variant variant;
};

constexpr size_t num_heads = 8;

struct Step {
    size_t q_batch, k_batch, v_batch, seq_q, seq_kv;
};

std::shared_ptr<ov::Model> make_sdpa_model(const SDPABroadcastParams& p) {
    const auto f16 = ov::element::f16;
    const auto D = static_cast<int64_t>(p.head_size);
    const auto H = static_cast<int64_t>(num_heads);
    auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, H, -1, D});
    auto key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, H, -1, D});
    auto value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, H, -1, D});
    ov::ParameterVector params{query, key, value};
    std::shared_ptr<ov::Node> sdpa;
    if (p.variant == Variant::MASK) {
        // [output batch, 1, L, S]: per-batch values, broadcast over the heads.
        auto mask = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{-1, 1, -1, -1});
        params.push_back(mask);
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, false);
    } else {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, p.variant == Variant::CAUSAL);
    }
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa)}, params);
}

ov::Tensor random_tensor(const ov::Shape& shape, std::mt19937& gen, float lo, float hi, float batch_step = 0.0f) {
    ov::Tensor tensor(ov::element::f16, shape);
    std::uniform_real_distribution<float> dist(lo, hi);
    auto* data = tensor.data<ov::float16>();
    const size_t per_batch = tensor.get_size() / shape[0];
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = ov::float16(dist(gen) + batch_step * static_cast<float>(i / per_batch));
    }
    return tensor;
}

// fp32 attention, opset13 semantics: batch b reads Q/K/V/mask row b % its own batch; causal keeps key j <= query i.
std::vector<float> reference(const std::vector<ov::Tensor>& in, const Step& s, size_t D, bool causal) {
    const size_t out_batch = std::max({s.q_batch, s.k_batch, s.v_batch});
    const auto* q = in[0].data<const ov::float16>();
    const auto* k = in[1].data<const ov::float16>();
    const auto* v = in[2].data<const ov::float16>();
    const ov::float16* m = in.size() > 3 ? in[3].data<const ov::float16>() : nullptr;
    const size_t mask_batch = m ? in[3].get_shape()[0] : 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> out(out_batch * num_heads * s.seq_q * D, 0.0f);
    std::vector<float> p(s.seq_kv);
    for (size_t b = 0; b < out_batch; ++b) {
        for (size_t h = 0; h < num_heads; ++h) {
            const auto* qh = q + ((b % s.q_batch) * num_heads + h) * s.seq_q * D;
            const auto* kh = k + ((b % s.k_batch) * num_heads + h) * s.seq_kv * D;
            const auto* vh = v + ((b % s.v_batch) * num_heads + h) * s.seq_kv * D;
            for (size_t i = 0; i < s.seq_q; ++i) {
                float max_logit = -std::numeric_limits<float>::infinity();
                for (size_t j = 0; j < s.seq_kv; ++j) {
                    float dot = 0.0f;
                    for (size_t d = 0; d < D; ++d) {
                        dot += static_cast<float>(qh[i * D + d]) * static_cast<float>(kh[j * D + d]);
                    }
                    p[j] = dot * scale + (m ? static_cast<float>(m[((b % mask_batch) * s.seq_q + i) * s.seq_kv + j]) : 0.0f);
                    if (causal && j > i) {
                        p[j] = -std::numeric_limits<float>::infinity();
                    }
                    max_logit = std::max(max_logit, p[j]);
                }
                float sum = 0.0f;
                for (size_t j = 0; j < s.seq_kv; ++j) {
                    p[j] = std::exp(p[j] - max_logit);
                    sum += p[j];
                }
                float* o = out.data() + ((b * num_heads + h) * s.seq_q + i) * D;
                for (size_t j = 0; j < s.seq_kv; ++j) {
                    for (size_t d = 0; d < D; ++d) {
                        o[d] += p[j] / sum * static_cast<float>(vh[j * D + d]);
                    }
                }
            }
        }
    }
    return out;
}

class SDPABatchBroadcastTest : public ::testing::TestWithParam<SDPABroadcastParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<SDPABroadcastParams>& info) {
        const auto& p = info.param;
        const char* names[] = {"plain", "mask", "causal"};
        return "D" + std::to_string(p.head_size) + "_" + names[static_cast<int>(p.variant)];
    }
};

TEST_P(SDPABatchBroadcastTest, MatchesReference) {
    const auto& p = GetParam();
    auto core = ov::test::utils::PluginCache::get().core();
    const auto model = make_sdpa_model(p);
    auto gpu_model = core->compile_model(model, ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto gpu = gpu_model.create_infer_request();

    std::mt19937 gen(static_cast<uint32_t>(p.head_size * 10 + static_cast<size_t>(p.variant)));
    // Whisper beam-search decode and a chunk (K/V batch 1, encoder length 1500), Q-only, K-only and V-only broadcast,
    // then equal batches.
    const std::vector<Step> steps = {{3, 1, 1, 1, 1500}, {3, 1, 1, 64, 1500}, {1, 2, 2, 1, 256}, {2, 1, 2, 16, 256}, {2, 2, 1, 16, 256}, {2, 2, 2, 16, 256}};
    for (const auto& s : steps) {
        SCOPED_TRACE("batch Q/K/V " + std::to_string(s.q_batch) + "/" + std::to_string(s.k_batch) + "/" + std::to_string(s.v_batch) + ", L " +
                     std::to_string(s.seq_q) + ", S " + std::to_string(s.seq_kv));
        const size_t D = p.head_size;
        const size_t out_batch = std::max({s.q_batch, s.k_batch, s.v_batch});
        // A wide Q range makes the softmax peaky, so a Q or K row from the wrong batch shows; V is offset per batch.
        std::vector<ov::Tensor> inputs;
        inputs.reserve(4);
        inputs.push_back(random_tensor({s.q_batch, num_heads, s.seq_q, D}, gen, -4.0f, 4.0f));
        inputs.push_back(random_tensor({s.k_batch, num_heads, s.seq_kv, D}, gen, -1.0f, 1.0f));
        inputs.push_back(random_tensor({s.v_batch, num_heads, s.seq_kv, D}, gen, 0.25f, 0.75f, 0.5f));
        if (p.variant == Variant::MASK) {
            inputs.push_back(random_tensor({out_batch, 1, s.seq_q, s.seq_kv}, gen, -2.0f, 0.0f));
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            gpu.set_input_tensor(i, inputs[i]);
        }
        gpu.infer();

        const auto actual = gpu.get_output_tensor(0);
        ASSERT_EQ(actual.get_shape(), (ov::Shape{out_batch, num_heads, s.seq_q, D}));
        ASSERT_EQ(actual.get_element_type(), ov::element::f16);
        const auto e = reference(inputs, s, D, p.variant == Variant::CAUSAL);
        ASSERT_EQ(e.size(), actual.get_size());
        // f16 inputs are exact in the reference, so this is the kernel's f16 compute error (about 1e-3).
        const float tolerance = 0.01f;
        const auto* a = actual.data<const ov::float16>();
        size_t mismatches = 0;
        float max_diff = 0.0f;
        size_t first_mismatch = 0;
        for (size_t i = 0; i < e.size(); ++i) {
            const float diff = std::fabs(e[i] - static_cast<float>(a[i]));
            if (!(diff <= tolerance)) {  // NaN counts as a mismatch
                if (mismatches++ == 0) {
                    first_mismatch = i;
                }
            }
            if (diff > max_diff || std::isnan(diff)) {
                max_diff = diff;
            }
        }
        EXPECT_EQ(mismatches, 0u) << mismatches << " of " << e.size() << " elements off by more than " << tolerance << ", max diff " << max_diff
                                  << ", first at " << first_mismatch << " (expected " << e[first_mismatch] << ", actual "
                                  << static_cast<float>(a[first_mismatch]) << ")";
    }
}

std::vector<SDPABroadcastParams> make_params() {
    std::vector<SDPABroadcastParams> params;
    for (const size_t head_size : {64, 128}) {
        for (const auto variant : {Variant::PLAIN, Variant::MASK, Variant::CAUSAL}) {
            params.push_back({head_size, variant});
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_SDPABatchBroadcast, SDPABatchBroadcastTest, ::testing::ValuesIn(make_params()), SDPABatchBroadcastTest::get_test_case_name);

}  // namespace
