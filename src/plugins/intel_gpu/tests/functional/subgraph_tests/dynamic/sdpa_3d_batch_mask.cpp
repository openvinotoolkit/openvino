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

// Rank-3 ScaledDotProductAttention (Q/K/V [B, L, E]) with an attention mask or without one, executed end to end and compared
// with an fp32 reference over the same f16 inputs. Dynamic: batch and sequence lengths are dynamic in one compiled model, the
// op is not decomposed and reaches the fused kernel. Static: one compiled model per shape; an XMX device decomposes it
// (should_decompose_sdpa_for_memory_size()) and a device without XMX runs the fused kernel, so the static variants check the
// numbers on whichever path the device takes.
//
// The rank-3 mask [B, Lq, Lk] is aligned with Q/K/V, which the kernel sees as [B, 1, L, E]. Guards canonicalizing it to
// [1, B, Lq, Lk], which makes the kernel read B as the head count and use mask[0] for every batch. The other 3D tests use a
// batch-1 mask or the base generator's mask data (period 2), so neither can see it: this test draws every mask element
// independently. Batch-1 and rank-2 masks (numpy-broadcast over the batch) and no mask are the controls.
namespace {

enum class MaskKind { PER_BATCH, BATCH1, RANK2, NONE };

struct SDPA3DMaskParams {
    size_t head_size;
    MaskKind mask;
    bool dynamic;
};

struct Step {
    size_t batch, seq_q, seq_kv;
};

std::shared_ptr<ov::Model> make_sdpa_model(const SDPA3DMaskParams& p, const Step& s) {
    const auto f16 = ov::element::f16;
    const auto D = static_cast<int64_t>(p.head_size);
    const auto dim = [&](size_t value) {
        return p.dynamic ? ov::Dimension::dynamic() : ov::Dimension(static_cast<int64_t>(value));
    };
    const auto mask_batch = p.mask == MaskKind::PER_BATCH ? dim(s.batch) : ov::Dimension(p.dynamic ? -1 : 1);
    auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{dim(s.batch), dim(s.seq_q), D});
    auto key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{dim(s.batch), dim(s.seq_kv), D});
    auto value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{dim(s.batch), dim(s.seq_kv), D});
    ov::ParameterVector params{query, key, value};
    std::shared_ptr<ov::Node> sdpa;
    if (p.mask == MaskKind::NONE) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, false);
    } else {
        auto mask = std::make_shared<ov::op::v0::Parameter>(
            f16,
            p.mask == MaskKind::RANK2 ? ov::PartialShape{dim(s.seq_q), dim(s.seq_kv)} : ov::PartialShape{mask_batch, dim(s.seq_q), dim(s.seq_kv)});
        params.push_back(mask);
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, false);
    }
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa)}, params);
}

ov::Tensor random_tensor(const ov::Shape& shape, std::mt19937& gen, float lo, float hi) {
    ov::Tensor tensor(ov::element::f16, shape);
    std::uniform_real_distribution<float> dist(lo, hi);
    auto* data = tensor.data<ov::float16>();
    for (size_t i = 0; i < tensor.get_size(); ++i) {
        data[i] = ov::float16(dist(gen));
    }
    return tensor;
}

// fp32 attention, opset13 semantics: the mask is [B, Lq, Lk] (PER_BATCH), [1, Lq, Lk] (BATCH1), [Lq, Lk] (RANK2) or absent (NONE).
std::vector<float> reference(const std::vector<ov::Tensor>& in, const Step& s, size_t D, MaskKind mask_kind) {
    const auto* q = in[0].data<const ov::float16>();
    const auto* k = in[1].data<const ov::float16>();
    const auto* v = in[2].data<const ov::float16>();
    const auto* m = mask_kind == MaskKind::NONE ? nullptr : in[3].data<const ov::float16>();
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> out(s.batch * s.seq_q * D, 0.0f);
    std::vector<float> p(s.seq_kv);
    for (size_t b = 0; b < s.batch; ++b) {
        const auto* mb = m ? m + (mask_kind == MaskKind::PER_BATCH ? b * s.seq_q * s.seq_kv : 0) : nullptr;
        for (size_t i = 0; i < s.seq_q; ++i) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < s.seq_kv; ++j) {
                float dot = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    dot += static_cast<float>(q[(b * s.seq_q + i) * D + d]) * static_cast<float>(k[(b * s.seq_kv + j) * D + d]);
                }
                p[j] = dot * scale + (mb ? static_cast<float>(mb[i * s.seq_kv + j]) : 0.0f);
                max_logit = std::max(max_logit, p[j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < s.seq_kv; ++j) {
                p[j] = std::exp(p[j] - max_logit);
                sum += p[j];
            }
            float* o = out.data() + (b * s.seq_q + i) * D;
            for (size_t j = 0; j < s.seq_kv; ++j) {
                for (size_t d = 0; d < D; ++d) {
                    o[d] += p[j] / sum * static_cast<float>(v[(b * s.seq_kv + j) * D + d]);
                }
            }
        }
    }
    return out;
}

size_t count_fused_sdpa(const ov::CompiledModel& compiled) {
    size_t count = 0;
    for (const auto& op : compiled.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = op->get_rt_info();
        const auto it = rt_info.find("layerType");
        if (it != rt_info.end() && it->second.as<std::string>() == "scaled_dot_product_attention") {
            ++count;
        }
    }
    return count;
}

class SDPA3DBatchMaskTest : public ::testing::TestWithParam<SDPA3DMaskParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<SDPA3DMaskParams>& info) {
        const auto& p = info.param;
        const char* names[] = {"per_batch_mask", "batch1_mask", "rank2_mask", "no_mask"};
        return "D" + std::to_string(p.head_size) + "_" + names[static_cast<int>(p.mask)] + (p.dynamic ? "_dynamic" : "_static");
    }
};

TEST_P(SDPA3DBatchMaskTest, MatchesReference) {
    const auto& p = GetParam();
    auto core = ov::test::utils::PluginCache::get().core();
    // The shape that exposes the mask bug first, then equal lengths, a single-token step, batch 1 (the control), and an odd length.
    const std::vector<Step> steps = {{10, 1024, 77}, {4, 64, 64}, {3, 1, 200}, {1, 128, 128}, {2, 33, 77}};

    ov::CompiledModel gpu_model;
    ov::InferRequest gpu;
    const auto compile = [&](const Step& s) {
        gpu_model = core->compile_model(make_sdpa_model(p, s), ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
        gpu = gpu_model.create_infer_request();
    };
    if (p.dynamic) {
        compile(steps.front());
    }

    std::mt19937 gen(static_cast<uint32_t>(p.head_size * 10 + static_cast<size_t>(p.mask)));
    bool checked_fused = false;
    for (const auto& s : steps) {
        SCOPED_TRACE("B " + std::to_string(s.batch) + ", Lq " + std::to_string(s.seq_q) + ", Lk " + std::to_string(s.seq_kv));
        if (!p.dynamic) {
            compile(s);
        }
        const size_t D = p.head_size;
        std::vector<ov::Tensor> inputs;
        inputs.reserve(4);
        inputs.push_back(random_tensor({s.batch, s.seq_q, D}, gen, -1.0f, 1.0f));
        inputs.push_back(random_tensor({s.batch, s.seq_kv, D}, gen, -1.0f, 1.0f));
        inputs.push_back(random_tensor({s.batch, s.seq_kv, D}, gen, -1.0f, 1.0f));
        // Every element independent, wide range: the mask decides the softmax, so a mask row of another batch shows.
        switch (p.mask) {
        case MaskKind::PER_BATCH:
            inputs.push_back(random_tensor({s.batch, s.seq_q, s.seq_kv}, gen, -3.0f, 0.0f));
            break;
        case MaskKind::BATCH1:
            inputs.push_back(random_tensor({1, s.seq_q, s.seq_kv}, gen, -3.0f, 0.0f));
            break;
        case MaskKind::RANK2:
            inputs.push_back(random_tensor({s.seq_q, s.seq_kv}, gen, -3.0f, 0.0f));
            break;
        case MaskKind::NONE:
            break;
        }
        for (size_t i = 0; i < inputs.size(); ++i) {
            gpu.set_input_tensor(i, inputs[i]);
        }
        gpu.infer();

        if (p.dynamic && !checked_fused) {
            // A decomposed SDPA would pass without touching the kernel this test is about.
            EXPECT_EQ(count_fused_sdpa(gpu_model), 1u) << "SDPA is not fused: the test no longer exercises the SDPA kernel";
            checked_fused = true;
        }

        const auto actual = gpu.get_output_tensor(0);
        ASSERT_EQ(actual.get_shape(), (ov::Shape{s.batch, s.seq_q, D}));
        ASSERT_EQ(actual.get_element_type(), ov::element::f16);
        const auto e = reference(inputs, s, D, p.mask);
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
                                  << ", first at batch " << first_mismatch / (s.seq_q * D) << " (expected " << e[first_mismatch] << ", actual "
                                  << static_cast<float>(a[first_mismatch]) << ")";
    }
}

std::vector<SDPA3DMaskParams> make_params() {
    std::vector<SDPA3DMaskParams> params;
    for (const bool dynamic : {true, false}) {
        for (const size_t head_size : {64, 128}) {
            for (const auto mask : {MaskKind::PER_BATCH, MaskKind::BATCH1, MaskKind::RANK2, MaskKind::NONE}) {
                params.push_back({head_size, mask, dynamic});
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_SDPA3DBatchMask, SDPA3DBatchMaskTest, ::testing::ValuesIn(make_params()), SDPA3DBatchMaskTest::get_test_case_name);

}  // namespace
