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

// A plain SDPA with a `-inf` attention mask whose first 128 or more keys are masked for a query (left padding, a
// block-diagonal mask) and large logits (|q.k / sqrt(d)| above about 100). Guards a NaN output of sdpa_ocl: a masked key
// tile saturates the running maximum, and the exp2 argument of a later tile must not come out positive. Q/K/V are
// non-negative so the logits are large, and the mask has `-inf` or, as the control, the f16 lowest (-65504). The oracle is an
// fp32 attention over the same f16 inputs. A softmax weights V convexly, so every output lies in the range of V: that holds
// whatever the logit rounding of the kernel is (f16 logits near 100 are 0.06 apart), and a NaN or an inf cannot pass it; the
// comparison with the reference is loose for the same reason.
namespace {

enum class MaskKind { LEFT_PAD_INF, BLOCK_DIAG_INF, LEFT_PAD_FINITE };

struct SDPAInfMaskParams {
    size_t head_size;
    MaskKind mask;
    bool dynamic;
};

constexpr size_t num_heads = 2;
constexpr size_t seq_len = 256;
constexpr float data_max = 6.0f;  // Q, K, V uniform in [0, 6]: |q.k / sqrt(d)| reaches about 104 at head 64 and 135 at head 128

std::shared_ptr<ov::Model> make_sdpa_model(const SDPAInfMaskParams& p) {
    const auto f16 = ov::element::f16;
    const auto E = static_cast<int64_t>(p.head_size);
    const auto H = static_cast<int64_t>(num_heads);
    const auto L = p.dynamic ? ov::Dimension::dynamic() : ov::Dimension(static_cast<int64_t>(seq_len));
    auto query = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, H, L, E});
    auto key = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, H, L, E});
    auto value = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, H, L, E});
    auto mask = std::make_shared<ov::op::v0::Parameter>(f16, ov::PartialShape{1, 1, L, L});
    auto sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(query, key, value, mask, false);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(sdpa)}, ov::ParameterVector{query, key, value, mask});
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

ov::Tensor make_mask(MaskKind kind) {
    ov::Tensor mask(ov::element::f16, ov::Shape{1, 1, seq_len, seq_len});
    auto* data = mask.data<ov::float16>();
    const float masked = kind == MaskKind::LEFT_PAD_FINITE ? -65504.0f : -std::numeric_limits<float>::infinity();
    for (size_t row = 0; row < seq_len; ++row) {
        for (size_t col = 0; col < seq_len; ++col) {
            const bool visible = kind == MaskKind::BLOCK_DIAG_INF ? (row / (seq_len / 4) == col / (seq_len / 4)) : (col >= seq_len / 2);
            data[row * seq_len + col] = ov::float16(visible ? 0.0f : masked);
        }
    }
    return mask;
}

// fp32 attention, the mask added to the scaled logits; a masked entry weighs 0 (exp(-inf), and exp(-65504 - max) underflows to 0).
std::vector<float> reference(const std::vector<ov::Tensor>& in, size_t E) {
    const auto* q = in[0].data<const ov::float16>();
    const auto* k = in[1].data<const ov::float16>();
    const auto* v = in[2].data<const ov::float16>();
    const auto* m = in[3].data<const ov::float16>();
    const float scale = 1.0f / std::sqrt(static_cast<float>(E));
    std::vector<float> out(num_heads * seq_len * E, 0.0f);
    std::vector<float> p(seq_len);
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < seq_len; ++i) {
            float max_logit = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (size_t d = 0; d < E; ++d) {
                    dot += static_cast<float>(q[(h * seq_len + i) * E + d]) * static_cast<float>(k[(h * seq_len + j) * E + d]);
                }
                p[j] = dot * scale + static_cast<float>(m[i * seq_len + j]);
                max_logit = std::max(max_logit, p[j]);
            }
            float sum = 0.0f;
            for (size_t j = 0; j < seq_len; ++j) {
                p[j] = std::exp(p[j] - max_logit);
                sum += p[j];
            }
            float* o = out.data() + (h * seq_len + i) * E;
            for (size_t j = 0; j < seq_len; ++j) {
                for (size_t d = 0; d < E; ++d) {
                    o[d] += p[j] / sum * static_cast<float>(v[(h * seq_len + j) * E + d]);
                }
            }
        }
    }
    return out;
}

class SDPAInfMaskTest : public ::testing::TestWithParam<SDPAInfMaskParams> {
public:
    static std::string get_test_case_name(const ::testing::TestParamInfo<SDPAInfMaskParams>& info) {
        const auto& p = info.param;
        const char* names[] = {"left_pad_inf", "block_diag_inf", "left_pad_finite"};
        return "E" + std::to_string(p.head_size) + "_" + names[static_cast<int>(p.mask)] + (p.dynamic ? "_dynamic" : "_static");
    }
};

TEST_P(SDPAInfMaskTest, FiniteAndWithinTheRangeOfV) {
    const auto& p = GetParam();
    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled = core->compile_model(make_sdpa_model(p), ov::test::utils::DEVICE_GPU, ov::hint::inference_precision(ov::element::f16));
    auto gpu = compiled.create_infer_request();

    std::mt19937 gen(static_cast<uint32_t>(p.head_size * 10 + static_cast<size_t>(p.mask)));
    const size_t E = p.head_size;
    std::vector<ov::Tensor> inputs;
    for (size_t i = 0; i < 3; ++i) {
        inputs.push_back(random_tensor({1, num_heads, seq_len, E}, gen, 0.0f, data_max));
    }
    inputs.push_back(make_mask(p.mask));
    for (size_t i = 0; i < inputs.size(); ++i) {
        gpu.set_input_tensor(i, inputs[i]);
    }
    gpu.infer();

    const auto actual = gpu.get_output_tensor(0);
    ASSERT_EQ(actual.get_shape(), (ov::Shape{1, num_heads, seq_len, E}));
    const auto expected = reference(inputs, E);
    const auto* a = actual.data<const ov::float16>();
    size_t not_finite = 0, out_of_range = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < expected.size(); ++i) {
        const float x = static_cast<float>(a[i]);
        if (!std::isfinite(x)) {
            ++not_finite;
            continue;
        }
        // V is in [0, data_max]; the slack is the f16 rounding of the weights and of the output.
        if (x < -0.05f || x > data_max + 0.05f) {
            ++out_of_range;
        }
        max_diff = std::max(max_diff, std::fabs(x - expected[i]));
    }
    EXPECT_EQ(not_finite, 0u) << not_finite << " of " << expected.size() << " outputs are NaN or inf";
    EXPECT_EQ(out_of_range, 0u) << out_of_range << " outputs outside the range of V";
    // Loose, see the header comment: the kernel's f16 logits near 100 and the weights' rounding move an element by a few tenths.
    EXPECT_LE(max_diff, 1.0f) << "max diff to the fp32 reference " << max_diff;
}

std::vector<SDPAInfMaskParams> make_params() {
    std::vector<SDPAInfMaskParams> params;
    for (const bool dynamic : {false, true}) {
        for (const size_t head_size : {64, 128}) {
            for (const auto mask : {MaskKind::LEFT_PAD_INF, MaskKind::BLOCK_DIAG_INF, MaskKind::LEFT_PAD_FINITE}) {
                params.push_back({head_size, mask, dynamic});
            }
        }
    }
    return params;
}

INSTANTIATE_TEST_SUITE_P(smoke_SDPAInfMask, SDPAInfMaskTest, ::testing::ValuesIn(make_params()), SDPAInfMaskTest::get_test_case_name);

}  // namespace
