// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace {

// Scale/ZP shared by two FullyConnectedCompressed (optionally the scale also by a Multiply), checked against a host reference.
using DecompressionParamsTestParams = std::tuple<ov::element::Type,  // activations and scale
                                                 ov::element::Type,  // weights
                                                 ov::element::Type,  // zero point (dynamic: none)
                                                 bool,               // scale / zero point [N, groups] (else [groups, N])
                                                 bool>;              // scale also read by a Multiply

class fully_connected_decompression_params : public testing::TestWithParam<DecompressionParamsTestParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<DecompressionParamsTestParams>& info) {
        const auto& [act, wei, zp, n_major, other_user] = info.param;
        std::ostringstream name;
        name << "act_" << act << "_wei_" << wei << "_zp_" << (zp == ov::element::dynamic ? std::string("none") : zp.get_type_name())
             << (n_major ? "_n_groups" : "_groups_n") << (other_user ? "_with_multiply" : "");
        return name.str();
    }
};

constexpr size_t K = 256, N = 64, group_size = 32, groups = K / group_size;

// Constant of `type` with `values` (integers, or reals for f16 / f32); u4 packs element 2i in the low nibble.
std::shared_ptr<ov::op::v0::Constant> make_constant(ov::element::Type type, const ov::Shape& shape, const std::vector<float>& values) {
    ov::Tensor tensor(type, shape);
    if (type == ov::element::u4) {
        auto* bytes = static_cast<uint8_t*>(tensor.data());
        std::fill_n(bytes, tensor.get_byte_size(), 0);
        for (size_t i = 0; i < values.size(); ++i)
            bytes[i / 2] |= static_cast<uint8_t>(static_cast<uint8_t>(values[i]) << ((i % 2) * 4));
    } else if (type == ov::element::u8) {
        std::transform(values.begin(), values.end(), tensor.data<uint8_t>(), [](float v) { return static_cast<uint8_t>(v); });
    } else if (type == ov::element::f16) {
        std::transform(values.begin(), values.end(), tensor.data<ov::float16>(), [](float v) { return ov::float16(v); });
    } else {
        std::copy(values.begin(), values.end(), tensor.data<float>());
    }
    return std::make_shared<ov::op::v0::Constant>(tensor);
}

// Values exactly representable in `type` (rounded through f16 when the type is f16).
std::vector<float> random_values(size_t count, float lo, float hi, ov::element::Type type, std::mt19937& gen, bool integers) {
    std::vector<float> values(count);
    std::uniform_real_distribution<float> dist(lo, hi);
    for (auto& v : values) {
        v = integers ? std::floor(dist(gen)) : dist(gen);
        if (type == ov::element::f16)
            v = static_cast<float>(ov::float16(v));
    }
    return values;
}

TEST_P(fully_connected_decompression_params, shared_constants) {
    // Not structured bindings: captured by the lambdas below
    const ov::element::Type act = std::get<0>(GetParam()), wei = std::get<1>(GetParam()), zp_type = std::get<2>(GetParam());
    const bool n_major = std::get<3>(GetParam()), other_user = std::get<4>(GetParam());
    const bool with_zp = zp_type != ov::element::dynamic;
    const float q_max = wei == ov::element::u4 ? 16.0f : 256.0f;
    std::mt19937 gen(42);

    // param[n][g] is stored at n * groups + g ([N, groups]) or g * N + n ([groups, N])
    auto param_index = [n_major](size_t n, size_t g) { return n_major ? n * groups + g : g * N + n; };
    const ov::Shape param_shape = n_major ? ov::Shape{N, groups} : ov::Shape{groups, N};
    const auto scale_values = random_values(N * groups, 0.001f, 0.02f, act, gen, false);
    const float zp_max = zp_type == ov::element::u4 ? 16.0f : q_max;  // a zero point fits its own type
    const auto zp_values = random_values(N * groups, 0.0f, zp_max, ov::element::f32, gen, true);
    auto scale = make_constant(act, param_shape, scale_values);
    auto zp = with_zp ? make_constant(zp_type, param_shape, zp_values) : nullptr;

    ov::ParameterVector params;
    ov::OutputVector outputs;
    std::vector<std::vector<float>> weight_values;
    for (int i = 0; i < 2; ++i) {
        weight_values.push_back(random_values(N * K, 0.0f, q_max, ov::element::f32, gen, true));
        auto x = std::make_shared<ov::op::v0::Parameter>(act, ov::PartialShape{-1, static_cast<int64_t>(K)});
        auto w = make_constant(wei, {N, K}, weight_values.back());
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        std::shared_ptr<ov::Node> fc;
        if (with_zp)
            fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(x, w, no_bias, scale, zp, act, true, n_major, n_major);
        else
            fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(x, w, no_bias, scale, act, true, n_major, n_major);
        params.push_back(x);
        outputs.push_back(fc);
    }
    if (other_user) {
        auto p = std::make_shared<ov::op::v0::Parameter>(act, param_shape);
        params.push_back(p);
        outputs.push_back(std::make_shared<ov::op::v1::Multiply>(p, scale));
    }
    auto model = std::make_shared<ov::Model>(outputs, params);

    ov::Core core;
    auto compiled = core.compile_model(model, "GPU", ov::hint::inference_precision(act), ov::hint::dynamic_quantization_group_size(0));
    auto request = compiled.create_infer_request();

    auto fill = [&](ov::Tensor& t, const std::vector<float>& v) {
        if (act == ov::element::f16)
            std::transform(v.begin(), v.end(), t.data<ov::float16>(), [](float f) { return ov::float16(f); });
        else
            std::copy(v.begin(), v.end(), t.data<float>());
    };
    auto read = [&](const ov::Tensor& t, size_t i) {
        return act == ov::element::f16 ? static_cast<float>(t.data<const ov::float16>()[i]) : t.data<const float>()[i];
    };

    for (size_t M : std::initializer_list<size_t>{1, 7}) {
        std::vector<std::vector<float>> x_values;
        for (int i = 0; i < 2; ++i) {
            x_values.push_back(random_values(M * K, -1.0f, 1.0f, act, gen, false));
            ov::Tensor x(act, {M, K});
            fill(x, x_values.back());
            request.set_input_tensor(i, x);
        }
        std::vector<float> p_values;
        if (other_user) {
            p_values = random_values(N * groups, -2.0f, 2.0f, act, gen, false);
            ov::Tensor p(act, param_shape);
            fill(p, p_values);
            request.set_input_tensor(2, p);
        }
        request.infer();

        for (int i = 0; i < 2; ++i) {
            auto y = request.get_output_tensor(i);
            ASSERT_EQ(y.get_shape(), (ov::Shape{M, N}));
            for (size_t m = 0; m < M; ++m) {
                for (size_t n = 0; n < N; ++n) {
                    double ref = 0.0;
                    for (size_t k = 0; k < K; ++k) {
                        const size_t idx = param_index(n, k / group_size);
                        const double w = weight_values[i][n * K + k] - (with_zp ? zp_values[idx] : 0.0f);
                        ref += static_cast<double>(x_values[i][m * K + k]) * w * scale_values[idx];
                    }
                    const float tol = 0.05f + 0.01f * static_cast<float>(std::fabs(ref));
                    ASSERT_NEAR(read(y, m * N + n), ref, tol) << "fc " << i << " M=" << M << " m=" << m << " n=" << n;
                }
            }
        }
        if (other_user) {
            auto y = request.get_output_tensor(2);
            ASSERT_EQ(y.get_shape(), param_shape);
            for (size_t j = 0; j < N * groups; ++j) {
                const float ref = p_values[j] * scale_values[j];
                ASSERT_NEAR(read(y, j), ref, 1e-3f + 1e-2f * std::fabs(ref)) << "multiply index " << j;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         fully_connected_decompression_params,
                         testing::Combine(testing::Values(ov::element::f16, ov::element::f32),
                                          testing::Values(ov::element::u8, ov::element::u4),
                                          testing::Values(ov::element::dynamic, ov::element::u8, ov::element::u4, ov::element::f16),
                                          testing::Bool(),
                                          testing::Bool()),
                         fully_connected_decompression_params::get_test_case_name);

}  // namespace
