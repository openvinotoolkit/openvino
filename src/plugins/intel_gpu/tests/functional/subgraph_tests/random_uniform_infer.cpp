// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/reference/random_uniform.hpp"
#include "openvino/runtime/core.hpp"

namespace {
// RandomUniform with constant inputs must not be folded into a constant by the GPU plugin, and every
// inference must produce a new sequence (the Philox state advances like in the CPU plugin).
using RandomUniformInferParams = std::tuple<uint64_t /* global_seed */, uint64_t /* op_seed */>;

class RandomUniformInferGPUTest : public ::testing::TestWithParam<RandomUniformInferParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<RandomUniformInferParams>& obj) {
        return "global_seed=" + std::to_string(std::get<0>(obj.param)) + "_op_seed=" + std::to_string(std::get<1>(obj.param));
    }

protected:
    const ov::Shape out_shape{1, 50, 32};

    std::shared_ptr<ov::Model> make_model(uint64_t global_seed, uint64_t op_seed) const {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, out_shape);
        auto shape = ov::op::v0::Constant::create(ov::element::i64, {out_shape.size()}, out_shape);
        auto min_val = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
        auto max_val = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
        auto rnd = std::make_shared<ov::op::v8::RandomUniform>(shape, min_val, max_val, ov::element::f32, global_seed, op_seed);
        auto add = std::make_shared<ov::op::v1::Add>(x, rnd);
        return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x});
    }

    static std::vector<float> infer(ov::InferRequest& req, const ov::Tensor& input) {
        req.set_input_tensor(input);
        req.infer();
        auto out = req.get_output_tensor();
        return std::vector<float>(out.data<float>(), out.data<float>() + out.get_size());
    }
};

TEST_P(RandomUniformInferGPUTest, NewSequenceOnEveryInference) {
    const auto [global_seed, op_seed] = GetParam();
    ov::Core core;
    auto compiled = core.compile_model(make_model(global_seed, op_seed), ov::test::utils::DEVICE_GPU);

    ov::Tensor input(ov::element::f32, out_shape);
    std::fill_n(input.data<float>(), input.get_size(), 0.0f);

    const size_t iterations = 3;
    auto req = compiled.create_infer_request();
    std::vector<std::vector<float>> results;
    for (size_t i = 0; i < iterations; ++i) {
        results.push_back(infer(req, input));
    }
    for (size_t i = 1; i < iterations; ++i) {
        ASSERT_NE(results[i - 1], results[i]) << "RandomUniform returned the same values on inference " << i;
    }

    if (global_seed != 0 || op_seed != 0) {
        // Deterministic seeds: each inference must match the sequence produced by the CPU plugin.
        std::vector<uint64_t> out_shape_u64(out_shape.begin(), out_shape.end());
        const float min_v = 0.0f, max_v = 1.0f;
        std::pair<uint64_t, uint64_t> state{0, 0};
        for (size_t it = 0; it < iterations; ++it) {
            std::vector<float> expected(ov::shape_size(out_shape));
            ov::reference::random_uniform(out_shape_u64.data(),
                                          reinterpret_cast<const char*>(&min_v),
                                          reinterpret_cast<const char*>(&max_v),
                                          reinterpret_cast<char*>(expected.data()),
                                          ov::Shape{out_shape.size()},
                                          ov::element::f32,
                                          global_seed,
                                          op_seed,
                                          state);
            // Same state progression as the CPU plugin: n += 256 * <number of output elements>.
            const uint64_t skip_count = ov::shape_size(out_shape) * 256;
            state.first += skip_count;
            if (state.first < skip_count) {
                state.second++;
            }
            for (size_t i = 0; i < expected.size(); ++i) {
                ASSERT_NEAR(expected[i], results[it][i], 1e-6f) << "inference " << it << ", i = " << i;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_RandomUniformInfer,
                         RandomUniformInferGPUTest,
                         ::testing::Values(RandomUniformInferParams{0, 0}, RandomUniformInferParams{80, 100}),
                         RandomUniformInferGPUTest::getTestCaseName);

// Dynamic input with RandomUniform shape taken from ShapeOf(input), as produced by torch.randn_like conversion.
// RandomUniform must not be skipped as a part of a shape_of subgraph when the input shape does not change.
TEST(smoke_RandomUniformDynamicInferGPUTest, ShapeOfInputNewSequenceOnEveryInference) {
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(x);
    auto min_val = ov::op::v0::Constant::create(ov::element::f32, {}, {0.0f});
    auto max_val = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
    auto rnd = std::make_shared<ov::op::v8::RandomUniform>(shape, min_val, max_val, ov::element::f32, 0, 0);
    auto add = std::make_shared<ov::op::v1::Add>(x, rnd);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{x});

    ov::Core core;
    auto req = core.compile_model(model, ov::test::utils::DEVICE_GPU).create_infer_request();
    ov::Tensor input(ov::element::f32, ov::Shape{1, 50, 32});
    std::fill_n(input.data<float>(), input.get_size(), 0.0f);

    std::vector<std::vector<float>> results;
    for (size_t i = 0; i < 3; ++i) {
        req.set_input_tensor(input);
        req.infer();
        auto out = req.get_output_tensor();
        results.emplace_back(out.data<float>(), out.data<float>() + out.get_size());
    }
    ASSERT_NE(results[0], results[1]);
    ASSERT_NE(results[1], results[2]);
}
}  // namespace
