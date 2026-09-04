// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/minimum_maximum.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {
using ov::test::MaxMinLayerTest;

const std::vector<std::vector<ov::Shape>> inShapes = {
        {{2}, {1}},
        {{1, 1, 1, 3}, {1}},
        {{1, 2, 4}, {1}},
        {{1, 4, 4}, {1}},
        {{1, 4, 4, 1}, {1}},
        {{256, 56}, {256, 56}},
        {{8, 1, 6, 1}, {7, 1, 5}},
};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
};

const std::vector<ov::test::utils::MinMaxOpType> opType = {
        ov::test::utils::MinMaxOpType::MINIMUM,
        ov::test::utils::MinMaxOpType::MAXIMUM,
};

const std::vector<ov::test::utils::InputLayerType> second_inputType = {
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum, MaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(second_inputType),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        MaxMinLayerTest::getTestCaseName);

template <typename OpType>
void compile_int_extremum(ov::element::Type precision) {
    const auto input0 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{4, 4});
    const auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{4, 4});
    const auto extremum = std::make_shared<OpType>(input0, input1);
    const auto result = std::make_shared<ov::op::v0::Result>(extremum);
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});

    ov::Core core;
    OV_ASSERT_NO_THROW(core.compile_model(model, ov::test::utils::DEVICE_GPU));
}

TEST(smoke_IntMinimum, CompileModel) {
    compile_int_extremum<ov::op::v1::Minimum>(ov::element::u8);
    compile_int_extremum<ov::op::v1::Minimum>(ov::element::u16);
}

TEST(smoke_IntMaximum, CompileModel) {
    compile_int_extremum<ov::op::v1::Maximum>(ov::element::u8);
    compile_int_extremum<ov::op::v1::Maximum>(ov::element::u16);
}

template <typename OpType>
void run_nan_propagation_test(ov::element::Type precision) {
    const auto param_a = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{4});
    const auto param_b = std::make_shared<ov::op::v0::Parameter>(precision, ov::Shape{4});
    const auto op = std::make_shared<OpType>(param_a, param_b);
    const auto result = std::make_shared<ov::op::v0::Result>(op);
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_a, param_b});

    ov::Core core;
    auto compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU);
    auto infer_req = compiled.create_infer_request();

    const float qnan = std::numeric_limits<float>::quiet_NaN();

    auto make_tensor = [&](float v0, float v1, float v2, float v3) -> ov::Tensor {
        if (precision == ov::element::f16) {
            std::vector<ov::float16> data = {ov::float16(v0), ov::float16(v1), ov::float16(v2), ov::float16(v3)};
            ov::Tensor t(precision, ov::Shape{4});
            std::memcpy(t.data(), data.data(), data.size() * sizeof(ov::float16));
            return t;
        }
        std::vector<float> data = {v0, v1, v2, v3};
        ov::Tensor t(precision, ov::Shape{4});
        std::memcpy(t.data(), data.data(), data.size() * sizeof(float));
        return t;
    };

    infer_req.set_tensor(param_a, make_tensor(qnan, 1.f, qnan, 5.f));
    infer_req.set_tensor(param_b, make_tensor(2.f, qnan, 3.f, 4.f));
    infer_req.infer();

    auto output = infer_req.get_output_tensor(0);

    auto get_float = [&](size_t idx) -> float {
        return (precision == ov::element::f16)
            ? static_cast<float>(output.data<ov::float16>()[idx])
            : output.data<float>()[idx];
    };

    constexpr bool is_min = std::is_same_v<OpType, ov::op::v1::Minimum>;
    EXPECT_TRUE(std::isnan(get_float(0))) << "op(NaN, 2.0) should be NaN";
    EXPECT_TRUE(std::isnan(get_float(1))) << "op(1.0, NaN) should be NaN";
    EXPECT_TRUE(std::isnan(get_float(2))) << "op(NaN, 3.0) should be NaN";
    EXPECT_FLOAT_EQ(get_float(3), is_min ? 4.f : 5.f) << "op(5.0, 4.0) without NaN";
}

TEST(smoke_MinimumNaN, NanPropagation_f32) {
    run_nan_propagation_test<ov::op::v1::Minimum>(ov::element::f32);
}

TEST(smoke_MinimumNaN, NanPropagation_f16) {
    ov::Core core;
    const auto caps = core.get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
    if (std::find(caps.begin(), caps.end(), ov::device::capability::FP16) == caps.end()) {
        GTEST_SKIP() << "fp16 not supported on this device";
    }
    run_nan_propagation_test<ov::op::v1::Minimum>(ov::element::f16);
}

TEST(smoke_MaximumNaN, NanPropagation_f32) {
    run_nan_propagation_test<ov::op::v1::Maximum>(ov::element::f32);
}

TEST(smoke_MaximumNaN, NanPropagation_f16) {
    ov::Core core;
    const auto caps = core.get_property(ov::test::utils::DEVICE_GPU, ov::device::capabilities);
    if (std::find(caps.begin(), caps.end(), ov::device::capability::FP16) == caps.end()) {
        GTEST_SKIP() << "fp16 not supported on this device";
    }
    run_nan_propagation_test<ov::op::v1::Maximum>(ov::element::f16);
}

}  // namespace
