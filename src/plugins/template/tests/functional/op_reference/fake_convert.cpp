// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct FakeConvertParams {
    std::string name;
    element::Type data_type;
    element::Type destination_type;
    std::vector<float> data;
    std::vector<float> expected;
};

ov::Tensor make_tensor(const element::Type& type, const std::vector<float>& values) {
    ov::Tensor tensor(type, Shape{values.size()});
    for (size_t i = 0; i < values.size(); ++i) {
        if (type == element::f32) {
            tensor.data<float>()[i] = values[i];
        } else if (type == element::f16) {
            tensor.data<float16>()[i] = float16(values[i]);
        } else if (type == element::bf16) {
            tensor.data<bfloat16>()[i] = bfloat16(values[i]);
        } else {
            OPENVINO_THROW("Unsupported data type: ", type);
        }
    }
    return tensor;
}

float get_value(const ov::Tensor& tensor, size_t i) {
    const auto& type = tensor.get_element_type();
    if (type == element::f32)
        return tensor.data<const float>()[i];
    if (type == element::f16)
        return static_cast<float>(tensor.data<const float16>()[i]);
    if (type == element::bf16)
        return static_cast<float>(tensor.data<const bfloat16>()[i]);
    OPENVINO_THROW("Unsupported data type: ", type);
}

class ReferenceFakeConvertLayerTest : public testing::TestWithParam<FakeConvertParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        const auto data = std::make_shared<op::v0::Parameter>(params.data_type, Shape{params.data.size()});
        const auto scale = op::v0::Constant::create(params.data_type, Shape{1}, {1.f});
        const auto shift = op::v0::Constant::create(params.data_type, Shape{1}, {0.f});
        const auto fake_convert = std::make_shared<op::v13::FakeConvert>(data, scale, shift, params.destination_type);
        function = std::make_shared<Model>(OutputVector{fake_convert}, ParameterVector{data});
        inputData = {make_tensor(params.data_type, params.data)};
        refOutData = {make_tensor(params.data_type, params.expected)};
    }

    // Exact check: the default comparator treats -0 and +0 as equal
    void Validate() override {
        const auto& expected = refOutData[0];
        const auto actual = inferRequest.get_tensor(executableNetwork.output());
        ASSERT_EQ(expected.get_element_type(), actual.get_element_type());
        ASSERT_EQ(expected.get_size(), actual.get_size());
        for (size_t i = 0; i < expected.get_size(); ++i) {
            const auto expected_value = get_value(expected, i);
            const auto actual_value = get_value(actual, i);
            if (std::isnan(expected_value)) {
                EXPECT_TRUE(std::isnan(actual_value)) << "index " << i;
            } else {
                EXPECT_EQ(expected_value, actual_value) << "index " << i;
                EXPECT_EQ(std::signbit(expected_value), std::signbit(actual_value)) << "index " << i;
            }
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<FakeConvertParams>& obj) {
        const auto& params = obj.param;
        std::ostringstream result;
        result << params.data_type << "_to_" << params.destination_type << "_" << params.name;
        return result.str();
    }
};

TEST_P(ReferenceFakeConvertLayerTest, CompareWithRefs) {
    Exec();
}

constexpr auto nan = std::numeric_limits<float>::quiet_NaN();
constexpr auto inf = std::numeric_limits<float>::infinity();

// f8e4m3 subnormal step is 2^-9; signed zero inputs are exact in f16 and bf16
const std::vector<float> signed_zero_data{-0.f, -0x1p-11f, -0.5f * 0x1p-9f, -0.625f * 0x1p-9f, 0.f, 0.5f * 0x1p-9f};
const std::vector<float> signed_zero_expected{-0.f, -0.f, -0.f, -0x1p-9f, 0.f, 0.f};

INSTANTIATE_TEST_SUITE_P(
    smoke_FakeConvert_Rounding_Special_Values_With_Hardcoded_Refs,
    ReferenceFakeConvertLayerTest,
    ::testing::Values(
        FakeConvertParams{"subnormal_sign_nan",
                          element::f32,
                          element::f8e4m3,
                          {1.25f * 0x1p-10f, 3 * 0x1p-10f, 7 * 0x1p-10f, -0x1p-11f, nan},
                          {0x1p-9f, 0x1p-8f, 0x1p-7f, -0.f, nan}},
        // just above a tie in f32, which is a tie in f16
        FakeConvertParams{"near_tie", element::f32, element::f8e4m3, {1.0625f + 0x1p-20f}, {1.125f}},
        FakeConvertParams{"signed_zero", element::f16, element::f8e4m3, signed_zero_data, signed_zero_expected},
        FakeConvertParams{"signed_zero", element::bf16, element::f8e4m3, signed_zero_data, signed_zero_expected},
        // -2^-18 is below half of the lowest f8e5m2 subnormal
        FakeConvertParams{"negative_zero", element::f32, element::f8e5m2, {-0x1p-18f}, {-0.f}},
        FakeConvertParams{"near_tie", element::f32, element::f8e5m2, {1.125f + 0x1p-19f}, {1.25f}},
        FakeConvertParams{"inf_nan", element::f16, element::f8e5m2, {inf, -inf, nan}, {57344.f, -57344.f, nan}}),
    ReferenceFakeConvertLayerTest::getTestCaseName);

}  // namespace
