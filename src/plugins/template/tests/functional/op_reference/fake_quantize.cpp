// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct FakeQuantizeParams {
    template <class IT>
    FakeQuantizeParams(const Shape& input_shape,
                       const Shape& expected_shape,
                       const element::Type& input_type,
                       const element::Type& expected_type,
                       const std::vector<IT>& input_data,
                       const std::vector<IT>& expected_data,
                       const std::shared_ptr<op::v0::Constant>& input_low,
                       const std::shared_ptr<op::v0::Constant>& input_high,
                       const std::shared_ptr<op::v0::Constant>& output_low,
                       const std::shared_ptr<op::v0::Constant>& output_high,
                       const std::size_t& levels,
                       const op::AutoBroadcastSpec& broadcast = op::AutoBroadcastType::NONE)
        : m_input_shape(input_shape),
          m_expected_shape(expected_shape),
          m_input_type(input_type),
          m_expected_type(expected_type),
          m_input_data(CreateTensor(input_shape, input_type, input_data)),
          m_expected_data(CreateTensor(expected_shape, expected_type, expected_data)),
          m_input_low(input_low),
          m_input_high(input_high),
          m_output_low(output_low),
          m_output_high(output_high),
          m_levels(levels),
          m_broadcast(broadcast) {}

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_data;
    ov::Tensor m_expected_data;
    std::shared_ptr<op::v0::Constant> m_input_low;
    std::shared_ptr<op::v0::Constant> m_input_high;
    std::shared_ptr<op::v0::Constant> m_output_low;
    std::shared_ptr<op::v0::Constant> m_output_high;
    std::size_t m_levels;
    op::AutoBroadcastSpec m_broadcast;
};

class ReferenceFakeQuantizeLayerTest : public testing::TestWithParam<FakeQuantizeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape,
                                  params.m_expected_shape,
                                  params.m_input_type,
                                  params.m_expected_type,
                                  params.m_input_low,
                                  params.m_input_high,
                                  params.m_output_low,
                                  params.m_output_high,
                                  params.m_levels,
                                  params.m_broadcast);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<FakeQuantizeParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "input_shape=" << param.m_input_shape << "; ";
        result << "expected_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "expected_type=" << param.m_expected_type << "; ";
        result << "input_low=" << param.m_input_low << "; ";
        result << "input_high=" << param.m_input_high << "; ";
        result << "output_low=" << param.m_output_low << "; ";
        result << "ouput_high=" << param.m_output_high << "; ";
        result << "broadcast=" << param.m_broadcast.m_type;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const Shape& expected_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_type,
                                                 const std::shared_ptr<op::v0::Constant>& input_low,
                                                 const std::shared_ptr<op::v0::Constant>& input_high,
                                                 const std::shared_ptr<op::v0::Constant>& output_low,
                                                 const std::shared_ptr<op::v0::Constant>& output_high,
                                                 const std::size_t& levels,
                                                 const op::AutoBroadcastSpec& broadcast) {
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        if (broadcast == op::AutoBroadcastType::NONE) {
            return std::make_shared<Model>(
                NodeVector{
                    std::make_shared<op::v0::FakeQuantize>(in, input_low, input_high, output_low, output_high, levels)},
                ParameterVector{in});

        } else {
            return std::make_shared<Model>(
                NodeVector{std::make_shared<
                    op::v0::FakeQuantize>(in, input_low, input_high, output_low, output_high, levels, broadcast)},
                ParameterVector{in});
        }
    }
};

TEST_P(ReferenceFakeQuantizeLayerTest, FakeQuantizeWithHardcodedRefs) {
    Exec();
}

template <typename T>
std::vector<T> iota_vector(size_t size) {
    std::vector<T> d(size);
    std::iota(begin(d), end(d), 0.f);
    return d;
}

template <element::Type_t IN_ET>
std::vector<FakeQuantizeParams> generateParamsForFakeQuantize() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FakeQuantizeParams> params{
        FakeQuantizeParams(
            ov::Shape{1, 2, 3, 4},
            ov::Shape{1, 2, 3, 4},
            IN_ET,
            IN_ET,
            iota_vector<T>(shape_size(ov::Shape{1, 2, 3, 4})),
            std::vector<T>{2.f,          2.f,          2.f,          2.f,          6.6666669f,   6.6666669f,
                           6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,
                           11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f,
                           11.33333301f, 11.33333301f, 16.f,         16.f,         16.f,         16.f},
            op::v0::Constant::create(IN_ET, Shape{}, {0.f}),
            op::v0::Constant::create(IN_ET, Shape{}, {23.f}),
            op::v0::Constant::create(IN_ET, Shape{}, {2.f}),
            op::v0::Constant::create(IN_ET, Shape{}, {16.f}),
            4),
        FakeQuantizeParams(ov::Shape{1, 2, 3, 4},
                           ov::Shape{1, 2, 3, 4},
                           IN_ET,
                           IN_ET,
                           iota_vector<T>(shape_size(Shape{1, 2, 3, 4})),
                           std::vector<T>{2.f,   2.f,   2.f,   2.f,   2.f,  5.5f, 5.5f, 5.5f, 5.5f, 9.f,  9.f,  9.f,
                                          12.5f, 12.5f, 12.5f, 12.5f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f},
                           op::v0::Constant::create(IN_ET, Shape{}, {3.f}),
                           op::v0::Constant::create(IN_ET, Shape{}, {17.f}),
                           op::v0::Constant::create(IN_ET, Shape{}, {2.f}),
                           op::v0::Constant::create(IN_ET, Shape{}, {16.f}),
                           5),
        FakeQuantizeParams(
            ov::Shape{1, 2, 5, 5},
            ov::Shape{1, 2, 5, 5},
            IN_ET,
            IN_ET,
            iota_vector<T>(shape_size(Shape{1, 2, 5, 5})),
            std::vector<T>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f, 20.0f, 20.0f, 20.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f,
                           50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f,
                           70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f},
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {5.f, 30.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {10.f, 40.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {0.f, 50.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {20.f, 70.f}),
            5),
        FakeQuantizeParams(
            ov::Shape{1, 2, 5, 5},
            ov::Shape{1, 2, 5, 5},
            IN_ET,
            IN_ET,
            iota_vector<T>(shape_size(Shape{1, 2, 5, 5})),
            std::vector<T>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f, 20.0f, 20.0f, 20.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f,
                           50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f,
                           70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f},
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {5.f, 30.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {10.f, 40.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {0.f, 50.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {20.f, 70.f}),
            5,
            op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, 1)),
        FakeQuantizeParams(
            ov::Shape{1, 2, 5, 5},
            ov::Shape{1, 2, 5, 5},
            IN_ET,
            IN_ET,
            iota_vector<T>(shape_size(Shape{1, 2, 5, 5})),
            std::vector<T>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f, 20.0f, 20.0f, 20.0f,
                           20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f,
                           50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f,
                           70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f},
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {5.f, 30.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {10.f, 40.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {0.f, 50.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {20.f, 70.f}),
            5,
            op::AutoBroadcastSpec(op::AutoBroadcastType::PDPD, -1)),
        FakeQuantizeParams(ov::Shape{1, 2, 3, 3},
                           ov::Shape{1, 2, 3, 3},
                           IN_ET,
                           IN_ET,
                           iota_vector<T>(shape_size(Shape{1, 2, 3, 3})),
                           std::vector<T>{
                               5.0f,
                               9.0f,
                               13.0f,
                               17.0f,
                               21.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                               25.0f,
                           },
                           op::v0::Constant::create(IN_ET,
                                                    Shape{1, 1, 3, 3},
                                                    {
                                                        0.f,
                                                    }),
                           op::v0::Constant::create(IN_ET,
                                                    Shape{1, 1, 3, 3},
                                                    {
                                                        5.f,
                                                    }),
                           op::v0::Constant::create(IN_ET,
                                                    Shape{1, 2, 3, 1},
                                                    {
                                                        5.f,
                                                    }),
                           op::v0::Constant::create(IN_ET,
                                                    Shape{1, 2, 1, 3},
                                                    {
                                                        25.f,
                                                    }),
                           16,
                           op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY)),
        FakeQuantizeParams(
            ov::Shape{1, 2, 4, 4},
            ov::Shape{1, 2, 4, 4},
            IN_ET,
            IN_ET,
            iota_vector<T>(shape_size(Shape{1, 2, 4, 4})),
            std::vector<T>{
                0,     0,     0,    0,    0,    0,    0,    0,     0,     8.75,  8.75,  8.75,  8.75, 8.75, 8.75, 17.5,
                23.75, 23.75, 27.5, 27.5, 27.5, 27.5, 27.5, 31.25, 31.25, 31.25, 31.25, 31.25, 35,   35,   35,   35,
            },
            op::v0::Constant::create(IN_ET, Shape{1, 2, 1, 1}, {5.f, 10.f}),
            op::v0::Constant::create(IN_ET, Shape{1, 1}, {30.f}),
            op::v0::Constant::create(IN_ET, Shape{2, 1, 1}, {0.f, 20.f}),
            op::v0::Constant::create(IN_ET, Shape{1}, {35.f}),
            5),

    };
    return params;
}

std::vector<FakeQuantizeParams> generateCombinedParamsForFakeQuantize() {
    const std::vector<std::vector<FakeQuantizeParams>> allTypeParams{
        generateParamsForFakeQuantize<element::Type_t::f32>(),
        generateParamsForFakeQuantize<element::Type_t::f16>()};

    std::vector<FakeQuantizeParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_With_Hardcoded_Refs,
                         ReferenceFakeQuantizeLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForFakeQuantize()),
                         ReferenceFakeQuantizeLayerTest::getTestCaseName);

}  // namespace
