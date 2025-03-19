// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/clamp.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ClampParams {
    template <class IT>
    ClampParams(const ov::Shape& shape,
                const ov::element::Type& iType,
                const std::vector<IT>& iValues,
                const std::vector<IT>& oValues,
                const double min,
                const double max)
        : min(min),
          max(max),
          shape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape, iType, iValues)),
          refData(CreateTensor(shape, iType, oValues)) {}

    double min = 0;
    double max = 0;

    ov::Shape shape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceClampLayerTest : public testing::TestWithParam<ClampParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.shape, params.inType, params.outType, params.min, params.max);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ClampParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.shape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "min=" << param.min << "_";
        result << "max=" << param.max;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ov::Shape& input_shape,
                                                 const ov::element::Type& input_type,
                                                 const ov::element::Type& expected_output_type,
                                                 const double min,
                                                 const double max) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Clamp = std::make_shared<op::v0::Clamp>(in, min, max);
        return std::make_shared<ov::Model>(NodeVector{Clamp}, ParameterVector{in});
    }
};

TEST_P(ReferenceClampLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ClampParams> generateClampFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    auto min = std::numeric_limits<T>::min();
    auto max = std::numeric_limits<T>::max();
    auto pinf = std::numeric_limits<float>::infinity();
    auto ninf = -std::numeric_limits<float>::infinity();
    std::vector<ClampParams> clampParams{
        ClampParams(ov::Shape{5, 2},
                    IN_ET,
                    std::vector<T>{-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                    std::vector<T>{0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6},
                    0.2,
                    0.6),
        ClampParams(ov::Shape{5, 2},
                    IN_ET,
                    std::vector<T>{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001},
                    std::vector<T>{10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0},
                    10.0,
                    20.0),
        ClampParams(ov::Shape{5, 2},
                    IN_ET,
                    std::vector<T>{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001},
                    std::vector<T>{10.0, max, 10.0, pinf, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.000001},
                    10.0,
                    pinf),
        ClampParams(ov::Shape{5, 2},
                    IN_ET,
                    std::vector<T>{min, max, ninf, pinf, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001},
                    std::vector<T>{min, 20.0, ninf, 20.0, 9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.0},
                    ninf,
                    20.0)};
    return clampParams;
}

template <element::Type_t IN_ET>
std::vector<ClampParams> generateClampIntParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    auto min = std::numeric_limits<T>::min();
    auto max = std::numeric_limits<T>::max();
    auto pinf = std::numeric_limits<float>::infinity();
    auto ninf = -std::numeric_limits<float>::infinity();
    std::vector<ClampParams> clampParams{ClampParams(ov::Shape{6},
                                                     IN_ET,
                                                     std::vector<T>{-1, 3, -10, 20, 6, 2},
                                                     std::vector<T>{1, 3, 1, 5, 5, 2},
                                                     0.4,
                                                     5.6),
                                         ClampParams(ov::Shape{6},
                                                     IN_ET,
                                                     std::vector<T>{-6, 1, -2, 0, -1, 2},
                                                     std::vector<T>{-5, -1, -2, -1, -1, -1},
                                                     -5.6,
                                                     -0.4),
                                         ClampParams(ov::Shape{4, 2},
                                                     IN_ET,
                                                     std::vector<T>{min, max, 9, 10, 11, 19, 20, 21},
                                                     std::vector<T>{10, 20, 10, 10, 11, 19, 20, 20},
                                                     10.0,
                                                     20.0),
                                         ClampParams(ov::Shape{4, 2},
                                                     IN_ET,
                                                     std::vector<T>{min, max, 9, 10, 11, 19, 20, 21},
                                                     std::vector<T>{10, max, 10, 10, 11, 19, 20, 21},
                                                     10.0,
                                                     pinf),
                                         ClampParams(ov::Shape{4, 2},
                                                     IN_ET,
                                                     std::vector<T>{min, max, 9, 10, 11, 19, 20, 21},
                                                     std::vector<T>{min, 20, 9, 10, 11, 19, 20, 20},
                                                     ninf,
                                                     20.0)};
    return clampParams;
}

template <element::Type_t IN_ET>
std::vector<ClampParams> generateClampUintParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    auto min = std::numeric_limits<T>::min();
    T max = (static_cast<T>(1) << (std::numeric_limits<T>::digits - 1)) - 1;
    auto pinf = static_cast<double>(max);
    auto ninf = -std::numeric_limits<float>::infinity();
    std::vector<ClampParams> clampParams{ClampParams(ov::Shape{4, 2},
                                                     IN_ET,
                                                     std::vector<T>{min, max, 9, 10, 11, 19, 20, 21},
                                                     std::vector<T>{10, 20, 10, 10, 11, 19, 20, 20},
                                                     10.0,
                                                     20.0),
                                         ClampParams(ov::Shape{4, 2},
                                                     IN_ET,
                                                     std::vector<T>{min, max, 9, 10, 11, 19, 20, 21},
                                                     std::vector<T>{10, max, 10, 10, 11, 19, 20, 21},
                                                     10.0,
                                                     pinf),
                                         ClampParams(ov::Shape{4, 2},
                                                     IN_ET,
                                                     std::vector<T>{min, max, 9, 10, 11, 19, 20, 21},
                                                     std::vector<T>{min, 20, 9, 10, 11, 19, 20, 20},
                                                     ninf,
                                                     20.0)};
    return clampParams;
}

std::vector<ClampParams> generateClampCombinedParams() {
    const std::vector<std::vector<ClampParams>> clampTypeParams{generateClampFloatParams<element::Type_t::f32>(),
                                                                generateClampFloatParams<element::Type_t::f16>(),
                                                                generateClampFloatParams<element::Type_t::bf16>(),
                                                                generateClampIntParams<element::Type_t::i8>(),
                                                                generateClampIntParams<element::Type_t::i16>(),
                                                                generateClampIntParams<element::Type_t::i32>(),
                                                                generateClampIntParams<element::Type_t::i64>(),
                                                                generateClampUintParams<element::Type_t::u8>(),
                                                                generateClampUintParams<element::Type_t::u16>(),
                                                                generateClampUintParams<element::Type_t::u32>(),
                                                                generateClampUintParams<element::Type_t::u64>()};
    std::vector<ClampParams> combinedParams;

    for (const auto& params : clampTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Clamp_With_Hardcoded_Refs,
                         ReferenceClampLayerTest,
                         testing::ValuesIn(generateClampCombinedParams()),
                         ReferenceClampLayerTest::getTestCaseName);

}  // namespace
