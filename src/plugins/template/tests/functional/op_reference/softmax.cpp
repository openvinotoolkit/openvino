// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

#ifdef _MSC_VER
#    pragma warning(disable : 4756)
#endif

namespace {
struct SoftmaxParams {
    template <class IT>
    SoftmaxParams(const ov::Shape& shape,
                  const ov::element::Type& iType,
                  const std::vector<IT>& iValues,
                  const std::vector<IT>& oValues,
                  const int64_t axis,
                  const std::string& test_name)
        : axis(axis),
          pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape, iType, iValues)),
          refData(CreateTensor(shape, iType, oValues)),
          test_case_name(test_name) {}

    int64_t axis = 0;

    ov::Shape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
    std::string test_case_name;
};

class ReferenceSoftmaxLayerTest : public testing::TestWithParam<SoftmaxParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.axis);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SoftmaxParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        if (param.test_case_name != "") {
            result << "axis=" << param.axis << "_";
            result << param.test_case_name;
        } else {
            result << "axis=" << param.axis;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type,
                                                 const int64_t axis) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Softmax = std::make_shared<op::v1::Softmax>(in, axis);
        return std::make_shared<ov::Model>(NodeVector{Softmax}, ParameterVector{in});
    }
};

TEST_P(ReferenceSoftmaxLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SoftmaxParams> generateSoftmaxFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    auto d0 = expf(-10) + expf(-1);
    auto d1 = expf(-20) + expf(-2);
    auto d2 = expf(-30) + expf(-3);
    auto d3 = expf(-40) + expf(-4);
    auto d4 = expf(-50) + expf(-5);
    auto d5 = expf(-60) + expf(-6);

    auto d0_a1 = expf(-10) + expf(-20) + expf(-30);
    auto d1_a1 = expf(-40) + expf(-50) + expf(-60);

    auto d0_a0 = expf(-10) + expf(-40);
    auto d1_a0 = expf(-20) + expf(-50);
    auto d2_a0 = expf(-30) + expf(-60);

    auto low = static_cast<float>(std::numeric_limits<T>::lowest());
    auto high = static_cast<float>(std::numeric_limits<T>::max());

    auto d0_uf = expf(low) + expf(3);
    auto d1_uf = expf(1) + expf(4);
    auto d2_uf = expf(2) + expf(5);

    auto d0_of = expf(high - high) + expf(3 - high);
    auto d1_of = expf(1) + expf(4);
    auto d2_of = expf(2) + expf(5);

    std::vector<SoftmaxParams> softmaxParams{
        SoftmaxParams(ov::Shape{2, 2, 3},
                      IN_ET,
                      std::vector<T>{-10, -20, -30, -40, -50, -60, -1, -2, -3, -4, -5, -6},
                      std::vector<T>{expf(-10) / d0,
                                     expf(-20) / d1,
                                     expf(-30) / d2,
                                     expf(-40) / d3,
                                     expf(-50) / d4,
                                     expf(-60) / d5,
                                     expf(-1) / d0,
                                     expf(-2) / d1,
                                     expf(-3) / d2,
                                     expf(-4) / d3,
                                     expf(-5) / d4,
                                     expf(-6) / d5},
                      0,
                      ""),
        SoftmaxParams(ov::Shape{2, 3},
                      IN_ET,
                      std::vector<T>{-10, -20, -30, -40, -50, -60},
                      std::vector<T>{expf(-10) / d0_a1,
                                     expf(-20) / d0_a1,
                                     expf(-30) / d0_a1,
                                     expf(-40) / d1_a1,
                                     expf(-50) / d1_a1,
                                     expf(-60) / d1_a1},
                      1,
                      ""),
        SoftmaxParams(ov::Shape{2, 3},
                      IN_ET,
                      std::vector<T>{-10, -20, -30, -40, -50, -60},
                      std::vector<T>{expf(-10) / d0_a0,
                                     expf(-20) / d1_a0,
                                     expf(-30) / d2_a0,
                                     expf(-40) / d0_a0,
                                     expf(-50) / d1_a0,
                                     expf(-60) / d2_a0},
                      0,
                      "test"),
        SoftmaxParams(ov::Shape{1, 2, 3},
                      IN_ET,
                      std::vector<T>{-10, -20, -30, -40, -50, -60},
                      std::vector<T>{1, 1, 1, 1, 1, 1},
                      0,
                      "trivial"),
        SoftmaxParams(ov::Shape{2, 3},
                      IN_ET,
                      std::vector<T>{low, 1, 2, 3, 4, 5},
                      std::vector<T>{expf(low) / d0_uf,
                                     expf(1) / d1_uf,
                                     expf(2) / d2_uf,
                                     expf(3) / d0_uf,
                                     expf(4) / d1_uf,
                                     expf(5) / d2_uf},
                      0,
                      "underflow"),
        SoftmaxParams(ov::Shape{2, 3},
                      IN_ET,
                      std::vector<T>{high, 1, 2, 3, 4, 5},
                      std::vector<T>{expf(high - high) / d0_of,
                                     expf(1) / d1_of,
                                     expf(2) / d2_of,
                                     expf(3 - high) / d0_of,
                                     expf(4) / d1_of,
                                     expf(5) / d2_of},
                      0,
                      "overflow")};
    return softmaxParams;
}

std::vector<SoftmaxParams> generateSoftmaxCombinedParams() {
    const std::vector<std::vector<SoftmaxParams>> softmaxTypeParams{
        generateSoftmaxFloatParams<element::Type_t::f64>(),
        generateSoftmaxFloatParams<element::Type_t::f32>(),
        generateSoftmaxFloatParams<element::Type_t::f16>(),
        generateSoftmaxFloatParams<element::Type_t::bf16>()};
    std::vector<SoftmaxParams> combinedParams;

    for (const auto& params : softmaxTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Softmax_With_Hardcoded_Refs,
                         ReferenceSoftmaxLayerTest,
                         testing::ValuesIn(generateSoftmaxCombinedParams()),
                         ReferenceSoftmaxLayerTest::getTestCaseName);

}  // namespace
