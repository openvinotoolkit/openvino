// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

namespace {
struct CumSumParams {
    template <class IT, class AT>
    CumSumParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues, const bool execlusive,
                 const bool reverse, const element::Type& axisType, AT axisValue, const PartialShape& axisShape)
        : execlusive(execlusive),
          reverse(reverse),
          axisValue(axisValue),
          axisShape(axisShape),
          inShape(shape),
          axisType(axisType),
          inType(iType),
          outType(iType),
          axisData(CreateBlob(axisType, std::vector<AT> {axisValue})),
          inputData(CreateBlob(iType, iValues)),
          refData(CreateBlob(iType, oValues)),
          testDefaults(false) {}

    template <class IT>
    CumSumParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues)
        : inShape(shape), inType(iType), outType(iType), inputData(CreateBlob(iType, iValues)), refData(CreateBlob(iType, oValues)), testDefaults(true) {}

    bool execlusive = false;
    bool reverse = false;
    int64_t axisValue = 0;

    PartialShape axisShape;
    PartialShape inShape;
    element::Type axisType;
    element::Type inType;
    element::Type outType;
    Blob::Ptr axisData;
    Blob::Ptr inputData;
    Blob::Ptr refData;

    bool testDefaults = false;
};

class ReferenceCumSumLayerTest : public testing::TestWithParam<CumSumParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        if (params.testDefaults) {
            function = CreateFunction(params.inShape, params.inType);
            inputData = {params.inputData};
            refOutData = {params.refData};
        } else {
            function = CreateFunction(params.inShape, params.inType, params.axisShape, params.axisType, params.execlusive, params.reverse);
            inputData = {params.inputData, params.axisData};
            refOutData = {params.refData};
        }
    }
    static std::string getTestCaseName(const testing::TestParamInfo<CumSumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "testDefaults=" << param.testDefaults << "_";
        result << "axisValue=" << param.axisValue << "_";
        result << "execlusive=" << param.execlusive << "_";
        result << "reverse=" << param.reverse << "_";
        result << "inShape=" << param.inShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& data_shape, const element::Type& data_type, const PartialShape& axis_shape,
                                                    const element::Type& axis_type, const bool execlusive, const bool reverse) {
        const auto data_param = std::make_shared<op::Parameter>(data_type, data_shape);
        const auto axis_param = std::make_shared<op::Parameter>(axis_type, axis_shape);
        const auto cum_sum = std::make_shared<op::v0::CumSum>(data_param, axis_param, execlusive, reverse);
        return std::make_shared<Function>(NodeVector {cum_sum}, ParameterVector {data_param, axis_param});
    }

    static std::shared_ptr<Function> CreateFunction(const PartialShape& data_shape, const element::Type& data_type) {
        const auto data_param = std::make_shared<op::Parameter>(data_type, data_shape);
        const auto cum_sum = std::make_shared<op::v0::CumSum>(data_param);
        return std::make_shared<Function>(NodeVector {cum_sum}, ParameterVector {data_param});
    }
};

TEST_P(ReferenceCumSumLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<CumSumParams> generateCumSumParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<CumSumParams> opParams {
        // Default axis input and attributes
        CumSumParams(PartialShape {6}, type, std::vector<T> {1, 2, 3, 4, 5, 6}, std::vector<T> {1, 3, 6, 10, 15, 21}),
        // Custom axis input and attributes
        CumSumParams(PartialShape {6}, type, std::vector<T> {1, 2, 3, 4, 5, 6}, std::vector<T> {1, 3, 6, 10, 15, 21}, false, false, element::i32, 0,
                     PartialShape {}),
    };
    return opParams;
}

std::vector<CumSumParams> generateGrnCombinedParams() {
    const std::vector<std::vector<CumSumParams>> opTypeParams {
        generateCumSumParams<element::Type_t::bf16>(element::bf16), generateCumSumParams<element::Type_t::f16>(element::f16),
        generateCumSumParams<element::Type_t::f32>(element::f32),   generateCumSumParams<element::Type_t::f64>(element::f64),
        generateCumSumParams<element::Type_t::i32>(element::i32),   generateCumSumParams<element::Type_t::i64>(element::i64),
        generateCumSumParams<element::Type_t::u32>(element::u32),   generateCumSumParams<element::Type_t::i8>(element::i8)};
    std::vector<CumSumParams> combinedParams;
    std::for_each(opTypeParams.begin(), opTypeParams.end(), [&](std::vector<CumSumParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_CumSum_With_Hardcoded_Refs, ReferenceCumSumLayerTest, ::testing::ValuesIn(generateGrnCombinedParams()),
                         ReferenceCumSumLayerTest::getTestCaseName);
}  // namespace
