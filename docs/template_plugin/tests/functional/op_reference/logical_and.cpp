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

using namespace ngraph;
using namespace InferenceEngine;

namespace {
struct LogicalAndParams {
    template <class IT, class OT>
    LogicalAndParams(const ngraph::PartialShape& input_shape1, const ngraph::PartialShape& input_shape2 , const ngraph::element::Type& iType,
                const ngraph::element::Type& oType, const std::vector<IT>& iValues1, const std::vector<IT>& iValues2, const std::vector<OT>& oValues)
        : pshape1(input_shape1), pshape2(input_shape2), inType(iType), outType(oType), inputData1(CreateBlob(iType, iValues1)),
        inputData2(CreateBlob(iType, iValues2)), refData(CreateBlob(oType, oValues)) {}
    ngraph::PartialShape pshape1;
    ngraph::PartialShape pshape2;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData1;
    InferenceEngine::Blob::Ptr inputData2;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceLogicalAndLayerTest : public testing::TestWithParam<LogicalAndParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape1, params.pshape2, params.inType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<LogicalAndParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "input_shape1=" << param.pshape1 << "_";
        result << "input_shape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape1, const PartialShape& input_shape2, const element::Type& input_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::Parameter>(input_type, input_shape2);
        const auto logical_and = std::make_shared<op::v1::LogicalAnd>(in, in2);
        return std::make_shared<Function>(NodeVector {logical_and}, ParameterVector {in, in2});
    }
};

TEST_P(ReferenceLogicalAndLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<LogicalAndParams> generateLogicalAndParams(const ngraph::element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<LogicalAndParams> logicalAndParams {
        // 1D // 2D // 3D // 4D
        LogicalAndParams(ngraph::PartialShape {2, 2}, ngraph::PartialShape {2, 2}, type, ngraph::element::boolean,
                std::vector<T> {1, 0, 1, 0},
                std::vector<T> {0, 1, 1, 0},
                std::vector<char> {0, 0, 1, 0}),
        LogicalAndParams(ngraph::PartialShape {1}, ngraph::PartialShape {1},  type, ngraph::element::boolean,
                std::vector<T> {1},
                std::vector<T> {1},
                std::vector<char> {1}),

        LogicalAndParams(ngraph::PartialShape {2, 1, 2, 1}, ngraph::PartialShape {1, 1, 2, 1}, type, ngraph::element::boolean,
                std::vector<T> {1, 0, 1, 0},
                std::vector<T> {1, 0},
                std::vector<char> {1, 0, 1, 0})};
    return logicalAndParams;
}

std::vector<LogicalAndParams> generateLogicalAndCombinedParams() {
    const std::vector<std::vector<LogicalAndParams>> logicalAndTypeParams {generateLogicalAndParams<element::Type_t::u8>(ngraph::element::boolean)};
    std::vector<LogicalAndParams> combinedParams;
    std::for_each(logicalAndTypeParams.begin(), logicalAndTypeParams.end(), [&](std::vector<LogicalAndParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_LogicalAnd_With_Hardcoded_Refs, ReferenceLogicalAndLayerTest, ::testing::ValuesIn(generateLogicalAndCombinedParams()),
                                 ReferenceLogicalAndLayerTest::getTestCaseName);
}  // namespace
