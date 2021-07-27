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


struct LogicalAndParams {
    template <class IT, class OT>
    LogicalAndParams(const ngraph::PartialShape& input_shape1, const ngraph::PartialShape& input_shape2 ,
                     const std::vector<IT>& iValues1, const std::vector<IT>& iValues2, const std::vector<OT>& oValues)
        : pshape1(input_shape1), pshape2(input_shape2), inType(ngraph::element::boolean), outType(ngraph::element::boolean),
          inputData1(CreateBlob(ngraph::element::boolean, iValues1)), inputData2(CreateBlob(ngraph::element::boolean, iValues2)),
          refData(CreateBlob(ngraph::element::boolean, oValues)) {}
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape1,
    const PartialShape& input_shape2, const element::Type& input_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::Parameter>(input_type, input_shape2);
        const auto logical_and = std::make_shared<op::v1::LogicalAnd>(in, in2);
        return std::make_shared<Function>(NodeVector {logical_and}, ParameterVector {in, in2});
    }
};

TEST_P(ReferenceLogicalAndLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_LogicalAnd_With_Hardcoded_Refs, ReferenceLogicalAndLayerTest,
    ::testing::Values(
        LogicalAndParams(ngraph::PartialShape {2, 2}, ngraph::PartialShape {2, 2},
                std::vector<char> {true, false, true, false},
                std::vector<char> {false, true, true, false},
                std::vector<char> {false, false, true, false}),
        LogicalAndParams(ngraph::PartialShape {2, 1, 2, 1}, ngraph::PartialShape {1, 1, 2, 1},
                std::vector<char> {true, false, true, false},
                std::vector<char> {true, false},
                std::vector<char> {true, false, true, false}),
        LogicalAndParams(ngraph::PartialShape {3, 4}, ngraph::PartialShape {3, 4},
                std::vector<char> {true, true, true, true, true, false, true, false, false, true, true, true},
                std::vector<char> {true, true, true, true, true, false, true, false, false, true, true, false},
                std::vector<char> {true, true, true, true, true, false, true, false, false, true, true, false})),
    ReferenceLogicalAndLayerTest::getTestCaseName);
