
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

struct LogicalNotParams {
    template <class IT, class OT>
    LogicalNotParams(const ngraph::PartialShape& input_shape, const ngraph::element::Type& iType, const ngraph::element::Type& oType,
                     const std::vector<IT>& iValues, const std::vector<OT>& oValues)
        : pshape(input_shape), inType(iType), outType(oType),
          inputData(CreateBlob(iType, iValues)),
          refData(CreateBlob(oType, oValues)) {}
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceLogicalNotLayerTest : public testing::TestWithParam<LogicalNotParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<LogicalNotParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "input_shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape,
                                                    const element::Type& input_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto logical_not = std::make_shared<op::v1::LogicalNot>(in);
        return std::make_shared<Function>(NodeVector {logical_not}, ParameterVector {in});
    }
};

TEST_P(ReferenceLogicalNotLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_LogicalNot_With_Hardcoded_Refs, ReferenceLogicalNotLayerTest,
    ::testing::Values(
        LogicalNotParams(ngraph::PartialShape {2, 2},
                element::boolean, element::boolean,
                std::vector<char> {true, false, true, false},
                std::vector<char> {false, true, false, true}),
        LogicalNotParams(ngraph::PartialShape {2, 2},
                element::i32, element::i32,
                std::vector<int32_t> {1, 0, 1, 0},
                std::vector<int32_t> {0, 1, 0, 1}),
        LogicalNotParams(ngraph::PartialShape {2, 2},
                element::i64, element::i64,
                std::vector<int64_t> {1, 0, 1, 0},
                std::vector<int64_t> {0, 1, 0, 1}),
        LogicalNotParams(ngraph::PartialShape {2, 2},
                element::u32, element::u32,
                std::vector<uint32_t> {1, 0, 1, 0},
                std::vector<uint32_t> {0, 1, 0, 1}),
        LogicalNotParams(ngraph::PartialShape {2, 2},
                element::u64, element::u64,
                std::vector<uint64_t> {1, 0, 1, 0},
                std::vector<uint64_t> {0, 1, 0, 1}),
        LogicalNotParams(ngraph::PartialShape {2, 2},
                element::f16, element::f16,
                std::vector<float16> {1, 0, 1, 0},
                std::vector<float16> {0, 1, 0, 1}),
        LogicalNotParams(ngraph::PartialShape {2, 2},
                element::f32, element::f32,
                std::vector<float> {1, 0, 1, 0},
                std::vector<float> {0, 1, 0, 1})),
    ReferenceLogicalNotLayerTest::getTestCaseName);
