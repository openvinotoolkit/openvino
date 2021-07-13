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

struct SignParams {
    template <class IT, class OT>
    SignParams(const ngraph::PartialShape& shape, const ngraph::element::Type& iType, const ngraph::element::Type& oType, const std::vector<IT>& iValues,
                  const std::vector<OT>& oValues, size_t iSize = 0, size_t oSize = 0)
        : pshape(shape), inType(iType), outType(oType), inputData(CreateBlob(iType, iValues, iSize)), refData(CreateBlob(oType, oValues, oSize)) {}
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceSignLayerTest : public testing::TestWithParam<SignParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SignParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto sign = std::make_shared<op::Sign>(in);
        return std::make_shared<Function>(NodeVector {sign}, ParameterVector {in});
    }
};

TEST_P(ReferenceSignLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Sign_With_Hardcoded_Refs, ReferenceSignLayerTest,
    ::testing::Values(
        // destination boolean
        SignParams(ngraph::PartialShape {6}, ngraph::element::f32, ngraph::element::f32,
                      std::vector<float> {1, -2, 0, -4.8f, 4.8f, -0.0f},
                      std::vector<float> {1, -1, 0, -1, 1, 0}),
        SignParams(ngraph::PartialShape {6}, ngraph::element::i16, ngraph::element::i16,
                      std::vector<int16_t> {1, -2, 0, -4, 4, -0},
                      std::vector<int16_t> {1, -1, 0, -1, 1, 0}),
        SignParams(ngraph::PartialShape {6}, ngraph::element::i32, ngraph::element::i32,
                      std::vector<int32_t> {1, -2, 0, -4, 4, -0},
                      std::vector<int32_t> {1, -1, 0, -1, 1, 0}),
        SignParams(ngraph::PartialShape {6}, ngraph::element::i64, ngraph::element::i64,
                      std::vector<int64_t> {1, -2, 0, -4, 4, -0},
                      std::vector<int64_t> {1, -1, 0, -1, 1, 0})),
    ReferenceSignLayerTest::getTestCaseName);
