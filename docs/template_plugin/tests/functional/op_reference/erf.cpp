// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <limits>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace ngraph;
using namespace InferenceEngine;

struct ErfParams {
    template <class IT>
    ErfParams(const ngraph::PartialShape& shape, const ngraph::element::Type& iType, const std::vector<IT>& iValues, size_t iSize = 0, size_t oSize = 0)
        : pshape(shape), inType(iType), outType(iType), inputData(CreateBlob(iType, iValues, iSize)) {
        std::vector<IT> oValues(iValues);
        std::transform(oValues.begin(), oValues.end(), oValues.begin(), [](float input) -> float {
            return std::erf(input);
        });
        refData = CreateBlob(outType, oValues, oSize);
    }
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceErfLayerTest : public testing::TestWithParam<ErfParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ErfParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto erf = std::make_shared<op::Erf>(in);
        return std::make_shared<Function>(NodeVector {erf}, ParameterVector {in});
    }
};

TEST_P(ReferenceErfLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_Erf_With_Hardcoded_Refs, ReferenceErfLayerTest,
                         ::testing::Values(ErfParams(ngraph::PartialShape {2, 5}, ngraph::element::f32,
                                                     std::vector<float> {-INFINITY, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, INFINITY})),
                         ReferenceErfLayerTest::getTestCaseName);
