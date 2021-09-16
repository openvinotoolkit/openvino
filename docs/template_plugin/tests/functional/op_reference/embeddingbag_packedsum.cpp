// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <limits>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

struct EmbeddingBagPackedSumParams {
    template <class IT>
    EmbeddingBagPackedSumParams(const ngraph::PartialShape& shape,
                                const ngraph::element::Type& iType,
                                const std::vector<IT>& iValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)) {
        std::vector<IT> oValues;
        std::vector<double> output;
        for (auto element : iValues)
            output.push_back(static_cast<double>(element));

        std::transform(output.begin(), output.end(), output.begin(), [](double input) -> double {
            return std::atanh(input);
        });

        if (std::is_integral<IT>()) {
            std::transform(output.begin(), output.end(), output.begin(), [](double input) -> double {
                return std::round(input);
            });
        }

        for (auto element : output)
            oValues.push_back(static_cast<IT>(element));
        refData = CreateBlob(outType, oValues);
    }
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceEmbeddingBagPackedSumLayerTest : public testing::TestWithParam<EmbeddingBagPackedSumParams>,
                                                public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<EmbeddingBagPackedSumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape,
                                                    const element::Type& input_type,
                                                    const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto atanh = std::make_shared<op::Atanh>(in);
        return std::make_shared<Function>(NodeVector{atanh}, ParameterVector{in});
    }
};

TEST_P(ReferenceEmbeddingBagPackedSumLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_EmbeddingBagPackedSum_With_Hardcoded_Refs,
                         ReferenceEmbeddingBagPackedSumLayerTest,
                         ::testing::Values(EmbeddingBagPackedSumParams(
                             ngraph::PartialShape{2, 4},
                             ngraph::element::f32,
                             std::vector<float>{-INFINITY, -2.0f, -1.0f, -0.5f, 0.0f, 0.8f, 1.0f, INFINITY})),
                         ReferenceEmbeddingBagPackedSumLayerTest::getTestCaseName);
