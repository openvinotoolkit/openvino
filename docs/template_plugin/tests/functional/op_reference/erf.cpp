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

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

struct ErfParams {
    template <class IT>
    ErfParams(const ngraph::PartialShape& shape, const ngraph::element::Type& iType, const std::vector<IT>& iValues)
        : pshape(shape), inType(iType), outType(iType), inputData(CreateBlob(iType, iValues)) {
        std::vector<IT> oValues;
        std::vector<double> output;
        for (auto element : iValues)
            output.push_back(static_cast<double>(element));

        std::transform(output.begin(), output.end(), output.begin(), [](double input) -> double {
            return std::erf(input);
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

INSTANTIATE_TEST_SUITE_P(
    smoke_Erf_With_Hardcoded_Refs, ReferenceErfLayerTest,
    ::testing::Values(ErfParams(ngraph::PartialShape {2, 5}, ngraph::element::f32,
                                std::vector<float> {-INFINITY, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, INFINITY}),
                      ErfParams(ngraph::PartialShape {2, 5}, ngraph::element::f16,
                                std::vector<float16> {-INFINITY, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, INFINITY}),
                      ErfParams(ngraph::PartialShape {2, 3}, ngraph::element::i32,
                                std::vector<int32_t> {std::numeric_limits<int32_t>::min(), -2, -1, 1, 2, std::numeric_limits<int32_t>::max()}),
                      ErfParams(ngraph::PartialShape {2, 3}, ngraph::element::u32,
                                std::vector<uint32_t> {std::numeric_limits<uint32_t>::min(), 0, 1, 2, 3, std::numeric_limits<uint32_t>::max()}),
                      ErfParams(ngraph::PartialShape {2, 3}, ngraph::element::i64,
                                std::vector<int64_t> {std::numeric_limits<int64_t>::min(), -2, -1, 1, 2, std::numeric_limits<int64_t>::max()}),
                      ErfParams(ngraph::PartialShape {2, 3}, ngraph::element::u64,
                                std::vector<uint64_t> {std::numeric_limits<uint64_t>::min(), 0, 1, 2, 3, std::numeric_limits<uint64_t>::max()})),
    ReferenceErfLayerTest::getTestCaseName);
