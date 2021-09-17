// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <limits>
#include <algorithm>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

namespace {
struct ClampParams {
    template <class IT>
    ClampParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues,
                const double min, const double max)
        : min(min),
          max(max),
          pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)),
          refData(CreateBlob(iType, oValues)) {}

    double min = 0;
    double max = 0;

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    Blob::Ptr inputData;
    Blob::Ptr refData;
};

class ReferenceClampLayerTest : public testing::TestWithParam<ClampParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.min, params.max);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ClampParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "min=" << param.min << "_";
        result << "max=" << param.max;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& expected_output_type, const double min, const double max) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto Clamp = std::make_shared<op::Clamp>(in, min, max);
        return std::make_shared<Function>(NodeVector {Clamp}, ParameterVector {in});
    }
};

TEST_P(ReferenceClampLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Clamp_With_Hardcoded_Refs, ReferenceClampLayerTest,
    ::testing::Values(ClampParams(ngraph::PartialShape {5, 2}, ngraph::element::f32,
                                std::vector<float> {-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
                                std::vector<float> {0.2, 0.2, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6},
                                0.2,
                                0.6),
                      ClampParams(ngraph::PartialShape {5, 2}, ngraph::element::f32,
                                std::vector<float> {std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
                                                   -INFINITY, INFINITY,
                                                   9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001},
                                std::vector<float> {10.0, 20.0, 10.0, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0},
                                10.0,
                                20.0),
                      ClampParams(ngraph::PartialShape {5, 2}, ngraph::element::f32,
                                std::vector<float> {std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
                                                   -INFINITY, INFINITY,
                                                   9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001},
                                std::vector<float> {10.0, std::numeric_limits<float>::max(), 10.0, INFINITY, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0},
                                10.0,
                                INFINITY),
                      ClampParams(ngraph::PartialShape {5, 2}, ngraph::element::f32,
                                std::vector<float> {std::numeric_limits<float>::min(), std::numeric_limits<float>::max(),
                                                   -INFINITY, INFINITY,
                                                   9.99999, 10.0, 10.000001, 19.999999, 20.0, 20.000001},
                                std::vector<float> {std::numeric_limits<float>::min(), 20.0, -INFINITY, 20.0, 10.0, 10.0, 10.000001, 19.999999, 20.0, 20.0},
                                -INFINITY,
                                20.0)),
    ReferenceClampLayerTest::getTestCaseName);

} // namespace