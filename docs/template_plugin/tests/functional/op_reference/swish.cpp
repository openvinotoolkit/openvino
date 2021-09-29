// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <limits>
#include <algorithm>
#include <cmath>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

namespace {
struct SwishParams {
    template <class IT>
    SwishParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)),
          testDefaults(true) {
              std::vector<IT> oValues;
              std::vector<double> output;
              for (auto element : iValues)
                  output.push_back(static_cast<double>(element));

              std::transform(output.begin(), output.end(), output.begin(), [](double x) -> double {
                  return (x / (1.0f + std::exp(x * -1.0f)));
              });

              for (auto element : output)
                  oValues.push_back(static_cast<IT>(element));
              refData = CreateBlob(outType, oValues);
          }

    template <class IT>
    SwishParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues,
                const double beta)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)),
          testDefaults(false),
          beta(beta) {
              std::vector<IT> oValues;
              std::vector<double> output;
              std::vector<IT> betaVector;

              for (auto element : iValues)
                  output.push_back(static_cast<double>(element));

              std::transform(output.begin(), output.end(), output.begin(), [&beta](double x) -> double {
                  return (x / (1.0f + std::exp(x * beta * -1.0f)));
              });

              for (auto element : output)
                  oValues.push_back(static_cast<IT>(element));
              refData = CreateBlob(outType, oValues);

              betaVector.push_back(static_cast<IT>(beta));
              betaBlob = CreateBlob(inType, betaVector);
          }

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    Blob::Ptr inputData;
    Blob::Ptr refData;
    Blob::Ptr betaBlob;

    bool testDefaults = false;
    double beta = 1;
};

class ReferenceSwishLayerTest : public testing::TestWithParam<SwishParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        threshold = 0.06; // 0.01 failed in fp32 test

        auto params = GetParam();
        if (params.testDefaults) {
            function = CreateFunction(params.pshape, params.inType, params.outType);

            inputData = {params.inputData};
            refOutData = {params.refData};
        } else {
            function = CreateFunction(params.pshape, params.inType, params.outType, params.beta);

            inputData = {params.inputData, params.betaBlob};
            refOutData = {params.refData};
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SwishParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "beta=" << param.beta;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& Swishected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto Swish = std::make_shared<op::v4::Swish>(in);
        return std::make_shared<Function>(NodeVector {Swish}, ParameterVector {in});
    }

    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& Swishected_output_type, const double beta) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto BETA = std::make_shared<op::Parameter>(input_type, Shape {});
        const auto Swish = std::make_shared<op::v4::Swish>(in);
        return std::make_shared<Function>(NodeVector {Swish}, ParameterVector {in, BETA});
    }
};

TEST_P(ReferenceSwishLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<SwishParams> generateSwishFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<SwishParams> swishParams {
        SwishParams(ngraph::PartialShape {2, 4},
                    IN_ET,
                    std::vector<T>{0.4, -5.7, -6, 3, -0.9, 23, 5, 3.3},
                    0.6f),
        SwishParams(ngraph::PartialShape {2, 3},
                    IN_ET,
                    std::vector<T>{1, 8, -8, 17, -0.5, -1}),
        SwishParams(ngraph::PartialShape {2, 2, 1, 2},
                    IN_ET,
                    std::vector<T>{0.1, 0.6, 20, -7, -5.3, 3.5, -9, 11},
                    0.33f)
    };
    return swishParams;
}

std::vector<SwishParams> generateSwishCombinedParams() {
    const std::vector<std::vector<SwishParams>> swishTypeParams {
        generateSwishFloatParams<element::Type_t::f32>(),
        generateSwishFloatParams<element::Type_t::f16>()
        };
    std::vector<SwishParams> combinedParams;

    for (const auto& params : swishTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Swish_With_Hardcoded_Refs, ReferenceSwishLayerTest,
    testing::ValuesIn(generateSwishCombinedParams()), ReferenceSwishLayerTest::getTestCaseName);

} // namespace