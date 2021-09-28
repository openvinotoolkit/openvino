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
struct MishParams {
    template <class IT>
    MishParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateBlob(iType, iValues)),
          refData(CreateBlob(iType, oValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    Blob::Ptr inputData;
    Blob::Ptr refData;
};

class ReferenceMishLayerTest : public testing::TestWithParam<MishParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MishParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape, const element::Type& input_type,
                                                    const element::Type& Mishected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);
        const auto Mish = std::make_shared<op::v4::Mish>(in);
        return std::make_shared<Function>(NodeVector {Mish}, ParameterVector {in});
    }
};

TEST_P(ReferenceMishLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<MishParams> generateMishFloatParams(const Shape& staticShape) {
    using T = typename element_type_traits<IN_ET>::value_type;

    // generate input tensor (with possible type conversion)
    auto staticSize = shape_size(staticShape);
    std::vector<T> expected;
    std::vector<T> input;
    {
        std::mt19937 gen{0};  // use fixed seed for reproducibility of the test
        std::normal_distribution<> d{0.0, 20.0};

        for (auto i = staticSize; i > 0; i--) {
            auto x = static_cast<T>(d(gen));
            auto y = static_cast<T>(static_cast<double>(x) * std::tanh(std::log(1.0 + std::exp(x))));
            input.push_back(x);
            expected.push_back(y);
        }
    }

    std::vector<MishParams> mishParams {
        MishParams(staticShape,
                    IN_ET,
                    input,
                    expected)
    };
    return mishParams;
}

std::vector<MishParams> generateMishCombinedParams() {
    const std::vector<std::vector<MishParams>> mishTypeParams {
        generateMishFloatParams<element::Type_t::f32>({2, 5}),
        generateMishFloatParams<element::Type_t::f32>({2, 3, 4, 5}),
        generateMishFloatParams<element::Type_t::f16>({2, 5}),
        generateMishFloatParams<element::Type_t::f16>({2, 3, 4, 5})
        };
    std::vector<MishParams> combinedParams;

    for (const auto& params : mishTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Mish_With_Hardcoded_Refs, ReferenceMishLayerTest,
    testing::ValuesIn(generateMishCombinedParams()), ReferenceMishLayerTest::getTestCaseName);

} // namespace