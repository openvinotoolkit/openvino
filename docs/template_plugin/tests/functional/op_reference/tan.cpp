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
struct TanParams {
    template <class IT>
    TanParams(const ngraph::PartialShape& shape, const ngraph::element::Type& iType, const std::vector<IT>& iValues,
              const std::vector<IT>& oValues)
        :pshape(shape), inType(iType), outType(iType), inputData(CreateBlob(iType, iValues)), refData(CreateBlob(iType, oValues)) {}
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceTanLayerTest : public testing::TestWithParam<TanParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<TanParams>& obj) {
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
        const auto tan = std::make_shared<op::Tan>(in);
        return std::make_shared<Function>(tan, ParameterVector {in});
    }
};

TEST_P(ReferenceTanLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<TanParams> generateTanParamsInt(const ngraph::element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<TanParams> tanParams {
        TanParams(ngraph::PartialShape {5}, type, std::vector<T> {-2, -1, 0, 1, 2},
                  std::vector<T> {2, -2, 0, 2, -2})
    };
    return tanParams;
}

std::vector<TanParams> generateTanCombinedParams() {
    const std::vector<std::vector<TanParams>> tanTypeParams {generateTanParamsInt<element::Type_t::i32>(ngraph::element::i32),
                                                            generateTanParamsInt<element::Type_t::i64>(ngraph::element::i64)};
    std::vector<TanParams> combinedParams;
    std::for_each(tanTypeParams.begin(), tanTypeParams.end(), [&](std::vector<TanParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_TAN_With_Hardcoded_Refs, ReferenceTanLayerTest, ::testing::ValuesIn(generateTanCombinedParams()),
                         ReferenceTanLayerTest::getTestCaseName);
}  // namespace
