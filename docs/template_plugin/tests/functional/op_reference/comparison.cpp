// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include "ngraph_functions/builders.hpp"
#include <tuple>

#include "base_reference_test.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using ComparisonTypes = ngraph::helpers::ComparisonTypes;

namespace {
struct RefComparisonParams {
    template <class IT, class OT>
    RefComparisonParams(const ComparisonTypes comp, const ngraph::PartialShape& input_shape1, const ngraph::PartialShape& input_shape2,
                     const ngraph::element::Type& iType, const ngraph::element::Type& oType, const std::vector<IT>& iValues1, const std::vector<IT>& iValues2,
                     const std::vector<OT>& oValues)
        : comparisonType(comp),
          pshape1(input_shape1),
          pshape2(input_shape2),
          inType(iType),
          outType(oType),
          inputData1(CreateBlob(iType, iValues1)),
          inputData2(CreateBlob(iType, iValues2)),
          refData(CreateBlob(oType, oValues)) {}
    ComparisonTypes comparisonType;
    ngraph::PartialShape pshape1;
    ngraph::PartialShape pshape2;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData1;
    InferenceEngine::Blob::Ptr inputData2;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceComparisonLayerTest : public testing::TestWithParam<RefComparisonParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.comparisonType, params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RefComparisonParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "comparisonType=" << param.comparisonType << "_";
        result << "inpt_shape1=" << param.pshape1 << "_";
        result << "inpt_shape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(ComparisonTypes& comp_op_type, PartialShape& input_shape1, const PartialShape& input_shape2,
                                                    const element::Type& input_type, const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::Parameter>(input_type, input_shape2);
        const auto comp = ngraph::builder::makeComparison(in, in2, comp_op_type);
        return std::make_shared<Function>(NodeVector {comp}, ParameterVector {in, in2});
    }
};

TEST_P(ReferenceComparisonLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<RefComparisonParams> generateComparisonParams(const ngraph::element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<RefComparisonParams> compParams {
        // 1D // 2D // 3D // 4D
        RefComparisonParams(ComparisonTypes::LESS, ngraph::PartialShape {2, 2}, ngraph::PartialShape {2, 2}, type, ngraph::element::boolean,
                std::vector<T> {0, 12, 23, 0},
                std::vector<T> {0, 12, 23, 0},
                std::vector<char> {0, 0, 0, 0}),
        RefComparisonParams(ComparisonTypes::LESS, ngraph::PartialShape {2, 3}, ngraph::PartialShape {2, 3}, type, ngraph::element::boolean,
                std::vector<T> {0, 6, 45, 1, 21, 21},
                std::vector<T> {1, 18, 23, 1, 19, 21},
                std::vector<char> {1, 1, 0, 0, 0, 0}),
        RefComparisonParams(ComparisonTypes::LESS, ngraph::PartialShape {1}, ngraph::PartialShape {1},  type, ngraph::element::boolean,
                std::vector<T> {53},
                std::vector<T> {53},
                std::vector<char> {0}),
        RefComparisonParams(ComparisonTypes::LESS, ngraph::PartialShape {2, 4}, ngraph::PartialShape {2, 4}, type, ngraph::element::boolean,
                std::vector<T> {0, 12, 23, 0, 1, 5, 11, 8},
                std::vector<T> {0, 12, 23, 0, 10, 5, 11, 8},
                std::vector<char> {0, 0, 0, 0, 1, 0, 0, 0}),
        RefComparisonParams(ComparisonTypes::LESS, ngraph::PartialShape {3, 1, 2}, ngraph::PartialShape {1, 2, 1}, type, ngraph::element::boolean,
                std::vector<T> {2, 1, 4, 1, 3, 1},
                std::vector<T> {1, 1},
                std::vector<char> {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
        RefComparisonParams(ComparisonTypes::LESS, ngraph::PartialShape {2, 1, 2, 1}, ngraph::PartialShape {1, 2, 1}, type, ngraph::element::boolean,
                std::vector<T> {2, 1, 4, 1},
                std::vector<T> {1, 1},
                std::vector<char> {0, 0, 0, 0}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 2}, ngraph::PartialShape {2, 2}, type, ngraph::element::boolean,
                std::vector<T> {0, 12, 23, 0},
                std::vector<T> {0, 12, 23, 0},
                std::vector<char> {1, 1, 1, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 3}, ngraph::PartialShape {2, 3}, type, ngraph::element::boolean,
                std::vector<T> {0, 6, 45, 1, 21, 21},
                std::vector<T> {1, 18, 23, 1, 19, 21},
                std::vector<char> {0, 0, 0, 1, 0, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {1}, ngraph::PartialShape {1},  type, ngraph::element::boolean,
                std::vector<T> {53},
                std::vector<T> {53},
                std::vector<char> {1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 4}, ngraph::PartialShape {2, 4}, type, ngraph::element::boolean,
                std::vector<T> {0, 12, 23, 0, 1, 5, 11, 8},
                std::vector<T> {0, 12, 23, 0, 10, 5, 11, 8},
                std::vector<char> {1, 1, 1, 1, 0, 1, 1, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {3, 1, 2}, ngraph::PartialShape {1, 2, 1}, type, ngraph::element::boolean,
                std::vector<T> {2, 1, 4, 1, 3, 1},
                std::vector<T> {1, 1},
                std::vector<char> {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1}),
        RefComparisonParams(ComparisonTypes::EQUAL, ngraph::PartialShape {2, 1, 2, 1}, ngraph::PartialShape {1, 2, 1}, type, ngraph::element::boolean,
                std::vector<T> {2, 1, 4, 1},
                std::vector<T> {1, 1},
                std::vector<char> {0, 1, 0, 1})};
    return compParams;
}

std::vector<RefComparisonParams> generateComparisonCombinedParams() {
    const std::vector<std::vector<RefComparisonParams>> compTypeParams {
        generateComparisonParams<element::Type_t::f32>(ngraph::element::f32),
        generateComparisonParams<element::Type_t::f16>(ngraph::element::f16),
        generateComparisonParams<element::Type_t::i32>(ngraph::element::i32),
        generateComparisonParams<element::Type_t::u32>(ngraph::element::u32),
        generateComparisonParams<element::Type_t::u8>(ngraph::element::boolean)};
    std::vector<RefComparisonParams> combinedParams;
    std::for_each(compTypeParams.begin(), compTypeParams.end(), [&](std::vector<RefComparisonParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Comparison_With_Hardcoded_Refs, ReferenceComparisonLayerTest, ::testing::ValuesIn(generateComparisonCombinedParams()),
                                 ReferenceComparisonLayerTest::getTestCaseName);
}  // namespace
