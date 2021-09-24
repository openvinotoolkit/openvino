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
struct ExpParams {
    template <class IT>
    ExpParams(const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues, const std::vector<IT>& oValues)
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

class ReferenceExpLayerTest : public testing::TestWithParam<ExpParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ExpParams>& obj) {
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
        const auto Exp = std::make_shared<op::Exp>(in);
        return std::make_shared<Function>(NodeVector {Exp}, ParameterVector {in});
    }
};

class ReferenceExpInPlaceLayerTest : public testing::TestWithParam<ExpParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ExpParams>& obj) {
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
        const auto Exp = std::make_shared<op::Exp>(in);
        const auto ExpInPlace = std::make_shared<op::Exp>(Exp);
        return std::make_shared<Function>(NodeVector {ExpInPlace}, ParameterVector {in});
    }
};

TEST_P(ReferenceExpLayerTest, CompareWithRefs) {
    Exec();
}

TEST_P(ReferenceExpInPlaceLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ExpParams> generateExpFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ExpParams> expParams {
        ExpParams(ngraph::PartialShape {8},
                    IN_ET,
                    std::vector<T>{-4, -3, -2, -1, 0, 1, 2, 3},
                    std::vector<T>{expf(-4), expf(-3), expf(-2), expf(-1), expf(0), expf(1), expf(2), expf(3)}),
        ExpParams(ngraph::PartialShape {1},
                    IN_ET,
                    std::vector<T>{13},
                    std::vector<T>{expf(13)})
    };
    return expParams;
}

template <element::Type_t IN_ET>
std::vector<ExpParams> generateExpIntParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ExpParams> expParams {
        ExpParams(ngraph::PartialShape {8},
                    IN_ET,
                    std::vector<T>{-4, -3, -2, -1, 0, 1, 2, 3},
                    std::vector<T>{static_cast<T>(expf(-4)), static_cast<T>(expf(-3)), static_cast<T>(expf(-2)), static_cast<T>(expf(-1)),
                                   static_cast<T>(expf(0)), static_cast<T>(expf(1)), static_cast<T>(expf(2)), static_cast<T>(expf(3))}),
        ExpParams(ngraph::PartialShape {1},
                    IN_ET,
                    std::vector<T>{13},
                    std::vector<T>{static_cast<T>(expf(13))})
    };
    return expParams;
}

template <element::Type_t IN_ET>
std::vector<ExpParams> generateExpUintParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ExpParams> expParams {
        ExpParams(ngraph::PartialShape {8},
                    IN_ET,
                    std::vector<T>{0, 1, 2, 3, 4, 5, 10, 100},
                    std::vector<T>{static_cast<T>(expf(0)), static_cast<T>(expf(1)), static_cast<T>(expf(2)), static_cast<T>(expf(3)),
                                   static_cast<T>(expf(4)), static_cast<T>(expf(5)), static_cast<T>(expf(10)), static_cast<T>(expf(100))}),
        ExpParams(ngraph::PartialShape {1},
                    IN_ET,
                    std::vector<T>{13},
                    std::vector<T>{static_cast<T>(expf(13))})
    };
    return expParams;
}

template <element::Type_t IN_ET>
std::vector<ExpParams> generateExpInPlaceFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ExpParams> expParams {
        ExpParams(ngraph::PartialShape {2},
                    IN_ET,
                    std::vector<T>{1, 3},
                    std::vector<T>{expf(expf(1)), expf(expf(3))})
    };
    return expParams;
}

std::vector<ExpParams> generateExpCombinedParams() {
    const std::vector<std::vector<ExpParams>> expTypeParams {
        generateExpFloatParams<element::Type_t::f32>(),
        generateExpFloatParams<element::Type_t::f16>(),
        generateExpIntParams<element::Type_t::i32>(),
        generateExpIntParams<element::Type_t::i64>(),
        generateExpUintParams<element::Type_t::u32>(),
        generateExpUintParams<element::Type_t::u64>()
        };
    std::vector<ExpParams> combinedParams;

    for (const auto& params : expTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

std::vector<ExpParams> generateExpInPlaceCombinedParams() {
    const std::vector<std::vector<ExpParams>> expTypeParams {
        generateExpInPlaceFloatParams<element::Type_t::f16>(),
        generateExpInPlaceFloatParams<element::Type_t::f32>()
        };
    std::vector<ExpParams> combinedParams;

    for (const auto& params : expTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Exp_With_Hardcoded_Refs, ReferenceExpLayerTest,
    testing::ValuesIn(generateExpCombinedParams()), ReferenceExpLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Exp_In_Place_With_Hardcoded_Refs, ReferenceExpInPlaceLayerTest,
    testing::ValuesIn(generateExpInPlaceCombinedParams()), ReferenceExpInPlaceLayerTest::getTestCaseName);

} // namespace