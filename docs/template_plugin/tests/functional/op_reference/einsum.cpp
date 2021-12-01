// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset1.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct EinsumParams {
    std::vector<Tensor> inputs;
    std::string equation;
    Tensor expectedResult;
    std::string testcaseName;
};

struct Builder : ParamsBuilder<EinsumParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, inputs);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, equation);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expectedResult);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, testcaseName);
};

class ReferenceEinsumTest : public testing::TestWithParam<EinsumParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        for (const auto& input_tensor : params.inputs) {
            inputData.push_back(input_tensor.data);
        }
        refOutData = {params.expectedResult.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<EinsumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "i0Type=" << param.inputs[0].type;
        result << "_i0Shape=" << param.inputs[0].shape;
        result << "_equation=" << param.equation;
        result << "_eType=" << param.expectedResult.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expectedResult.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_eShape=" << param.expectedResult.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const EinsumParams& params) {
        OutputVector output_vector;
        ParameterVector param_vector;
        for (const auto& input_tensor : params.inputs) {
            auto param = std::make_shared<opset1::Parameter>(input_tensor.type, input_tensor.shape);
            output_vector.push_back(param);
            param_vector.push_back(param);
        }
        const auto einsum = std::make_shared<opset7::Einsum>(output_vector, params.equation);
        const auto f = std::make_shared<Function>(OutputVector{einsum}, param_vector);
        return f;
    }
};

TEST_P(ReferenceEinsumTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<EinsumParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<EinsumParams> params {
        Builder {}
            .inputs({Tensor{ET, {3}, std::vector<T>{1, 2, 3}},
                     Tensor{ET, {3}, std::vector<T>{4, 5, 6}}})
            .equation("i,i->")
            .expectedResult(Tensor{ET, {}, std::vector<T>{32}})
    };
    return params;
}

std::vector<EinsumParams> generateCombinedParams() {
    const std::vector<std::vector<EinsumParams>> generatedParams {
        generateParams<element::Type_t::i8>(),
        generateParams<element::Type_t::i16>(),
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::i64>(),
        generateParams<element::Type_t::u8>(),
        generateParams<element::Type_t::u16>(),
        generateParams<element::Type_t::u32>(),
        generateParams<element::Type_t::u64>(),
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
    };
    std::vector<EinsumParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Einsum_With_Hardcoded_Refs, ReferenceEinsumTest,
    testing::ValuesIn(generateCombinedParams()), ReferenceEinsumTest::getTestCaseName);
} // namespace