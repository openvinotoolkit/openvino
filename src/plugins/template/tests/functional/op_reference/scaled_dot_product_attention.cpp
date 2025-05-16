// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct SDPAParams {
    PartialShape qShape;
    PartialShape kShape;
    PartialShape vShape;
    PartialShape attentionMaskShape;
    PartialShape outputShape;
    bool isCausal;
    std::string testcaseName;
    reference_tests::Tensor qData;
    reference_tests::Tensor kData;
    reference_tests::Tensor vData;
    reference_tests::Tensor attentionMaskData;
    reference_tests::Tensor expectedOutputData;
};

template <typename T, typename TMask>
SDPAParams PrepareTestCaseParams(const PartialShape& qShape,
                                 const PartialShape& kShape,
                                 const PartialShape& vShape,
                                 const PartialShape& attentionMaskShape,
                                 const PartialShape& outputShape,
                                 bool isCausal,
                                 const std::vector<T>& qData,
                                 const std::vector<T>& kData,
                                 const std::vector<T>& vData,
                                 const std::vector<TMask>& attentionMaskData,
                                 const std::vector<T>& expectedOutputData,
                                 const std::string& description) {
    SDPAParams ret;
    const auto elementType = element::from<T>();

    ret.qShape = qShape;
    ret.kShape = kShape;
    ret.vShape = vShape;
    ret.attentionMaskShape = attentionMaskShape;
    ret.outputShape = outputShape;
    ret.isCausal = isCausal;
    ret.testcaseName = description;
    ret.qData = reference_tests::Tensor(elementType, qShape.to_shape(), qData);
    ret.kData = reference_tests::Tensor(elementType, kShape.to_shape(), kData);
    ret.vData = reference_tests::Tensor(elementType, vShape.to_shape(), vData);
    ret.attentionMaskData =
        reference_tests::Tensor(element::from<TMask>(), attentionMaskShape.to_shape(), attentionMaskData);
    ret.expectedOutputData = reference_tests::Tensor(elementType, outputShape.to_shape(), expectedOutputData);
    return ret;
}

class ReferenceSDPATest : public testing::TestWithParam<SDPAParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.qData.data, params.kData.data, params.vData.data};
        if (params.attentionMaskShape.size() != 0) {
            inputData.push_back(params.attentionMaskData.data);
        }
        refOutData = {params.expectedOutputData.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SDPAParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "type=" << param.qData.data.get_element_type();
        result << "_qShape=" << param.qShape;
        result << "_kShape=" << param.kShape;
        result << "_vShape=" << param.vShape;
        result << "_attentionMaskShape=" << param.attentionMaskShape;
        result << "_outputShape=" << param.outputShape;
        result << "_isCausal=" << param.isCausal;
        result << "_=" << param.testcaseName;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const SDPAParams& params) {
        const auto q = std::make_shared<op::v0::Parameter>(params.qData.data.get_element_type(), params.qShape);
        const auto k = std::make_shared<op::v0::Parameter>(params.kData.data.get_element_type(), params.kShape);
        const auto v = std::make_shared<op::v0::Parameter>(params.vData.data.get_element_type(), params.vShape);

        OutputVector inputs = {q, k, v};
        ParameterVector paramsVec = {q, k, v};

        if (params.attentionMaskShape.size() != 0) {
            const auto attentionMask =
                std::make_shared<op::v0::Parameter>(params.attentionMaskData.data.get_element_type(),
                                                    params.attentionMaskShape);
            inputs.push_back(attentionMask);
            paramsVec.push_back(attentionMask);
        }

        const auto op = std::make_shared<op::v13::ScaledDotProductAttention>(inputs, params.isCausal);

        return std::make_shared<Model>(OutputVector{op}, paramsVec);
    }
};

TEST_P(ReferenceSDPATest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<SDPAParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<SDPAParams> params;

#define TEST_DATA(q_shape,                                                              \
                  k_shape,                                                              \
                  v_shape,                                                              \
                  attention_mask_shape,                                                 \
                  output_shape,                                                         \
                  is_causal,                                                            \
                  is_attention_mask_bool,                                               \
                  q_data,                                                               \
                  k_data,                                                               \
                  v_data,                                                               \
                  attention_mask_data,                                                  \
                  expected_output_data,                                                 \
                  description)                                                          \
    {                                                                                   \
        using TMask = typename std::conditional<is_attention_mask_bool, char, T>::type; \
        std::vector<TMask> attention_mask_data_vec = attention_mask_data;               \
        params.push_back(PrepareTestCaseParams<T, TMask>(q_shape,                       \
                                                         k_shape,                       \
                                                         v_shape,                       \
                                                         attention_mask_shape,          \
                                                         output_shape,                  \
                                                         is_causal,                     \
                                                         q_data,                        \
                                                         k_data,                        \
                                                         v_data,                        \
                                                         attention_mask_data_vec,       \
                                                         expected_output_data,          \
                                                         description));                 \
    }

#include "unit_test_utils/tests_data/scaled_dot_product_attention_data.h"
#undef TEST_DATA

    return params;
}

std::vector<SDPAParams> generateCombinedParams() {
    const std::vector<std::vector<SDPAParams>> generatedParams{generateParams<element::Type_t::f32>(),
                                                               generateParams<element::Type_t::f16>(),
                                                               generateParams<element::Type_t::f64>()};
    std::vector<SDPAParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SDPA_With_Hardcoded_Refs,
                         ReferenceSDPATest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceSDPATest::getTestCaseName);
}  // namespace
