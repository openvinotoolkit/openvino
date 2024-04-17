// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prior_box_clustered.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prior_box.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct PriorBoxClusteredParams {
    template <class IT>
    PriorBoxClusteredParams(const std::vector<float>& widths,
                            const std::vector<float>& heights,
                            const bool clip,
                            const ov::element::Type& iType,
                            const std::vector<IT>& layerShapeValues,
                            const std::vector<IT>& imageShapeValues,
                            const Shape& output_shape,
                            const std::vector<float>& oValues,
                            const std::vector<float>& variances = {},
                            const std::string& testcaseName = "")
        : inType(iType),
          outType(ov::element::Type_t::f32),
          layerShapeData(CreateTensor(Shape{2}, iType, layerShapeValues)),
          imageShapeData(CreateTensor(Shape{2}, iType, imageShapeValues)),
          refData(CreateTensor(output_shape, outType, oValues)),
          testcaseName(testcaseName) {
        attrs.widths = widths;
        attrs.heights = heights;
        attrs.clip = clip;
        if (variances.size() != 0)
            attrs.variances = variances;
    }

    ov::op::v0::PriorBoxClustered::Attributes attrs;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor layerShapeData;
    ov::Tensor imageShapeData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferencePriorBoxClusteredLayerTest : public testing::TestWithParam<PriorBoxClusteredParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<PriorBoxClusteredParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "variancesSize=" << param.attrs.variances.size() << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        if (param.testcaseName != "")
            result << "_" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PriorBoxClusteredParams& params) {
        auto LS = std::make_shared<op::v0::Constant>(params.layerShapeData);
        auto IS = std::make_shared<op::v0::Constant>(params.imageShapeData);
        const auto PriorBoxClustered = std::make_shared<op::v0::PriorBoxClustered>(LS, IS, params.attrs);
        return std::make_shared<ov::Model>(NodeVector{PriorBoxClustered}, ParameterVector{});
    }
};

TEST_P(ReferencePriorBoxClusteredLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<PriorBoxClusteredParams> generatePriorBoxClusteredFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<PriorBoxClusteredParams> priorBoxClusteredParams{
        PriorBoxClusteredParams(
            {3.0f},
            {3.0f},
            true,
            IN_ET,
            std::vector<T>{2, 2},
            std::vector<T>{10, 10},
            Shape{2, 16},
            std::vector<float>{0,    0,        0.15f, 0.15f,    0.34999f, 0,        0.64999f, 0.15f,
                               0,    0.34999f, 0.15f, 0.64999f, 0.34999f, 0.34999f, 0.64999f, 0.64999f,
                               0.1f, 0.1f,     0.1f,  0.1f,     0.1f,     0.1f,     0.1f,     0.1f,
                               0.1f, 0.1f,     0.1f,  0.1f,     0.1f,     0.1f,     0.1f,     0.1f}),
        PriorBoxClusteredParams(
            {3.0f},
            {3.0f},
            true,
            IN_ET,
            std::vector<T>{2, 2},
            std::vector<T>{10, 10},
            Shape{2, 16},
            std::vector<float>{0,    0,        0.15f, 0.15f,    0.34999f, 0,        0.64999f, 0.15f,
                               0,    0.34999f, 0.15f, 0.64999f, 0.34999f, 0.34999f, 0.64999f, 0.64999f,
                               0.1f, 0.2f,     0.3f,  0.4f,     0.1f,     0.2f,     0.3f,     0.4f,
                               0.1f, 0.2f,     0.3f,  0.4f,     0.1f,     0.2f,     0.3f,     0.4f},
            {0.1f, 0.2f, 0.3f, 0.4f}),
    };
    return priorBoxClusteredParams;
}

std::vector<PriorBoxClusteredParams> generatePriorBoxClusteredCombinedParams() {
    const std::vector<std::vector<PriorBoxClusteredParams>> priorBoxClusteredTypeParams{
        generatePriorBoxClusteredFloatParams<element::Type_t::i64>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::i32>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::i16>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::i8>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u64>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u32>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u16>(),
        generatePriorBoxClusteredFloatParams<element::Type_t::u8>(),
    };
    std::vector<PriorBoxClusteredParams> combinedParams;

    for (const auto& params : priorBoxClusteredTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_PriorBoxClustered_With_Hardcoded_Refs,
                         ReferencePriorBoxClusteredLayerTest,
                         testing::ValuesIn(generatePriorBoxClusteredCombinedParams()),
                         ReferencePriorBoxClusteredLayerTest::getTestCaseName);

}  // namespace
