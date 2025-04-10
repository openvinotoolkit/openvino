// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/markup_bias.hpp"
#include "low_precision/rt_info/bias_attribute.hpp"
#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/markup_bias.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;

class MarkupBiasTestParams {
public:
    ov::PartialShape input_shape;
    ov::PartialShape bias_shape;
    bool is_bias;
};

using MarkupBiasTestValues = std::tuple<ov::element::Type, MarkupBiasTestParams, std::string>;

class MarkupBiasTests : public testing::WithParamInterface<MarkupBiasTestValues>, public LayerTransformation {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MarkupBiasTestValues>& obj) {
        ov::element::Type precision;
        MarkupBiasTestParams test_values;
        std::string layer_type;
        std::tie(precision, test_values, layer_type) = obj.param;

        std::ostringstream result;
        result << precision << "IS=" << test_values.input_shape << "_bias_shape=" << test_values.bias_shape << "_"
               << layer_type << "_is_bias=" << test_values.is_bias;
        return result.str();
    }

protected:
    void SetUp() override {
        ov::element::Type precision;
        MarkupBiasTestParams test_values;
        std::string layer_type;
        std::tie(precision, test_values, layer_type) = GetParam();

        actualFunction = ov::builder::subgraph::MarkupBiasFunction::get(precision,
                                                                            test_values.input_shape,
                                                                            test_values.bias_shape,
                                                                            layer_type,
                                                                            false);
        SimpleLowPrecisionTransformer transformer;
        transformer.transform(actualFunction);
    }
};

TEST_P(MarkupBiasTests, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    const auto addOps = LayerTransformation::get<ov::op::v1::Add>(actualFunction);
    EXPECT_EQ(1ul, addOps.size()) << "unexpected addOps size";

    const bool is_bias = std::get<1>(GetParam()).is_bias;
    auto biasAttr = ov::pass::low_precision::getAttribute<ov::BiasAttribute>(addOps[0]);
    EXPECT_EQ(!biasAttr.empty(), is_bias) << "Bias markup failed";
}

namespace MarkupBiasTestsInstantiation {
std::vector<ov::element::Type> precisions = {
    ov::element::f32,
};

std::vector<MarkupBiasTestParams> test_params_4d = {
    {{1, 10, 16, 16}, {1, 10, 1, 1}, true},
    {{1, 10, 16, 16}, {1, 1, 1, 1}, true},
    {{1, 10, 16, 16}, {1, 10, 16, 16}, false},
    {{1, 10, 16, 16}, ov::PartialShape::dynamic(), false},
};

std::vector<std::string> layer_types_4d = {
    "Convolution",
    "GroupConvolution",
    "ConvolutionBackpropData",
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT_4D_Positive,
                         MarkupBiasTests,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(test_params_4d),
                                            ::testing::ValuesIn(layer_types_4d)),
                         MarkupBiasTests::getTestCaseName);

std::vector<MarkupBiasTestParams> test_params_2d = {
    {{1, 10}, {1, 10}, true},
    {{1, 10}, {1, 1}, true},
    {{1, 10}, ov::PartialShape::dynamic(), false},
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT_2D_Positive,
                         MarkupBiasTests,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::ValuesIn(test_params_2d),
                                            ::testing::Values("MatMulWithConstant")),
                         MarkupBiasTests::getTestCaseName);

}  // namespace MarkupBiasTestsInstantiation
