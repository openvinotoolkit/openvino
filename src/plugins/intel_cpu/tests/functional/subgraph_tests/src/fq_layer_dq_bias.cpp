// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/markup_bias_function.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"

using namespace ngraph;
using namespace ov::test;
using namespace CPUTestUtils;
using namespace InferenceEngine;

namespace SubgraphTestsDefinitions {
using FQLayerDQBiasParams = std::tuple<InputShape, std::string>;

class FQLayerDQBias : virtual public SubgraphBaseTest,
                      public CpuTestWithFusing,
                      public testing::WithParamInterface<FQLayerDQBiasParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<FQLayerDQBiasParams> obj) {
        InputShape input_shape;
        std::string layer_type;
        std::tie(input_shape, layer_type) = obj.param;

        std::ostringstream result;
        result << "IS=(" << CommonTestUtils::partialShape2str({input_shape.first}) << ")_TS=(";
        for (const auto& item : input_shape.second) {
            result << CommonTestUtils::vec2str(item) << "_";
        }
        result << ")_layer_type=" << layer_type;
        return result.str();
    }

protected:
    void SetUp() override {
        InputShape input_shape;
        std::string layer_type;
        std::tie(input_shape, layer_type) = GetParam();

        targetDevice = CommonTestUtils::DEVICE_CPU;
        fusedOps = std::vector<std::string>{"Add"};
        std::tie(inFmts, outFmts, priority, selectedType) = CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type};
        std::unordered_map<std::string, std::string> ngraph_type_to_plugin_type{
            {"Convolution", "Convolution"},
            {"GroupConvolution", "Convolution"},
            {"ConvolutionBackpropData", "Deconvolution"},
            {"MatMul", "MatMul"},
            {"MatMulWithConstant", "FullyConnected"},
        };
        node_type = ngraph_type_to_plugin_type[layer_type];

        const auto shapes = layer_type == "MatMul" ? std::vector<InputShape>{input_shape, input_shape}
                                                   : std::vector<InputShape>{input_shape};
        init_input_shapes(shapes);
        function = ngraph::builder::subgraph::MarkupBiasFunction::get(ov::element::f32, inputDynamicShapes[0], {}, layer_type);
    }

    std::string node_type;
};

TEST_P(FQLayerDQBias, smoke_CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, node_type);
}

namespace {
const std::vector<InputShape> input_shapes_4D_static = {
    {{}, {{1, 3, 1, 1}}},
    {{}, {{1, 3, 64, 64}}}
};

const std::vector<std::string> layer_types_4D_static = {
    "Convolution",
    "GroupConvolution",
    "ConvolutionBackpropData",
    "MatMul",
};

INSTANTIATE_TEST_SUITE_P(smoke_FQLayerDQBias_4D_static, FQLayerDQBias,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_4D_static),
                                            ::testing::ValuesIn(layer_types_4D_static)),
                         FQLayerDQBias::getTestCaseName);

const std::vector<InputShape> input_shapes_4D_dynamic = {
    {{-1, 3, -1, -1}, {{1, 3, 64, 64}}}
};

const std::vector<std::string> layer_types_4D_dynamic = {
    "Convolution",
    "GroupConvolution",
    "MatMul",
};

INSTANTIATE_TEST_SUITE_P(smoke_FQLayerDQBias_4D_dynamic, FQLayerDQBias,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_4D_dynamic),
                                            ::testing::ValuesIn(layer_types_4D_dynamic)),
                         FQLayerDQBias::getTestCaseName);

const std::vector<InputShape> input_shapes_2D = {
    {{-1, 768}, {{1, 768}}}
};

const std::vector<std::string> layer_types_2D = {
    "MatMulWithConstant",
};

INSTANTIATE_TEST_SUITE_P(smoke_FQLayerDQBias_2D, FQLayerDQBias,
                         ::testing::Combine(::testing::ValuesIn(input_shapes_2D),
                                            ::testing::ValuesIn(layer_types_2D)),
                         FQLayerDQBias::getTestCaseName);

} // namespace
} // namespace SubgraphTestsDefinitions
