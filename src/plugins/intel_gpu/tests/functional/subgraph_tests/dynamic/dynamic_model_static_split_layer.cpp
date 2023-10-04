// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/split.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace ngraph;
using namespace InferenceEngine;
using namespace ov::test;

namespace GPULayerTestsDefinitions {

typedef std::tuple<
        std::vector<InputShape>, // input shapes
        ElementType, // Network precision
        TargetDevice, // Device name
        std::map<std::string, std::string> // Additional network configuration
> DynamicModelStaticSplitLayerGPUTestParamsSet;

const std::vector<ElementType> netPrecisions = {
    ElementType::f16
};

class DynamicModelStaticSplitLayerGPUTest : public testing::WithParamInterface<DynamicModelStaticSplitLayerGPUTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicModelStaticSplitLayerGPUTestParamsSet>& obj) {
        DynamicModelStaticSplitLayerGPUTestParamsSet basicParamsSet = obj.param;
        std::ostringstream result;
        std::vector<InputShape> inputShapes;
        ElementType netType;
        TargetDevice targetDevice;
        std::map<std::string, std::string> additionalConfig;

        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
            for (const auto& actual_shape : shape.second) {
                result << ov::test::utils::partialShape2str({actual_shape}) << "_";
            }
        }
        result << "NetType=" << netType << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
     void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override {
          inputs.clear();
          const auto& funcInputs = function->inputs();
          for (size_t i = 0; i < funcInputs.size(); ++i) {
              const auto& funcInput = funcInputs[i];
              ov::Tensor tensor;
              tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                               targetInputStaticShapes[i],
                                                               80,
                                                               0,
                                                               8);
              inputs.insert({funcInput.get_node_shared_ptr(), tensor});
          }
    }

    void SetUp() override {
        DynamicModelStaticSplitLayerGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        ElementType netType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(inputShapes);
        const auto inShapSplit = inputDynamicShapes[0];
        const auto inShapeElt = inputDynamicShapes[1];
        ov::ParameterVector params;
        for (auto&& shape : {inShapSplit, inShapeElt}) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));
        }
        auto paramOuts = helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));

        auto axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        axis->set_friendly_name("axis");

        auto split_sizes = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 1});
        split_sizes->set_friendly_name("split_sizes");

        auto variadicSplitOp = std::make_shared<ngraph::opset1::VariadicSplit>(paramOuts[0], axis, split_sizes);
        variadicSplitOp->set_friendly_name("variadicSplit");

        auto addOp = ngraph::builder::makeEltwise(paramOuts[1], variadicSplitOp->output(1), ngraph::helpers::EltwiseTypes::ADD);
        addOp->set_friendly_name("add");

        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(addOp)};
        function = std::make_shared<ngraph::Function>(results, params, "eltwise_add_out");
    }
};


TEST_P(DynamicModelStaticSplitLayerGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
std::map<std::string, std::string> emptyAdditionalConfig;
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        // Input for static VariadicSplit
        {{2}, {{2}, {2}}},
        // Input for dynamic Add
        {{-1, -1}, {{5, 10}, {2, 3}}},
    },
    {
        // Input for static VariadicSplit
        {{2, 5}, {{2, 5}, {2, 5}}},
        // Input for dynamic Add
        {{-1, -1, -1}, {{1, 5, 1}, {1, 1, 1}}},
    },
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(netPrecisions), // netprec
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_model_static_split, DynamicModelStaticSplitLayerGPUTest,
                         testParams_smoke, DynamicModelStaticSplitLayerGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
