// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/single_layer/reduce_ops.hpp"
#include "shared_test_classes/single_layer/convolution_backprop_data.hpp"
#include <shared_test_classes/single_layer/concat.hpp>
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
> reduceDeconvConcatDynamicGPUTestParamsSet;

const std::vector<ElementType> netPrecisions = {
    ElementType::f16,
};

// Reduce should have preferred format for ouput layout
class ReduceDeconvConcatDynamicGPUTest : public testing::WithParamInterface<reduceDeconvConcatDynamicGPUTestParamsSet>,
                             virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<reduceDeconvConcatDynamicGPUTestParamsSet>& obj) {
        reduceDeconvConcatDynamicGPUTestParamsSet basicParamsSet = obj.param;
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
        reduceDeconvConcatDynamicGPUTestParamsSet basicParamsSet = this->GetParam();
        std::vector<InputShape> inputShapes;
        ElementType netType;
        std::map<std::string, std::string> additionalConfig;
        std::tie(inputShapes, netType, targetDevice, additionalConfig) = basicParamsSet;

        init_input_shapes(inputShapes);

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes)
            params.push_back(std::make_shared<ov::op::v0::Parameter>(netType, shape));

        auto deconvOp = ngraph::builder::makeConvolutionBackpropData(params[0], netType, {2, 2, 2}, {2, 2, 2}, {0, 0, 0},
                                                                  {0, 0, 0}, {1, 1, 1}, ov::op::PadType::EXPLICIT, 16);
        deconvOp->set_friendly_name("deconv");

        std::vector<int> reduce_axes = {5};
        auto reduceAxesNode = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape({1}), reduce_axes));
        auto reduceOp = ngraph::builder::makeReduce(params[1], reduceAxesNode, false, ngraph::helpers::ReductionType::Max);
        reduceOp->set_friendly_name("reduce");

        auto concatOp = ngraph::builder::makeConcat({deconvOp, reduceOp}, 1);
        concatOp->set_friendly_name("concat");

        std::vector<int> transpose_order = {0, 1, 2, 4, 3};
        auto transposeOrderNode = std::dynamic_pointer_cast<ngraph::Node>(
                                 std::make_shared<ngraph::opset3::Constant>(ngraph::element::Type_t::i64, ngraph::Shape({5}), transpose_order));
        auto transposeOp = std::make_shared<ngraph::opset3::Transpose>(concatOp, transposeOrderNode);
        transposeOp->set_friendly_name("transpose");

        ngraph::ResultVector results = {std::make_shared<ngraph::opset1::Result>(transposeOp)};
        function = std::make_shared<ngraph::Function>(results, params, "transpose_out");
    }
};


TEST_P(ReduceDeconvConcatDynamicGPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

namespace {
std::map<std::string, std::string> emptyAdditionalConfig;
const std::vector<std::vector<ov::test::InputShape>> dynInputShapes = {
    {
        // Input for Deconv
        {{1, 32, 64, ov::Dimension::dynamic(), ov::Dimension::dynamic()}, {{1, 32, 64, 64, 64}}},
        // Input for Reduce
        {{1, 8, 128, ov::Dimension::dynamic(), ov::Dimension::dynamic(), 4}, {{1, 8, 128, 128, 128, 4}}}
    }
};


const auto testParams_smoke = ::testing::Combine(::testing::ValuesIn(dynInputShapes),
                                                   ::testing::ValuesIn(netPrecisions), // netprec
                                                   ::testing::Values(ov::test::utils::DEVICE_GPU),
                                                   ::testing::Values(emptyAdditionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_reduce_deconv_concat, ReduceDeconvConcatDynamicGPUTest,
                         testParams_smoke, ReduceDeconvConcatDynamicGPUTest::getTestCaseName);
} // namespace
} // namespace GPULayerTestsDefinitions
