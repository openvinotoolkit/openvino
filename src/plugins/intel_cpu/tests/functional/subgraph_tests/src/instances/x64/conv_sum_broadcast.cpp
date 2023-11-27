// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/src/classes/conv_sum_broadcast.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/convolution_params.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/builders.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

using namespace CPUTestUtils;
using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {
namespace ConvSumBroadcast {
const auto fusingMulAddFQMullAdd = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Multiply>(cfg.input, constNode);
        }, "Multiply(PerChannel)"},
        {[](postNodeConfig& cfg) {
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            auto constNode = ngraph::builder::makeConstant(cfg.type, newShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Add>(cfg.input, constNode);
        }, "Add(PerChannel)"}}), {"Add"} };

const auto fusingDivSubFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.input);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Divide>(cfg.input, secondMultInput);
        }, "Divide(PerChannel)"},
        {[](postNodeConfig& cfg){
            ngraph::Shape secondMultInShape = generatePerChannelShape(cfg.input);
            auto secondMultInput = ngraph::builder::makeConstant(cfg.type, secondMultInShape, std::vector<float>{}, true);
            return std::make_shared<ngraph::opset1::Subtract>(cfg.input, secondMultInput);
        }, "Subtract(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"} };

const auto fusingSigmoidFQFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Sigmoid);
        }, "Sigmoid"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"Sigmoid", "FakeQuantize", "FakeQuantize"} };

const auto fusingClampFQ = fusingSpecificParams{ std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg){
            return ngraph::builder::makeActivation(cfg.input, cfg.type, ngraph::helpers::Clamp, {}, {3.0f, 6.0f});
        }, "Clamp"},
        {[](postNodeConfig& cfg){
            auto localPrc = cfg.input->get_element_type();
            ngraph::Shape newShape = generatePerChannelShape(cfg.input);
            return ngraph::builder::makeFakeQuantize(cfg.input, localPrc, 256, newShape);
        }, "FakeQuantize(PerChannel)"}}), {"FakeQuantize"} };

const std::vector<fusingSpecificParams> fusingParamsSet{
        fusingSigmoid,
        fusingFakeQuantizePerTensorRelu,
        fusingFakeQuantizePerChannelRelu,
        fusingFQPerChannelSigmoidFQPerChannel,
        fusingReluScaleShift,
        fusingMulAddFQMullAdd,
        fusingSigmoidFQFQ,
        fusingDivSubFQ
};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{
        emptyFusingSpec,
        fusingSigmoid,
        fusingReluScaleShift
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_FP32_x64, ConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape()),
                                 ::testing::ValuesIn(secondInp()),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_BF16, ConvSumInPlaceTest,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape()),
                                 ::testing::ValuesIn(secondInp()),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvSumInPlaceTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Sum_Broadcast_INT8, ConvSumInPlaceTestInt8,
                         ::testing::Combine(
                                 ::testing::Values(convInpShape()),
                                 ::testing::ValuesIn(secondInp()),
                                 ::testing::Values(true, false),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvSumInPlaceTest::getTestCaseName);
} // namespace ConvSumBroadcast
} // namespace SubgraphTestsDefinitions
