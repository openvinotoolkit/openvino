// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/runtime_precision.hpp"

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/binary_convolution.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/runtime/exec_model_info.hpp"

#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/relu.hpp"

namespace ExecutionGraphTests {

std::shared_ptr<ov::Model> makeEltwiseFunction(const std::vector<ov::element::Type>& inputPrecisions) {
    OPENVINO_ASSERT(inputPrecisions.size() == 2);

    ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0],
                                                                       ov::Shape{1, 16, 5, 4}),
                               std::make_shared<ov::op::v0::Parameter>(inputPrecisions[1],
                                                                       ov::Shape{1, 16, 5, 4})};

    auto eltwise = ov::test::utils::make_eltwise(inputs[0], inputs[1], ov::test::utils::EltwiseTypes::ADD);
    eltwise->set_friendly_name("Eltwise");

    auto function = std::make_shared<ov::Model>(eltwise, inputs, "EltwiseWithTwoDynamicInputs");
    return function;
}

std::shared_ptr<ov::Model> makeFakeQuantizeReluFunction(const std::vector<ov::element::Type>& inputPrecisions) {
    OPENVINO_ASSERT(inputPrecisions.size() == 1);

    ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], ov::Shape{1, 16, 5, 4})};
    auto inputLowNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{0});
    auto inputHighNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{255});
    auto outputLowNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{0});
    auto outputHighNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{255});
    auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(inputs[0], inputLowNode, inputHighNode, outputLowNode, outputHighNode, 256);
    fakeQuantize->set_friendly_name("FakeQuantize");

    auto relu = std::make_shared<ov::op::v0::Relu>(fakeQuantize);
    relu->set_friendly_name("Relu");

    auto function = std::make_shared<ov::Model>(relu, inputs, "FakeQuantizeRelu");
    return function;
}

std::shared_ptr<ov::Model> makeFakeQuantizeBinaryConvolutionFunction(const std::vector<ov::element::Type> &inputPrecisions) {
    OPENVINO_ASSERT(inputPrecisions.size() == 1);

    ov::ParameterVector inputs{std::make_shared<ov::op::v0::Parameter>(inputPrecisions[0], ov::Shape{1, 16, 5, 4})};
    auto inputLowNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{1});
    auto inputHighNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{1});
    auto outputLowNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{0});
    auto outputHighNode = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, std::vector<float>{1});
    auto fakeQuantize = std::make_shared<ov::op::v0::FakeQuantize>(inputs[0], inputLowNode, inputHighNode, outputLowNode, outputHighNode, 2);
    fakeQuantize->set_friendly_name("FakeQuantize");

    auto binConv = ov::test::utils::make_binary_convolution(fakeQuantize, {3, 3}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, ov::op::PadType::EXPLICIT, 32, 0);
    binConv->set_friendly_name("BinaryConvolution");

    auto function = std::make_shared<ov::Model>(binConv, inputs, "FakeQuantizeBinaryConvolution");
    return function;
}

std::string ExecGraphRuntimePrecision::getTestCaseName(testing::TestParamInfo<ExecGraphRuntimePrecisionParams> obj) {
    RuntimePrecisionSpecificParams specificParams;
    std::string targetDevice;
    std::tie(specificParams, targetDevice) = obj.param;

    std::ostringstream result;
    result << "Function=" << specificParams.makeFunction(specificParams.inputPrecisions)->get_friendly_name() << "_";
    result << "InPrcs=" << ov::test::utils::vec2str(specificParams.inputPrecisions) << "_";
    result << "targetDevice=" << targetDevice;

    return result.str();
}

void ExecGraphRuntimePrecision::SetUp() {
    RuntimePrecisionSpecificParams specificParams;
    std::tie(specificParams, targetDevice) = this->GetParam();
    expectedPrecisions = specificParams.expectedPrecisions;
    fnPtr = specificParams.makeFunction(specificParams.inputPrecisions);
}

void ExecGraphRuntimePrecision::TearDown() {
}

TEST_P(ExecGraphRuntimePrecision, CheckRuntimePrecision) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto core = ov::test::utils::PluginCache::get().core();
    auto execNet = core->compile_model(fnPtr, targetDevice);
    const auto execGraph = execNet.get_runtime_model();

    auto ops = execGraph->get_ops();
    for (auto expectedPrc : expectedPrecisions) {
        auto opIter = std::find_if(ops.begin(), ops.end(), [&expectedPrc](std::shared_ptr<ov::Node> op) {
            return op->get_friendly_name() == expectedPrc.first;
        });

        if (opIter == ops.end())
            FAIL() << "Execution graph doesn't contain node with name: " << expectedPrc.first;

        const auto& rtInfo = opIter->get()->get_rt_info();
        const auto& rtIter = rtInfo.find("runtimePrecision");

        if (rtIter == rtInfo.end())
            FAIL() << "Runtime precision is not found for node: " << opIter->get()->get_friendly_name();

        if (expectedPrc.second.to_string() != rtIter->second.as<std::string>())
            FAIL() << "`" << expectedPrc.first << "' node runtime precision mismatch: actual = " <<
                rtIter->second.as<std::string>() << ", expected = " << expectedPrc.second;
    }

    fnPtr.reset();
};

}  // namespace ExecutionGraphTests
