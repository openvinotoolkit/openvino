// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/node_builders/binary_convolution.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/add.hpp"

#include "execution_graph_tests/num_inputs_fusing_bin_conv.hpp"

namespace ExecutionGraphTests {

std::string ExecGraphInputsFusingBinConv::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::string targetDevice = obj.param;
    return "targetDevice=" + targetDevice;
}

void ExecGraphInputsFusingBinConv::SetUp() {
    const std::vector<size_t> inputShapes = { 1, 16, 30, 30}, binConvKernelSize = {2, 2}, convKernelSize = {3, 3};
    const size_t numOutChannels = 16, numGroups = 16;
    const std::vector<size_t> strides = {1, 1}, dilations = {1, 1};
    const std::vector<ptrdiff_t> padsBegin = {1, 1}, padsEnd = {0, 0};
    const ov::op::PadType paddingType = ov::op::PadType::EXPLICIT;
    const float padValue = 1.0;

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShapes))};
    auto binConv = ov::test::utils::make_binary_convolution(params[0], binConvKernelSize, strides, padsBegin, padsEnd, dilations, paddingType, numOutChannels,
                                                          padValue);
    auto conv = ov::test::utils::make_group_convolution(binConv, ov::element::f32, convKernelSize, strides, padsBegin, padsEnd, dilations, paddingType,
                                                      numOutChannels, numGroups);

    auto biasNode = std::make_shared<ov::op::v0::Constant>(ov::element::f32, std::vector<size_t>{16, 1, 1});
    auto add = std::make_shared<ov::op::v1::Add>(conv, biasNode);
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(add)};
    ov_model = std::make_shared<ov::Model>(results, params, "BinConvFuseConv");
}

void ExecGraphInputsFusingBinConv::TearDown() {
}

TEST_P(ExecGraphInputsFusingBinConv, CheckNumInputsInBinConvFusingWithConv) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto targetDevice = this->GetParam();

    auto core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(ov_model, targetDevice);

    auto runtime_model = compiled_model.get_runtime_model();
    ASSERT_NE(runtime_model, nullptr);

    for (const auto & op : runtime_model->get_ops()) {
        const auto & rtInfo = op->get_rt_info();
        auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
            auto it = rtInfo.find(paramName);
            OPENVINO_ASSERT(rtInfo.end() != it);
            return it->second.as<std::string>();
        };

        auto layerType = getExecValue("layerType");
        if (layerType == "BinaryConvolution") {
            auto originalLayersNames = getExecValue("originalLayersNames");
            ASSERT_TRUE(originalLayersNames.find("BinaryConvolution") != std::string::npos);
            ASSERT_EQ(op->get_input_size(), 2);
        }
    }

    ov_model.reset();
};

}  // namespace ExecutionGraphTests
