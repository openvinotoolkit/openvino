// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"
#include "shared_test_classes/read_ir/read_ir.hpp"
#include "shared_test_classes/read_ir/generate_inputs.hpp"

namespace LayerTestsDefinitions {

static bool isRoiOperation(const std::shared_ptr<ngraph::Node>& op) {
    return (ngraph::is_type<ngraph::op::v0::PSROIPooling>(op) ||
            ngraph::is_type<ngraph::op::v0::ROIPooling>(op) ||
            ngraph::is_type<ngraph::op::v3::ROIAlign>(op));
}

std::string ReadIRTest::getTestCaseName(const testing::TestParamInfo<std::tuple<std::string, std::string>>& obj) {
    std::string pathToModel, deviceName;
    std::tie(pathToModel, deviceName) = obj.param;

    std::ostringstream result;
    result << "ModelPath=" << pathToModel << "_";
    result << "TargetDevice=" << deviceName << "_";
    return result.str();
}

void ReadIRTest::SetUp() {
    std::tie(pathToModel, targetDevice) = this->GetParam();
    cnnNetwork = getCore()->ReadNetwork(pathToModel);
    function = cnnNetwork.getFunction();
}

void ReadIRTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    auto inputMap = getInputMap();

    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    for (const auto& param : function->get_parameters()) {
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;

        InferenceEngine::Blob::Ptr blob(nullptr);
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto& node : param->get_output_target_inputs(i)) {
                const auto nodePtr = node.get_node()->shared_from_this();
                auto it = inputMap.find(nodePtr->get_type_info());
                if (it != inputMap.end()) {
                    blob = it->second(nodePtr, *info);
                } else {
                    blob = GenerateInput(*info);
                }
            }
        }
        inferRequest.SetBlob(info->name(), blob);
        inputs.push_back(blob);
    }
    if (configuration.count(InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_ENABLED) &&
        configuration.count(InferenceEngine::PluginConfigParams::YES)) {
        auto batchSize = executableNetwork.GetInputsInfo().begin()->second->getTensorDesc().getDims()[0] / 2;
        inferRequest.SetBatch(batchSize);
    }
    inferRequest.Infer();
}
} // namespace LayerTestsDefinitions

