// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"
#include "shared_test_classes/read_ir/read_ir.hpp"

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

InferenceEngine::Blob::Ptr ReadIRTest::generateROIblob(const InferenceEngine::InputInfo &info, const std::shared_ptr<ngraph::Node> node) const {
    InferenceEngine::Blob::Ptr blob;
    if (isRoiOperation(node) && info.getTensorDesc().getDims().size() == 1 &&
        ngraph::is_type<ngraph::op::v3::ROIAlign>(node)) {
            std::vector<int> roiIdxVector(node->get_shape()[0]);
            ROIAlignLayerTest::fillIdxTensor(roiIdxVector, node->get_shape()[0]);
            std::vector<float> a;
            blob = FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), roiIdxVector.data(), roiIdxVector.size());
    } else if (isRoiOperation(node) && info.getTensorDesc().getDims().size() == 2) {
        blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();
        const auto& inputShape = cnnNetwork.getInputShapes().begin()->second;

        if (ngraph::is_type<ngraph::op::v0::PSROIPooling>(node)) {
            const auto& PSROIPoolNode = std::dynamic_pointer_cast<ngraph::op::v0::PSROIPooling>(node);
            PSROIPoolingLayerTest::fillROITensor(blob->buffer(),
                                                 blob->size() / 5,
                                                 inputShape[0],
                                                 inputShape[2],
                                                 inputShape[3],
                                                 PSROIPoolNode->get_group_size(),
                                                 PSROIPoolNode->get_spatial_scale(),
                                                 PSROIPoolNode->get_spatial_bins_x(),
                                                 PSROIPoolNode->get_spatial_bins_y(),
                                                 PSROIPoolNode->get_mode());
        } else if (ngraph::is_type<ngraph::op::v0::ROIPooling>(node)) {
            const auto& ROIPoolNode = std::dynamic_pointer_cast<ngraph::op::v0::ROIPooling>(node);
            CommonTestUtils::fill_data_roi(blob->buffer(),
                                           blob->size(),
                                           ROIPoolNode->get_input_shape(0).front() - 1,
                                           inputShape[2],
                                           inputShape[3],
                                           1.0f,
                                           ROIPoolNode->get_method() == "max");
        } else if (ngraph::is_type<ngraph::op::v3::ROIAlign>(node)) {
            const auto &ROIAlignNode = std::dynamic_pointer_cast<ngraph::op::v3::ROIAlign>(node);
            std::vector<float> blobData(ROIAlignNode->get_shape()[0] * 4);
            ROIAlignLayerTest::fillCoordTensor(blobData,
                                               inputShape[2],
                                               inputShape[3],
                                               ROIAlignNode->get_spatial_scale(),
                                               ROIAlignNode->get_sampling_ratio(),
                                               ROIAlignNode->get_pooled_h(),
                                               ROIAlignNode->get_pooled_w());
            blob = FuncTestUtils::createAndFillBlobWithFloatArray(info.getTensorDesc(), blobData.data(), blobData.size());
        }
    } else {
        blob = GenerateInput(info);
    }
    return blob;
}

void ReadIRTest::Infer() {
    inferRequest = executableNetwork.CreateInferRequest();
    inputs.clear();

    const auto& inputsInfo = executableNetwork.GetInputsInfo();
    for (const auto& param : function->get_parameters()) {
        const auto infoIt = inputsInfo.find(param->get_friendly_name());
        GTEST_ASSERT_NE(infoIt, inputsInfo.cend());

        const auto& info = infoIt->second;

        InferenceEngine::Blob::Ptr blob(nullptr);
        for (size_t i = 0; i < param->get_output_size(); i++) {
            for (const auto& node : param->get_output_target_inputs(i)) {
                const auto nodePtr = node.get_node()->shared_from_this();
                if (isRoiOperation(nodePtr)) {
                    blob = generateROIblob(*info, nodePtr);
                    break;
                }
            }
        }
        if (!blob) {
            blob = GenerateInput(*info);
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

