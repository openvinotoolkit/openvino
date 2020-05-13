 // Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <functional>

#include <ie_core.hpp>
#include <ie_precision.hpp>

#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"

#include "single_layer_tests/select.hpp"

namespace LayerTestsDefinitions {

    std::string SelectLayerTest::getTestCaseName(const testing::TestParamInfo<selectTestParams> &obj) {
        std::vector<std::vector<size_t>> dataShapes(3);
        InferenceEngine::Precision dataType;
        ngraph::op::AutoBroadcastSpec broadcast;
        std::string targetDevice;
        std::tie(dataShapes, dataType, broadcast, targetDevice) = obj.param;
        std::ostringstream result;
        result << "COND=BOOL_" << CommonTestUtils::vec2str(dataShapes[0]);
        result << "_THEN=" << dataType.name() << "_" << CommonTestUtils::vec2str(dataShapes[1]);
        result << "_ELSE=" << dataType.name() << "_" << CommonTestUtils::vec2str(dataShapes[2]);
        result << "_" << broadcast.m_type;
        result << "_targetDevice=" << targetDevice;
        return result.str();
    }

    void SelectLayerTest::SetUp() {
        inputShapes.resize(NGraphFunctions::Select::numOfInputs);
        std::tie(inputShapes, inputPrecision, broadcast, targetDevice) = this->GetParam();
        layer = NGraphFunctions::Select(FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision), inputShapes, broadcast);
    }

    TEST_P(SelectLayerTest, CompareWithRefImpl) {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        InferenceEngine::CNNNetwork cnnNet(layer.fnPtr);

        auto outputName = cnnNet.getOutputsInfo().begin()->first;

        auto ie = PluginCache::get().ie();
        auto execNet = ie->LoadNetwork(cnnNet, targetDevice);
        auto req = execNet.CreateInferRequest();

        std::vector<InferenceEngine::Blob::Ptr> inBlobs;

        std::vector<uint32_t> range = {2, 30, 30};
        std::vector<int32_t> startFrom = {0, 0, 30};
        int i = 0;
        for (const auto &inputItem : cnnNet.getInputsInfo()) {
            auto currentBlob = FuncTestUtils::createAndFillBlob(inputItem.second->getTensorDesc(), range[i], startFrom[i]);
            req.SetBlob(inputItem.first, currentBlob);
            inBlobs.push_back(currentBlob);
            i++;
        }

        std::vector<InferenceEngine::Blob::Ptr> castedBlobs = inBlobs;
        std::vector<const float *> inRawData;
        for (size_t i = 0; i < castedBlobs.size(); i++) {
            castedBlobs[i] = FuncTestUtils::copyBlobWithCast<InferenceEngine::Precision::FP32>(inBlobs[i]);
            inRawData.push_back(castedBlobs[i]->cbuffer().as<float *>());
        }

        req.Infer();

        auto outBlob = req.GetBlob(outputName);
        auto resShape = outBlob->getTensorDesc().getDims();
        const auto& outPrecision = outBlob->getTensorDesc().getPrecision();

        size_t outElementsCount = std::accumulate(begin(resShape), end(resShape), 1, std::multiplies<size_t>());
        std::vector<float> refOutData = layer.RefImpl<float>(inRawData, inputShapes, resShape);

        if (outPrecision != InferenceEngine::Precision::I32 && outPrecision != InferenceEngine::Precision::FP32)
            THROW_IE_EXCEPTION << "Test for select layer doesn't support output precision different from I32 or FP32";

        if (outPrecision == InferenceEngine::Precision::I32) {
            std::vector<int32_t> convRefOutData(outElementsCount);
            for (size_t i = 0; i < outElementsCount; i++)
                convRefOutData[i] = static_cast<int32_t>(refOutData[i]);
            FuncTestUtils::compareRawBuffers(outBlob->cbuffer().as<int32_t *>(), convRefOutData.data(),
                    outElementsCount, outElementsCount, FuncTestUtils::CompareType::ABS_AND_REL);
        } else {
            float thr1, thr2;
            FuncTestUtils::GetComparisonThreshold(InferenceEngine::Precision::FP32, thr1, thr2);
            FuncTestUtils::compareRawBuffers(outBlob->cbuffer().as<float *>(), refOutData.data(),
                                                             outElementsCount, outElementsCount,
                                                             FuncTestUtils::CompareType::ABS_AND_REL,
                                                             thr1, thr2);
        }

        layer.fnPtr.reset();
    }

}  // namespace LayerTestsDefinitions
