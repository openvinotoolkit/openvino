// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "base/behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

namespace BehaviorTestsDefinitions {
using namespace CommonTestUtils;

using InferRequestSetBlobByTypeParams = std::tuple<
        FuncTestUtils::BlobType,           // Blob type
        std::string,                       // Device name
        std::map<std::string, std::string> // Device config
>;

class InferRequestSetBlobByType : public testing::WithParamInterface<InferRequestSetBlobByTypeParams>,
                                  public CommonTestUtils::TestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestSetBlobByTypeParams> obj) {
        FuncTestUtils::BlobType BlobType;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(BlobType, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "BlobType=" << BlobType << "_";
        result << "Device="<< targetDevice << "_";
        result << "Config=" << configuration;
        return result.str();
    }

    void SetUp() override {
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        std::map<std::string, std::string> config;
        std::tie(blobType, targetDevice, config) = this->GetParam();
        std::shared_ptr<ngraph::Function> function = ngraph::builder::subgraph::makeConvPoolRelu(
                {4, 3, 6, 8}, ngraph::element::Type_t::u8);
        InferenceEngine::CNNNetwork cnnNetwork(function);
        executableNetwork = ie->LoadNetwork(cnnNetwork, targetDevice, config);
    }

protected:
    bool blobTypeIsSupportedByDevice() {
        switch (blobType) {
            case FuncTestUtils::BlobType::Memory:
                return true;
            case FuncTestUtils::BlobType::Compound:
            case FuncTestUtils::BlobType::I420:
//            case FuncTestUtils::BlobType::Remote:
            case FuncTestUtils::BlobType::NV12:
                return false;
            case FuncTestUtils::BlobType::Batched: {
                std::vector<std::string> supported_metrics = ie->GetMetric(targetDevice, METRIC_KEY(SUPPORTED_METRICS));
                if (std::find(supported_metrics.begin(), supported_metrics.end(),
                              METRIC_KEY(OPTIMIZATION_CAPABILITIES)) == supported_metrics.end()) {
                    return false;
                }

                std::vector<std::string> optimization_caps =
                        ie->GetMetric(targetDevice, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
                return std::find(optimization_caps.begin(), optimization_caps.end(),
                                 METRIC_VALUE(BATCHED_BLOB)) != optimization_caps.end();
            }
            default:
                IE_THROW() << "Test does not support the blob kind";
        }
    }

    std::string targetDevice;
    FuncTestUtils::BlobType blobType;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
};

TEST_P(InferRequestSetBlobByType, setInputBlobsByType) {
    // Create InferRequest
    auto req = executableNetwork.CreateInferRequest();
    for (const auto &input : executableNetwork.GetInputsInfo()) {
        const auto &info = input.second;
        auto blob = FuncTestUtils::createBlobByType(info->getTensorDesc(), blobType);
        if (blobTypeIsSupportedByDevice()) {
            EXPECT_NO_THROW(req.SetBlob(info->name(), blob));
        } else {
            EXPECT_THROW(req.SetBlob(info->name(), blob), InferenceEngine::Exception);
        }
    }
}
} // namespace BehaviorTestsDefinitions
