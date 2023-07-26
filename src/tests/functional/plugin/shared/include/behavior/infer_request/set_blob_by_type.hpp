// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <future>

#include "base/behavior_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

namespace BehaviorTestsDefinitions {

using InferRequestSetBlobByTypeParams = std::tuple<
        FuncTestUtils::BlobType,           // Blob type
        std::string,                       // Device name
        std::map<std::string, std::string> // Device config
>;

class InferRequestSetBlobByType : public testing::WithParamInterface<InferRequestSetBlobByTypeParams>,
                                  public BehaviorTestsUtils::IEInferRequestTestBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<InferRequestSetBlobByTypeParams> obj) {
        using namespace ov::test::utils;

        FuncTestUtils::BlobType BlobType;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(BlobType, targetDevice, configuration) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        std::ostringstream result;
        result << "BlobType=" << BlobType << "_";
        result << "Device="<< targetDevice << "_";
        result << "Config=" << configuration;
        return result.str();
    }

    void SetUp() override {
        std::map<std::string, std::string> config;
        std::tie(blobType, target_device, config) = this->GetParam();
        // Skip test according to plugin specific disabledTestPatterns() (if any)
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        APIBaseTest::SetUp();
        std::shared_ptr<ngraph::Function> function = ngraph::builder::subgraph::makeConvPoolRelu(
                {4, 3, 6, 8}, ngraph::element::Type_t::u8);
        InferenceEngine::CNNNetwork cnnNetwork(function);
        executableNetwork = ie->LoadNetwork(cnnNetwork, target_device, config);
    }

protected:
    bool blobTypeIsSupportedByDevice() {
        switch (blobType) {
            case FuncTestUtils::BlobType::Memory:
                return true;
            case FuncTestUtils::BlobType::Compound:
            case FuncTestUtils::BlobType::Remote:
                return false;
            case FuncTestUtils::BlobType::Batched: {
                auto supported_metrics = ie->GetMetric(target_device, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
                if (std::find(supported_metrics.begin(), supported_metrics.end(),
                              METRIC_KEY(OPTIMIZATION_CAPABILITIES)) == supported_metrics.end()) {
                    return false;
                }

               auto optimization_caps =
                        ie->GetMetric(target_device, METRIC_KEY(OPTIMIZATION_CAPABILITIES)).as<std::vector<std::string>>();
                return std::find(optimization_caps.begin(), optimization_caps.end(),
                                 METRIC_VALUE(BATCHED_BLOB)) != optimization_caps.end();
            }
            default:
                IE_THROW() << "Test does not support the blob kind";
        }
    }

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
