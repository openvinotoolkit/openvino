// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "shared_test_classes/subgraph/parameter_result.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace SubgraphTestsDefinitions;
using namespace ov::test;
using namespace InferenceEngine;

namespace CPULayerTestsDefinitions {

class ParameterResultCustomBlobTest : public ParameterResultSubgraphTestLegacyApi {
protected:
    void Infer() override {
        constexpr size_t inferIterations = 10lu;

        inferRequest = executableNetwork.CreateInferRequest();

        auto inputBlob = inputs.front();
        const size_t elementsCount = inputBlob->size();
        for (size_t i = 0; i < inferIterations; ++i) {
            ov::test::utils::fill_data_random<Precision::FP32>(inputBlob, 10, 0, 1, i);
            auto inputsInfo = cnnNetwork.getInputsInfo().begin()->second;
            std::string inputName = cnnNetwork.getInputsInfo().begin()->first;

            std::vector<float> customInpData(elementsCount);
            auto inpBlobData = inputBlob->buffer().as<const float*>();
            std::copy(inpBlobData, inpBlobData + elementsCount, customInpData.begin());

            auto& tensorDesc = inputsInfo->getTensorDesc();
            auto customBlob = make_shared_blob<float>(tensorDesc, customInpData.data(), elementsCount);
            inferRequest.SetBlob(inputName, customBlob);

            inferRequest.Infer();

            ParameterResultSubgraphTestLegacyApi::Validate();
        }
    }
    void Validate() override {
        // Do nothing. We call Validate() in the Infer() method
    }
};

TEST_P(ParameterResultCustomBlobTest, CompareWithRefs) {
    // Just to show that it is not possible to set different precisions for inputs and outputs with the same name.
    // If it was possible, the input would have I8 precision and couldn't store data from the custom blob.
    inPrc = Precision::I8;
    outPrc = Precision::FP32;

    Run();
}

TEST_P(ParameterResultCustomBlobTest, CompareWithRefs_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16.to_string()});

    // Just to show that it is not possible to set different precisions for inputs and outputs with the same name.
    // If it was possible, the input would have I8 precision and couldn't store data from the custom blob.
    inPrc = Precision::I8;
    outPrc = Precision::FP32;

    Run();
}

namespace {
INSTANTIATE_TEST_SUITE_P(smoke_Check_Custom_Blob,
                         ParameterResultCustomBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::InputShape{{1, 3, 10, 10}, {{}}}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ParameterResultSubgraphTestBase::getTestCaseName);
}  // namespace

class ParameterResultSameBlobTest : public ParameterResultSubgraphTestLegacyApi {
protected:
    void Infer() override {
        constexpr size_t inferIterations = 10lu;

        for (size_t i = 0; i < inferIterations; ++i) {
            ParameterResultSubgraphTestLegacyApi::Infer();
            ParameterResultSubgraphTestLegacyApi::Validate();
        }
    }
    void Validate() override {
        // Do nothing. We call Validate() in the Infer() method
    }
};

TEST_P(ParameterResultSameBlobTest, CompareWithRefs) {
    Run();
}


TEST_P(ParameterResultSameBlobTest, CompareWithRefs_FP16) {
    if (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16())) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }
    configuration.insert({ov::hint::inference_precision.name(), ov::element::f16.to_string()});

    Run();
}
namespace {
INSTANTIATE_TEST_SUITE_P(smoke_Check_Same_Blob,
                         ParameterResultSameBlobTest,
                         ::testing::Combine(::testing::Values(ov::test::InputShape{{1, 3, 10, 10}, {{}}}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         ParameterResultSubgraphTestBase::getTestCaseName);
}  // namespace
}  // namespace CPULayerTestsDefinitions
