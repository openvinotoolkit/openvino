// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include "behavior_test_plugin.h"
#include <thread>

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + "_" + obj.param.input_blob_precision.name()
           + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
}
}

// Setting empty config to LoadNetwork doesn't throw
TEST_P(BehaviorPluginTestInferRequest, SetEmptyConfig) {
    InferenceEngine::Core core;

    const std::string device = GetParam().device;
    ASSERT_NO_THROW(core.SetConfig(GetParam().config, GetParam().device));

    InferenceEngine::CNNNetwork cnnNetwork = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);
    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    std::map<std::string, std::string> config;
    if (device.find(CommonTestUtils::DEVICE_MULTI) == std::string::npos &&
        device.find(CommonTestUtils::DEVICE_HETERO) == std::string::npos) {
        ASSERT_NO_THROW(exeNetwork = core.LoadNetwork(cnnNetwork, GetParam().device, config));
    } else {
        ASSERT_NO_THROW(exeNetwork = core.LoadNetwork(cnnNetwork, GetParam().device, GetParam().config));
    }
}

// Load incorrect network to Plugin to get executable network
TEST_P(BehaviorPluginTestInferRequest, canNotLoadNetworkToGetExeNetworkWithoutWeights) {
    InferenceEngine::Core core;
    CNNNetwork network = core.ReadNetwork(GetParam().model_xml_str, Blob::CPtr());

    ASSERT_THROW(core.LoadNetwork(network, GetParam().device, GetParam().config),
                 InferenceEngineException);
}

// Load correct network to Plugin to get executable network
TEST_P(BehaviorPluginTestInferRequest, canLoadCorrectNetworkToGetExecutable) {
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork cnnNetwork = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);
    ASSERT_NO_THROW(core.LoadNetwork(cnnNetwork, GetParam().device, GetParam().config));
}

TEST_P(BehaviorPluginTestInferRequest, CanCreateTwoExeNetworks) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork cnnNetwork = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);

    for (auto i = 0; i < 2; i++) {
        ASSERT_NO_THROW(core.LoadNetwork(cnnNetwork, GetParam().device, GetParam().config));
    }
}

TEST_P(BehaviorPluginTestInferRequest, CanCreateInferRequest) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
}

TEST_P(BehaviorPluginTestInferRequest, failToSetNullptrForInput) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr inputBlob = nullptr;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), inputBlob, &response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    std::string refError = NOT_ALLOCATED_str + "Failed to set empty blob with name: \'" + testEnv->inputName + "\'";
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetEmptyInputBlob) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), blob, &response);
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    std::string refError = NOT_ALLOCATED_str + "Failed to set empty blob with name: \'" + testEnv->inputName + "\'";
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetEmptyOutputBlob) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), blob, &response);
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    std::string refError = NOT_ALLOCATED_str + "Failed to set empty blob with name: \'" + testEnv->outputName + "\'";
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetNotAllocatedInput) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input = makeNotAllocatedBlob(GetParam().input_blob_precision,
                                           TensorDesc::getLayoutByDims(testEnv->inputDims), testEnv->inputDims);
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), input, &response));
    std::string refError = "Input data was not allocated. Input name: \'" + testEnv->inputName + "\'";
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetNotAllocatedOutput) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr output = makeNotAllocatedBlob(GetParam().input_blob_precision,
                                            TensorDesc::getLayoutByDims(testEnv->outputDims), testEnv->outputDims);
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), output, &response));
    std::string refError = "Input data was not allocated. Input name: \'" + testEnv->outputName + "\'";
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetBlobWithIncorrectName) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto input = makeNotAllocatedBlob(GetParam().input_blob_precision, TensorDesc::getLayoutByDims(testEnv->inputDims),
                                      testEnv->inputDims);
    input->allocate();
    sts = testEnv->inferRequest->SetBlob(FuncTestUtils::TestModel::incorrect_input_name, input, &response);
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    std::string refError =
            NOT_FOUND_str + "Failed to find input or output with name: \'" +
            FuncTestUtils::TestModel::incorrect_input_name + "\'";
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetInputWithIncorrectSizes) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    SizeVector incorrectSizes = testEnv->inputDims;
    /* to use 2x size of first dim to simulate using of an input blob of another size */
    incorrectSizes[0] *= 2;
    auto input = makeNotAllocatedBlob(GetParam().input_blob_precision, TensorDesc::getLayoutByDims(incorrectSizes),
                                      incorrectSizes);
    input->allocate();
    int in_size = std::accumulate(testEnv->inputDims.begin(), testEnv->inputDims.end(), 1, std::multiplies<int>());
    std::string refError = "Input blob size is not equal network input size (" + std::to_string(input->size()) + "!=" +
                           std::to_string(in_size) + ").";
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), input, &response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetOutputWithIncorrectSizes) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    SizeVector incorrectSizes = testEnv->outputDims;
    /* to use 2x size of first dim to simulate using of an output blob of another size */
    incorrectSizes[0] *= 2;
    Blob::Ptr output = _prepareOutputBlob(GetParam().input_blob_precision, incorrectSizes);
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), output, &response));
    int out_size = std::accumulate(testEnv->outputDims.begin(), testEnv->outputDims.end(), 1, std::multiplies<int>());
    std::string refError =
            "Output blob size is not equal network output size (" + std::to_string(output->size()) + "!=" +
            std::to_string(out_size) + ").";
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, failToSetInputBlobWithPrecisionNotMatchInputPrecision) {

    std::string refError;
    if (GetParam().device != CommonTestUtils::DEVICE_CPU) {
        // MKLDNNPlugin now supports input blobs with format other than the network format,
        // so there is no 'not corresponding user input precision' error

        refError =
                PARAMETER_MISMATCH_str + "Failed to set Blob with precision not corresponding to user input precision";
    } else {
        // ...but it still doesn't support Precision::UNSPECIFIED blobs.

        refError = PARAMETER_MISMATCH_str + "Failed to set Blob with precision";
    }

    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto inputBlob = prepareInputBlob(Precision::UNSPECIFIED, testEnv->inputDims);
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), inputBlob, &response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    response.msg[refError.length()] = '\0';

    if (GetParam().device != CommonTestUtils::DEVICE_CPU) {
        ASSERT_EQ(refError, response.msg);
    } else {
        ASSERT_STR_CONTAINS(response.msg, refError);
    }


}

TEST_P(BehaviorPluginTestInferRequest, failToSetOutputBlobWithPrecisionNotMatchOutputPrecision) {
    std::string refError =
            PARAMETER_MISMATCH_str + "Failed to set Blob with precision not corresponding to user output precision";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto outputBlob = _prepareOutputBlob(Precision::UNSPECIFIED, testEnv->outputDims);
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), outputBlob, &response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts);
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, canInferWithoutSetAndGetInOut) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedInputBlobAfterGetBlob) {
    std::string refError = "Input data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedInputBlobAfterGetBlobForAsync) {
    std::string refError = "Input data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->StartAsync(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedInputBlobAfterGetAndSetBlob) {
    std::string refError = "Input data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedInputBlobAfterSetBlob) {
    std::string refError = "Input data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto blob = makeNotAllocatedBlob(GetParam().input_blob_precision, TensorDesc::getLayoutByDims(testEnv->inputDims),
                                     testEnv->inputDims);
    blob->allocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedOutputBlobAfterGetBlob) {
    std::string refError = "Output data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedOutputBlobAfterGetBlobForAsync) {
    std::string refError = "Output data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->StartAsync(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedOutputBlobAfterGetAndSetBlob) {
    std::string refError = "Output data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr blob;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, canProcessDeallocatedOutputBlobAfterSetBlob) {
    std::string refError = "Output data was not allocated";
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto blob = makeNotAllocatedBlob(GetParam().output_blob_precision, TensorDesc::getLayoutByDims(testEnv->outputDims),
                                     testEnv->outputDims);
    blob->allocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), blob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    blob->deallocate();
    ASSERT_NO_THROW(sts = testEnv->inferRequest->Infer(&response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    EXPECT_THAT(std::string(response.msg), HasSubstr(refError));
}

TEST_P(BehaviorPluginTestInferRequest, DISABLED_secondCallGetOutputDoNotReAllocateData) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr getBlob1;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), getBlob1, &response));
    Blob::Ptr getBlob2;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), getBlob2, &response));
    ASSERT_EQ(getBlob1.get(), getBlob2.get());
}

TEST_P(BehaviorPluginTestInferRequest, CorrectOneAsyncInferWithGetInOutWithInfWait) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    Blob::Ptr result;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    sts = testEnv->inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), result, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(BehaviorPluginTestInferRequest, canStartAsyncInferWithGetInOutWithStatusOnlyWait) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    Blob::Ptr result;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    sts = testEnv->inferRequest->Wait(IInferRequest::WaitMode::STATUS_ONLY, &response);
    ASSERT_TRUE(sts == StatusCode::OK || StatusCode::RESULT_NOT_READY) << response.msg;
}

// Plugin correct infer request with allocating input and result BlobMaps inside plugin
TEST_P(BehaviorPluginTestInferRequest, FailedAsyncInferWithNegativeTimeForWait) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string refError = PARAMETER_MISMATCH_str;
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    Blob::Ptr result;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    ASSERT_NO_THROW(sts = testEnv->inferRequest->Wait(-2, &response));
    ASSERT_EQ(StatusCode::GENERAL_ERROR, sts) << response.msg;
    response.msg[refError.length()] = '\0';
    ASSERT_EQ(refError, response.msg);
}

TEST_P(BehaviorPluginTestInferRequest, canRun3SyncRequestsConsistentlyFromThreads) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    IInferRequest::Ptr inferRequest2;
    static_cast<IExecutableNetwork::Ptr &>(testEnv->exeNetwork)->CreateInferRequest(inferRequest2, &response);
    ASSERT_NE(inferRequest2, nullptr) << response.msg;
    IInferRequest::Ptr inferRequest3;
    static_cast<IExecutableNetwork::Ptr &>(testEnv->exeNetwork)->CreateInferRequest(inferRequest3, &response);
    ASSERT_NE(inferRequest3, nullptr) << response.msg;

    Blob::Ptr input1;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input1, &response);
    inferRequest2->SetBlob(testEnv->inputName.c_str(), input1, &response);
    inferRequest3->SetBlob(testEnv->inputName.c_str(), input1, &response);

    InferenceEngine::ResponseDesc response1, response2, response3;
    InferenceEngine::StatusCode sts1, sts2, sts3;
    std::thread t1([&] { sts1 = testEnv->inferRequest->Infer(&response1); });
    std::thread t2([&] { sts2 = inferRequest2->Infer(&response2); });
    std::thread t3([&] { sts3 = inferRequest3->Infer(&response3); });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_EQ((int) StatusCode::OK, sts1) << response1.msg;
    ASSERT_EQ((int) StatusCode::OK, sts2) << response2.msg;
    ASSERT_EQ((int) StatusCode::OK, sts3) << response3.msg;
}

TEST_P(BehaviorPluginTestInferRequest, canRun3AsyncRequestsConsistentlyWithWait) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    IInferRequest::Ptr inferRequest2;
    static_cast<IExecutableNetwork::Ptr &>(testEnv->exeNetwork)->CreateInferRequest(inferRequest2, &response);
    ASSERT_NE(inferRequest2, nullptr) << response.msg;
    IInferRequest::Ptr inferRequest3;
    static_cast<IExecutableNetwork::Ptr &>(testEnv->exeNetwork)->CreateInferRequest(inferRequest3, &response);
    ASSERT_NE(inferRequest3, nullptr) << response.msg;
    Blob::Ptr input1;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input1, &response);
    inferRequest2->SetBlob(testEnv->inputName.c_str(), input1, &response);
    inferRequest3->SetBlob(testEnv->inputName.c_str(), input1, &response);

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = inferRequest2->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = inferRequest3->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = testEnv->inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = inferRequest2->Wait(IInferRequest::WaitMode::RESULT_READY, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = inferRequest3->Wait(IInferRequest::WaitMode::RESULT_READY, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequest, canRun3AsyncRequestsConsistentlyFromThreadsWithoutWait) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    IInferRequest::Ptr inferRequest2;
    static_cast<IExecutableNetwork::Ptr &>(testEnv->exeNetwork)->CreateInferRequest(inferRequest2, &response);
    ASSERT_NE(inferRequest2, nullptr) << response.msg;
    IInferRequest::Ptr inferRequest3;
    static_cast<IExecutableNetwork::Ptr &>(testEnv->exeNetwork)->CreateInferRequest(inferRequest3, &response);
    ASSERT_NE(inferRequest3, nullptr) << response.msg;
    Blob::Ptr input1;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input1, &response);
    inferRequest2->SetBlob(testEnv->inputName.c_str(), input1, &response);
    inferRequest3->SetBlob(testEnv->inputName.c_str(), input1, &response);

    InferenceEngine::ResponseDesc response1, response2, response3;
    InferenceEngine::StatusCode sts1, sts2, sts3;
    std::thread t1([&] { sts1 = testEnv->inferRequest->StartAsync(&response1); });
    std::thread t2([&] { sts2 = inferRequest2->StartAsync(&response2); });
    std::thread t3([&] { sts3 = inferRequest3->StartAsync(&response3); });

    t1.join();
    t2.join();
    t3.join();

    ASSERT_EQ((int) StatusCode::OK, sts1) << response1.msg;
    ASSERT_EQ((int) StatusCode::OK, sts2) << response2.msg;
    ASSERT_EQ((int) StatusCode::OK, sts3) << response3.msg;
}

TEST_P(BehaviorPluginTestInferRequest, canWaitWithotStartAsync) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    sts = testEnv->inferRequest->Wait(IInferRequest::WaitMode::RESULT_READY, &response);
    ASSERT_EQ(StatusCode::INFER_NOT_STARTED, sts);
    sts = testEnv->inferRequest->Wait(IInferRequest::WaitMode::STATUS_ONLY, &response);
    ASSERT_EQ(StatusCode::INFER_NOT_STARTED, sts);
    sts = testEnv->inferRequest->Wait(1, &response);
    ASSERT_EQ(StatusCode::INFER_NOT_STARTED, sts);
}

TEST_P(BehaviorPluginTestInferRequest, returnDeviceBusyOnSetBlobAfterAsyncInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    ASSERT_EQ((int) StatusCode::OK, sts) << response.msg;

    sts = testEnv->inferRequest->Wait(IInferRequest::WaitMode::STATUS_ONLY, &response);
    ASSERT_EQ(StatusCode::INFER_NOT_STARTED, sts) << response.msg;

    std::map<std::string, InferenceEngineProfileInfo> perfMap;

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ((int) StatusCode::OK, sts) << response.msg;

    sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), input, &response);
    if (sts == StatusCode::REQUEST_BUSY) {
        ASSERT_TRUE(_wasDeviceBusy(response));
    } else {
        ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    }
    response.msg[0] = 0;

    sts = testEnv->inferRequest->Wait(IInferRequest::WaitMode::STATUS_ONLY, &response);
    ASSERT_TRUE(sts == StatusCode::OK || sts == StatusCode::RESULT_NOT_READY) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequest, returnDeviceBusyOnGetBlobAfterAsyncInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    ResponseDesc response2;

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response2);
    if (sts == StatusCode::REQUEST_BUSY)
        ASSERT_TRUE(_wasDeviceBusy(response2));
    else
        ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequest, returnDeviceBusyOnGetPerformanceCountAfterAsyncInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    std::map<std::string, InferenceEngineProfileInfo> perfMap;
    ResponseDesc response2;

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = testEnv->inferRequest->GetPerformanceCounts(perfMap, &response2);
    if (sts == StatusCode::REQUEST_BUSY)
        ASSERT_TRUE(_wasDeviceBusy(response2));
    else
        ASSERT_EQ(StatusCode::OK, sts);
}

TEST_P(BehaviorPluginTestInferRequest, returnDeviceBusyOnStartInferAfterAsyncInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    ResponseDesc response2;

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts);
    sts = testEnv->inferRequest->StartAsync(&response2);
    if (sts == StatusCode::REQUEST_BUSY)
        ASSERT_TRUE(_wasDeviceBusy(response2));
    else
        ASSERT_EQ(StatusCode::OK, sts);
}

TEST_P(BehaviorPluginTestInferRequest, returnDeviceBusyOnGetUserDataAfterAsyncInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    ResponseDesc response2;

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts);
    testEnv->inferRequest->GetUserData(nullptr, &response2);
    auto waitStatus = testEnv->inferRequest->Wait(IInferRequest::WaitMode::STATUS_ONLY, &response);
    if (waitStatus == StatusCode::RESULT_NOT_READY)
        ASSERT_TRUE(_wasDeviceBusy(response2));
    else
        ASSERT_TRUE(waitStatus == StatusCode::OK);
}

TEST_P(BehaviorPluginTestInferRequest, returnDeviceBusyOnSetUserDataAfterAsyncInfer) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    ResponseDesc response2;

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts);
    testEnv->inferRequest->SetUserData(nullptr, &response2);
    auto waitStatus = testEnv->inferRequest->Wait(IInferRequest::WaitMode::STATUS_ONLY, &response);
    if (waitStatus == StatusCode::RESULT_NOT_READY)
        ASSERT_TRUE(_wasDeviceBusy(response2));
    else
        ASSERT_TRUE(waitStatus == StatusCode::OK);
}
