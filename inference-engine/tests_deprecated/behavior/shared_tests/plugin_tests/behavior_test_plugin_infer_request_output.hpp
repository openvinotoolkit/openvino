// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
std::string getOutputTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + "_" + obj.param.output_blob_precision.name()
           + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
}

}

TEST_P(BehaviorPluginTestInferRequestOutput, canSetOutputBlobForAsyncRequest) {
    TestEnv::Ptr testEnv;
    Blob::Ptr actualBlob;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto outputBlob = _prepareOutputBlob(GetParam().output_blob_precision, testEnv->outputDims);

    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), outputBlob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_NO_THROW(testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), actualBlob, &response));

    ASSERT_EQ(outputBlob, actualBlob);
}

TEST_P(BehaviorPluginTestInferRequestOutput, canSetOutputBlobForSyncRequest) {
    TestEnv::Ptr testEnv;
    Blob::Ptr actualBlob;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto outputBlob = _prepareOutputBlob(GetParam().output_blob_precision, testEnv->outputDims);

    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), outputBlob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_NO_THROW(testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), actualBlob, &response));

    ASSERT_EQ(outputBlob, actualBlob);
}

TEST_P(BehaviorPluginTestInferRequestOutput, canInferWithSetInOut) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto input = prepareInputBlob(GetParam().input_blob_precision, testEnv->inputDims);
    testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), input, &response);
    auto output = _prepareOutputBlob(GetParam().output_blob_precision, testEnv->outputDims);
    testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), output, &response);

    sts = testEnv->inferRequest->Infer(&response);

    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequestOutput, canGetOutputBlob_deprecatedAPI) {
    TestEnv::Ptr testEnv;
    Blob::Ptr output;
    auto param = GetParam();

    StatusCode sts = StatusCode::OK;
    ResponseDesc response;

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(param, testEnv));
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), output, &response));

    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_TRUE(output) << "Plugin didn't allocate output blobs";
    ASSERT_FALSE(output->buffer() == nullptr) << "Plugin didn't allocate output blobs";
    auto dims = output->getTensorDesc().getDims();
    ASSERT_TRUE(testEnv->outputDims == dims) << "Output blob dimensions don't match network output";
    // [IE FPGA] The plugin ignores custom output precision: CVS-8122
    if (param.device != CommonTestUtils::DEVICE_FPGA && param.output_blob_precision != Precision::FP32) {
        ASSERT_EQ(param.output_blob_precision, output->getTensorDesc().getPrecision())
                                    << "Output blob precision don't match network output";
    } else if (param.device == CommonTestUtils::DEVICE_FPGA) {
        set<Precision> supportedOutputs = {Precision::FP16, Precision::FP32};
        ASSERT_TRUE(supportedOutputs.find(output->getTensorDesc().getPrecision()) != supportedOutputs.end()) << "Output blob precision don't match network output";
    } else {
        ASSERT_EQ(Precision::FP32, output->getTensorDesc().getPrecision()) << "Output blob precision don't match network output";
    }
}

TEST_P(BehaviorPluginTestInferRequestOutput, canGetOutputBlob) {
    TestEnv::Ptr testEnv;
    Blob::Ptr output;
    auto param = GetParam();

    StatusCode sts = StatusCode::OK;
    ResponseDesc response;

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(param, testEnv));
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), output, &response));

    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_TRUE(output) << "Plugin didn't allocate output blobs";
    ASSERT_FALSE(output->buffer() == nullptr) << "Plugin didn't allocate output blobs";

    auto tensorDescription = output->getTensorDesc();
    auto dims = tensorDescription.getDims();
    ASSERT_TRUE(testEnv->outputDims == dims) << "Output blob dimensions don't match network output";
    // [IE FPGA] The plugin ignores custom output precision: CVS-8122
    std::cout << "Device: " << param.device << std::endl;
    if (param.device != CommonTestUtils::DEVICE_FPGA && param.output_blob_precision != Precision::FP32) {
        ASSERT_EQ(param.output_blob_precision, tensorDescription.getPrecision())
                                    << "Output blob precision don't match network output";
    } else if (param.device == CommonTestUtils::DEVICE_FPGA) {
        set<Precision> supportedOutputs = {Precision::FP16, Precision::FP32};
        ASSERT_TRUE(supportedOutputs.find(tensorDescription.getPrecision()) != supportedOutputs.end()) << "Output blob precision don't match network output";
    } else {
        ASSERT_EQ(Precision::FP32, tensorDescription.getPrecision()) << "Output blob precision don't match network output";
    }
}

TEST_P(BehaviorPluginTestInferRequestOutput, getOutputAfterSetOutputDoNotChangeOutput) {
    TestEnv::Ptr testEnv;
    ResponseDesc response;

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr outputSetBlob = _prepareOutputBlob(GetParam().output_blob_precision, testEnv->outputDims);
    ASSERT_EQ(StatusCode::OK, testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), outputSetBlob, &response));
    Blob::Ptr outputGetBlob;
    ASSERT_EQ(StatusCode::OK, testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), outputGetBlob, &response));
    ASSERT_EQ(outputGetBlob.get(), outputSetBlob.get());
}

TEST_P(BehaviorPluginTestInferRequestOutput, canInferWithGetInOut) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    Blob::Ptr result;

    StatusCode sts = StatusCode::OK;
    ResponseDesc response;

    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), result, &response);
    sts = testEnv->inferRequest->Infer(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequestOutput, canStartAsyncInferWithGetInOut) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    Blob::Ptr result;

    StatusCode sts = StatusCode::OK;
    ResponseDesc response;

    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);

    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    sts = testEnv->inferRequest->Wait(500, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;

    testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), result, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}
