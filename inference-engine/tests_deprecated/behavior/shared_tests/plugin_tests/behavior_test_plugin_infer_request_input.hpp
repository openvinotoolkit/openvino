// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + "_" + obj.param.input_blob_precision.name() + "_" + getModelName(obj.param.model_xml_str)
                + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
}
}

TEST_P(BehaviorPluginTestInferRequestInput, canSetInputBlobForSyncRequest) {
    TestEnv::Ptr testEnv;
    Blob::Ptr actualBlob;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto inputBlob = prepareInputBlob(GetParam().input_blob_precision, testEnv->inputDims);

    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), inputBlob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_NO_THROW(testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), actualBlob, &response));

    ASSERT_EQ(inputBlob, actualBlob);
}

TEST_P(BehaviorPluginTestInferRequestInput, canSetInputBlobForAsyncRequest) {
    TestEnv::Ptr testEnv;
    Blob::Ptr actualBlob;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto inputBlob = prepareInputBlob(GetParam().input_blob_precision, testEnv->inputDims);

    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), inputBlob, &response));
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_NO_THROW(testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), actualBlob, &response));

    ASSERT_EQ(inputBlob, actualBlob);
}

TEST_P(BehaviorPluginTestInferRequestInput, canInferWithSetInOut) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    auto input = prepareInputBlob(GetParam().input_blob_precision, testEnv->inputDims);
    testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), input, &response);
    auto output = _prepareOutputBlob(GetParam().output_blob_precision, testEnv->outputDims);
    testEnv->inferRequest->SetBlob(testEnv->outputName.c_str(), output, &response);
    sts = testEnv->inferRequest->Infer(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequestInput, canGetInputBlob_deprecatedAPI) {
    TestEnv::Ptr testEnv;
    Blob::Ptr input;
    auto param = GetParam();

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(param, testEnv));
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response));

    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_TRUE(input) << "Plugin didn't allocate input blobs";
    ASSERT_FALSE(input->buffer() == nullptr) << "Plugin didn't allocate input blobs";
    auto dims = input->getTensorDesc().getDims();
    ASSERT_TRUE(testEnv->inputDims == dims) << "Input blob dimensions don't match network input";

    ASSERT_EQ(param.input_blob_precision, input->getTensorDesc().getPrecision()) << "Input blob precision don't match network input";
}

TEST_P(BehaviorPluginTestInferRequestInput, canGetInputBlob) {
    TestEnv::Ptr testEnv;
    Blob::Ptr input;
    auto param = GetParam();

    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(param, testEnv));
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response));

    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    ASSERT_TRUE(input) << "Plugin didn't allocate input blobs";
    ASSERT_FALSE(input->buffer() == nullptr) << "Plugin didn't allocate input blobs";

    auto tensorDescription = input->getTensorDesc();
    auto dims = tensorDescription.getDims();
    ASSERT_TRUE(testEnv->inputDims == dims) << "Input blob dimensions don't match network input";

    ASSERT_EQ(param.input_blob_precision, tensorDescription.getPrecision()) << "Input blob precision don't match network input";
}

TEST_P(BehaviorPluginTestInferRequestInput, getInputAfterSetInputDoNotChangeInput) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr inputSetBlob = prepareInputBlob(GetParam().input_blob_precision, testEnv->inputDims);
    ASSERT_NO_THROW(sts = testEnv->inferRequest->SetBlob(testEnv->inputName.c_str(), inputSetBlob, &response));
    Blob::Ptr inputGetBlob;
    ASSERT_NO_THROW(sts = testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), inputGetBlob, &response));
    ASSERT_EQ(inputGetBlob.get(), inputSetBlob.get());
}

TEST_P(BehaviorPluginTestInferRequestInput, canInferWithGetInOut) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    Blob::Ptr result;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), result, &response);
    sts = testEnv->inferRequest->Infer(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}

TEST_P(BehaviorPluginTestInferRequestInput, canStartAsyncInferWithGetInOut) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    Blob::Ptr input;
    Blob::Ptr result;
    testEnv->inferRequest->GetBlob(testEnv->inputName.c_str(), input, &response);
    sts = testEnv->inferRequest->StartAsync(&response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    sts = testEnv->inferRequest->Wait(500, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
    testEnv->inferRequest->GetBlob(testEnv->outputName.c_str(), result, &response);
    ASSERT_EQ(StatusCode::OK, sts) << response.msg;
}
