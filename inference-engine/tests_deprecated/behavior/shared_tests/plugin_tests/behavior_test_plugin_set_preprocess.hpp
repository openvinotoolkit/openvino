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
        return obj.param.device + "_" + obj.param.input_blob_precision.name()
               + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
    }
}

TEST_P(BehaviorPluginTestPreProcess, SetPreProcessToInputInfo) {
    InferenceEngine::Core core;

    CNNNetwork cnnNetwork = core.ReadNetwork(GetParam().model_xml_str, GetParam().weights_blob);

    auto &preProcess = cnnNetwork.getInputsInfo().begin()->second->getPreProcess();
    preProcess.setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);

    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    ASSERT_NO_THROW(exeNetwork = core.LoadNetwork(cnnNetwork, GetParam().device, GetParam().config));

    IInferRequest::Ptr inferRequest;
    ASSERT_EQ(StatusCode::OK, exeNetwork->CreateInferRequest(inferRequest, &response));

    {
        ConstInputsDataMap inputsMap;
        ASSERT_EQ(StatusCode::OK, exeNetwork->GetInputsInfo(inputsMap, &response));
        const auto& name = inputsMap.begin()->second->name();
        const PreProcessInfo *info;
        inferRequest->GetPreProcess(name.c_str(), &info, &response);

        ASSERT_EQ(info->getResizeAlgorithm(), ResizeAlgorithm::RESIZE_BILINEAR);
        ASSERT_PREPROCESS_INFO_EQ(preProcess, *info);
    }
}

TEST_P(BehaviorPluginTestPreProcess, SetPreProcessToInferRequest) {
    TestEnv::Ptr testEnv;
    ASSERT_NO_FATAL_FAILURE(_createAndCheckInferRequest(GetParam(), testEnv));
    ResponseDesc response;

    auto& request = testEnv->inferRequest;
    PreProcessInfo preProcessInfo;
    preProcessInfo.setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);

    IInferRequest::Ptr untouched_request = testEnv->exeNetwork.CreateInferRequest();

    ConstInputsDataMap inputs = testEnv->exeNetwork.GetInputsInfo();
    auto input_name = inputs.begin()->second->name();
    auto inputBlob = prepareInputBlob(GetParam().input_blob_precision, testEnv->inputDims);

    ASSERT_EQ(StatusCode::OK, request->SetBlob(input_name.c_str(), inputBlob, preProcessInfo, &response));

    {
        const PreProcessInfo *info = nullptr;
        ASSERT_EQ(StatusCode::OK, request->GetPreProcess(input_name.c_str(), &info, &response));
        ASSERT_EQ(info->getResizeAlgorithm(), ResizeAlgorithm::RESIZE_BILINEAR);
        ASSERT_PREPROCESS_INFO_EQ(preProcessInfo, *info);
    }

    {
        const PreProcessInfo *info = nullptr;
        ASSERT_EQ(StatusCode::OK, untouched_request->GetPreProcess(input_name.c_str(), &info, &response));
        ASSERT_EQ(testEnv->network.getInputsInfo()[input_name]->getPreProcess().getResizeAlgorithm(),info->getResizeAlgorithm());
    }
}
