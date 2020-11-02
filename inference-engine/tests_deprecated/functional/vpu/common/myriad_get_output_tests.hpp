// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <myriad_layers_tests.hpp>
#include "myriad_xml_tests.hpp"

using namespace InferenceEngine;
using GetOutputTestsParams = std::tuple<std::tuple<std::string*, std::string*>, std::string>;

class myriadGetOutput_nightly :
        public myriadLayersTests_nightly,
        public testing::WithParamInterface<GetOutputTestsParams> {
public:
    std::string name_model_full;
    std::string name_model_crop;
    std::string name_output;
};

TEST_P(myriadGetOutput_nightly, AddOutput) {
    StatusCode st;

    name_model_full = (*(std::get<0>(std::get<0>(GetParam()))));
    name_model_crop = (*(std::get<1>(std::get<0>(GetParam()))));
    name_output = std::get<1>(GetParam());

    TBlob<uint8_t>::Ptr weights(GenWeights( ( 32786944 + 2000) / sizeof(ie_fp16),  0, 1));

    InferenceEngine::Core ie;
    auto crop_network = ie.ReadNetwork(name_model_crop, weights);

    InferenceEngine::InputsDataMap networkInputs;
    ASSERT_NO_THROW(networkInputs = crop_network.getInputsInfo());
    InferenceEngine::OutputsDataMap networkOutputs;
    ASSERT_NO_THROW(networkOutputs = crop_network.getOutputsInfo());

    networkInputs.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    networkOutputs.begin()->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::Blob::Ptr inputBlob;

    InferenceEngine::IExecutableNetwork::Ptr exeNetwork;
    std::map<std::string, std::string> networkConfig;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetwork, crop_network, networkConfig, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NE(exeNetwork, nullptr) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr inferRequest;
    ASSERT_NO_THROW(st = exeNetwork->CreateInferRequest(inferRequest, &_resp));

    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequest->GetBlob(networkInputs.begin()->first.c_str(), inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    GenRandomData(inputBlob);

    InferenceEngine::Blob::Ptr output_crop;
    ASSERT_NO_THROW(st = inferRequest->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NO_THROW(st = inferRequest->GetBlob(networkOutputs.begin()->first.c_str(), output_crop, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    /*Full Network Infer */

    auto full_network = ie.ReadNetwork(name_model_full, weights);

    full_network.addOutput(name_output, 0);

    InferenceEngine::InputsDataMap networkInputsFull;
    networkInputsFull = full_network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputsFull;
    networkOutputsFull = full_network.getOutputsInfo();

    networkInputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    networkOutputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    (++networkOutputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::IExecutableNetwork::Ptr exeNetworkFull;
    ASSERT_NO_THROW(st = _vpuPluginPtr->LoadNetwork(exeNetworkFull, full_network, networkConfig, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::IInferRequest::Ptr inferRequestFull;
    ASSERT_NO_THROW(st = exeNetworkFull->CreateInferRequest(inferRequestFull, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    ASSERT_NO_THROW(st = inferRequestFull->SetBlob("data", inputBlob, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    InferenceEngine::Blob::Ptr output_full;
    ASSERT_NO_THROW(st = inferRequestFull->Infer(&_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;
    ASSERT_NO_THROW(st = inferRequestFull->GetBlob(name_output.c_str(), output_full, &_resp));
    ASSERT_EQ(StatusCode::OK, st) << _resp.msg;

    CompareCommonAbsolute(output_full, output_crop, 0.0f);
}

std::string getTestCaseName(const testing::TestParamInfo<GetOutputTestsParams>& param) {
    return  "addOutput_" + std::get<1>(param.param);
}

class myriadCheckOutput_nightly :
        public myriadLayersTests_nightly {
};
