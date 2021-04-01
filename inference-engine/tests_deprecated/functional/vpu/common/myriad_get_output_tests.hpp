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

    InferenceEngine::ExecutableNetwork exeNetwork;
    std::map<std::string, std::string> networkConfig;
    ASSERT_NO_THROW(exeNetwork = _vpuPluginPtr->LoadNetwork(crop_network, networkConfig));

    InferenceEngine::InferRequest inferRequest;
    ASSERT_NO_THROW(inferRequest = exeNetwork.CreateInferRequest());

    ASSERT_NO_THROW(inputBlob = inferRequest.GetBlob(networkInputs.begin()->first.c_str()));
    GenRandomData(inputBlob);

    InferenceEngine::Blob::Ptr output_crop;
    ASSERT_NO_THROW(inferRequest.Infer());
    ASSERT_NO_THROW(output_crop = inferRequest.GetBlob(networkOutputs.begin()->first.c_str()));

    /*Full Network Infer */

    auto full_network = ie.ReadNetwork(name_model_full, weights);

    full_network.addOutput(name_output, 0);

    InferenceEngine::InputsDataMap networkInputsFull = full_network.getInputsInfo();
    InferenceEngine::OutputsDataMap networkOutputsFull = full_network.getOutputsInfo();

    networkInputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    networkOutputsFull.begin()->second->setPrecision(InferenceEngine::Precision::FP16);
    (++networkOutputsFull.begin())->second->setPrecision(InferenceEngine::Precision::FP16);

    InferenceEngine::ExecutableNetwork exeNetworkFull;
    ASSERT_NO_THROW(exeNetworkFull = _vpuPluginPtr->LoadNetwork(full_network, networkConfig));

    InferenceEngine::InferRequest inferRequestFull;
    ASSERT_NO_THROW(inferRequestFull = exeNetworkFull.CreateInferRequest());

    ASSERT_NO_THROW(inferRequestFull.SetBlob("data", inputBlob));

    InferenceEngine::Blob::Ptr output_full;
    ASSERT_NO_THROW(inferRequestFull.Infer());
    ASSERT_NO_THROW(output_full = inferRequestFull.GetBlob(name_output.c_str()));

    CompareCommonAbsolute(output_full, output_crop, 0.0f);
}

std::string getTestCaseName(const testing::TestParamInfo<GetOutputTestsParams>& param) {
    return  "addOutput_" + std::get<1>(param.param);
}

class myriadCheckOutput_nightly :
        public myriadLayersTests_nightly {
};
