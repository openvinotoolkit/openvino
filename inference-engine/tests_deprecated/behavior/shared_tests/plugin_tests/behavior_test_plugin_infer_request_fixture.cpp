// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior_test_plugin.h"
#include "details/ie_cnn_network_tools.h"

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

Blob::Ptr BehaviorPluginTestInferRequest::prepareInputBlob(Precision blobPrecision, SizeVector inputDims) {
    auto input = makeNotAllocatedBlob(blobPrecision, TensorDesc::getLayoutByDims(inputDims), inputDims);
    input->allocate();
    return input;
}

Blob::Ptr BehaviorPluginTestInferRequest::_prepareOutputBlob(Precision blobPrecision, SizeVector outputDims) {
    auto output = makeNotAllocatedBlob(blobPrecision, TensorDesc::getLayoutByDims(outputDims), outputDims);
    output->allocate();
    return output;
}

void BehaviorPluginTestInferRequest::_setInputPrecision(
    const BehTestParams &param,
    CNNNetwork &cnnNetwork,
    TestEnv::Ptr &testEnv,
    const size_t expectedNetworkInputs) {

    InputsDataMap networkInputs = cnnNetwork.getInputsInfo();
    if (expectedNetworkInputs != 0) {
        ASSERT_EQ(networkInputs.size(), expectedNetworkInputs);
    }
    testEnv->networkInput = networkInputs.begin()->second;
    testEnv->networkInput->setPrecision(param.input_blob_precision);
    testEnv->inputDims = testEnv->networkInput->getTensorDesc().getDims();
    testEnv->inputName = networkInputs.begin()->first;
}

void BehaviorPluginTestInferRequest::_setOutputPrecision(
    const BehTestParams &param,
    CNNNetwork &cnnNetwork,
    TestEnv::Ptr &testEnv,
    const size_t expectedNetworkOutputs) {

    OutputsDataMap networkOutputs = cnnNetwork.getOutputsInfo();
    if (expectedNetworkOutputs != 0) {
        ASSERT_EQ(networkOutputs.size(), expectedNetworkOutputs);
    }
    testEnv->networkOutput = networkOutputs.begin()->second;
    testEnv->networkOutput->setPrecision(param.output_blob_precision);
    testEnv->outputDims = testEnv->networkOutput->getTensorDesc().getDims();
    testEnv->outputName = networkOutputs.begin()->first;
}

void BehaviorPluginTestInferRequest::_createAndCheckInferRequest(
    const BehTestParams &param,
    TestEnv::Ptr &testEnv,
    const std::map<std::string, std::string> &config,
    const size_t expectedNetworkInputs,
    const size_t expectedNetworkOutputs,
    InferenceEngine::IExtensionPtr extension) {

    testEnv = make_shared<TestEnv>();
    if (extension) {
        ASSERT_NO_THROW(testEnv->core.AddExtension(extension));
    }

    Core ie;
    testEnv->network = ie.ReadNetwork(param.model_xml_str, param.weights_blob);
    /* Call conversion from CNNNetwork NgraphImpl to CNNNetwork */
    testEnv->network.begin();

    _setInputPrecision(param, testEnv->network, testEnv, expectedNetworkInputs);
    _setOutputPrecision(param, testEnv->network, testEnv, expectedNetworkOutputs);

    std::map<std::string, std::string> full_config = config;
    full_config.insert(param.config.begin(), param.config.end());

#ifdef DUMP_EXECUTION_GRAPH
    full_config[PluginConfigParams::KEY_DUMP_EXEC_GRAPH_AS_DOT] = "behavior_tests_execution_graph_dump";
#endif

     ResponseDesc response;
//     ASSERT_NO_THROW(testEnv->exeNetwork = testEnv->core.LoadNetwork(testEnv->network, param.device, full_config));
     try {
         testEnv->exeNetwork = testEnv->core.LoadNetwork(testEnv->network, param.device, full_config);
     } catch (InferenceEngineException ex) {
         std::cout << "LoadNetwork failed. Status: " << ex.getStatus() << ", Response: " << ex.what();
         throw ex;
     } catch (std::exception ex) {
         std::cout << "LoadNetwork failed. Exception: " << typeid(ex).name() << ", what(): " << ex.what() << std::endl;
         throw;
     } catch (...) {
         std::cout << "LoadNetwork failed with unknown reason.";
         throw;
     }
     testEnv->actualInferRequest = testEnv->exeNetwork.CreateInferRequest();
     testEnv->inferRequest = static_cast<IInferRequest::Ptr &>(testEnv->actualInferRequest);
}

bool BehaviorPluginTestInferRequest::_wasDeviceBusy(ResponseDesc response) {
    std::cout << response.msg << "\n";
    std::string refError = REQUEST_BUSY_str;
    response.msg[refError.length()] = '\0';
    return !refError.compare(response.msg);
}
