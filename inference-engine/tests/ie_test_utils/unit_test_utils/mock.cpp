// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unit_test_utils/mocks/mock_allocator.hpp"
#include "unit_test_utils/mocks/mock_icnn_network.hpp"
#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/mock_task_executor.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_async_infer_request_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"

#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"

#include <legacy/ie_layers.h>

using namespace InferenceEngine;

void MockNotEmptyICNNNetwork::getOutputsInfo(OutputsDataMap& out) const noexcept {
    IE_SUPPRESS_DEPRECATED_START
    auto data = std::make_shared<Data>(MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME, Precision::UNSPECIFIED);
    getInputTo(data)[""] = std::make_shared<CNNLayer>(LayerParams{
        MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME,
        "FullyConnected",
        Precision::FP32 });
    out[MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME] = data;
    IE_SUPPRESS_DEPRECATED_END
}

void MockNotEmptyICNNNetwork::getInputsInfo(InputsDataMap &inputs) const noexcept {
    IE_SUPPRESS_DEPRECATED_START
    auto inputInfo = std::make_shared<InputInfo>();

    auto inData = std::make_shared<Data>(MockNotEmptyICNNNetwork::INPUT_BLOB_NAME, Precision::UNSPECIFIED);
    auto inputLayer = std::make_shared<CNNLayer>(LayerParams{
        MockNotEmptyICNNNetwork::INPUT_BLOB_NAME,
        "Input",
        Precision::FP32 });
    getInputTo(inData)[MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME] = inputLayer;
    inData->setDims(MockNotEmptyICNNNetwork::INPUT_DIMENTIONS);
    inData->setLayout(Layout::NCHW);
    inputInfo->setInputData(inData);

    auto outData = std::make_shared<Data>(MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME, Precision::UNSPECIFIED);
    outData->setDims(MockNotEmptyICNNNetwork::OUTPUT_DIMENTIONS);
    outData->setLayout(Layout::NCHW);
    getInputTo(outData)[""] = std::make_shared<CNNLayer>(LayerParams{
        MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME,
        "FullyConnected",
        Precision::FP32 });

    inputLayer->outData.push_back(outData);

    inputs[MockNotEmptyICNNNetwork::INPUT_BLOB_NAME] = inputInfo;
    IE_SUPPRESS_DEPRECATED_END
}
