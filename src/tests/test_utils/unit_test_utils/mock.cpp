// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/relu.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_async_infer_request_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_executable_thread_safe_default.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/impl/mock_inference_plugin_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iexecutable_network_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinfer_request_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_iinference_plugin.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_ivariable_state_internal.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/mock_task_executor.hpp"
#include "unit_test_utils/mocks/mock_allocator.hpp"
#include "unit_test_utils/mocks/mock_icnn_network.hpp"
#include "unit_test_utils/mocks/mock_iexecutable_network.hpp"
#include "unit_test_utils/mocks/mock_iinfer_request.hpp"
#include "unit_test_utils/mocks/mock_not_empty_icnn_network.hpp"

using namespace InferenceEngine;

void MockNotEmptyICNNNetwork::getOutputsInfo(OutputsDataMap& out) const noexcept {
    IE_SUPPRESS_DEPRECATED_START
    auto data = std::make_shared<Data>(MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME, Precision::UNSPECIFIED);
    out[MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME] = data;
    IE_SUPPRESS_DEPRECATED_END
}

void MockNotEmptyICNNNetwork::getInputsInfo(InputsDataMap& inputs) const noexcept {
    IE_SUPPRESS_DEPRECATED_START
    auto inputInfo = std::make_shared<InputInfo>();

    auto inData = std::make_shared<Data>(MockNotEmptyICNNNetwork::INPUT_BLOB_NAME, Precision::UNSPECIFIED);
    inData->setDims(MockNotEmptyICNNNetwork::INPUT_DIMENSIONS);
    inData->setLayout(Layout::NCHW);
    inputInfo->setInputData(inData);

    auto outData = std::make_shared<Data>(MockNotEmptyICNNNetwork::OUTPUT_BLOB_NAME, Precision::UNSPECIFIED);
    outData->setDims(MockNotEmptyICNNNetwork::OUTPUT_DIMENSIONS);
    outData->setLayout(Layout::NCHW);

    inputs[MockNotEmptyICNNNetwork::INPUT_BLOB_NAME] = inputInfo;
    IE_SUPPRESS_DEPRECATED_END
}

std::shared_ptr<ngraph::Function> MockNotEmptyICNNNetwork::getFunction() noexcept {
    ngraph::ParameterVector parameters;
    parameters.push_back(std::make_shared<ngraph::op::v0::Parameter>(
        ov::element::f32,
        std::vector<ov::Dimension>{INPUT_DIMENSIONS.begin(), INPUT_DIMENSIONS.end()}));
    parameters.back()->set_friendly_name(INPUT_BLOB_NAME);
    auto relu = std::make_shared<ov::op::v0::Relu>(parameters.back());
    relu->set_friendly_name(OUTPUT_BLOB_NAME);
    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::op::v0::Result>(relu));
    return std::make_shared<ov::Model>(results, parameters, "empty_function");
}

std::shared_ptr<const ngraph::Function> MockNotEmptyICNNNetwork::getFunction() const noexcept {
    ngraph::ParameterVector parameters;
    parameters.push_back(std::make_shared<ngraph::op::v0::Parameter>(
        ov::element::f32,
        std::vector<ov::Dimension>{INPUT_DIMENSIONS.begin(), INPUT_DIMENSIONS.end()}));
    parameters.back()->set_friendly_name(INPUT_BLOB_NAME);
    auto relu = std::make_shared<ov::op::v0::Relu>(parameters.back());
    relu->set_friendly_name(OUTPUT_BLOB_NAME);
    ngraph::ResultVector results;
    results.push_back(std::make_shared<ngraph::op::v0::Result>(relu));
    return std::make_shared<const ov::Model>(results, parameters, "empty_function");
}
