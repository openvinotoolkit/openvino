// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_infer_request.hpp"

#include "blob_factory.hpp"
#include "ie_ngraph_utils.hpp"
#include "mock_compiled_model.hpp"
#include "transformations/utils/utils.hpp"

void MockInferRequest::allocate_blobs() {
    for (const auto& it : _networkInputs) {
        m_inputs[it.first] = make_blob_with_precision(it.second->getTensorDesc());
        m_inputs[it.first]->allocate();
    }
    for (const auto& it : _networkOutputs) {
        m_outputs[it.first] = make_blob_with_precision(it.second->getTensorDesc());
        m_outputs[it.first]->allocate();
    }
    for (const auto& input : m_compiled_model->m_model->inputs()) {
        OPENVINO_ASSERT(m_inputs.find(ngraph::op::util::create_ie_output_name(input)) != m_inputs.end());
    }
    for (const auto& output : m_compiled_model->m_model->outputs()) {
        OPENVINO_ASSERT(m_outputs.find(ngraph::op::util::create_ie_output_name(output.get_node()->input_value(0))) !=
                        m_outputs.end());
    }
}
InferenceEngine::Blob::Ptr MockInferRequest::GetBlob(const std::string& name) {
    if (m_inputs.find(name) != m_inputs.end())
        return m_inputs[name];
    else if (m_outputs.find(name) != m_outputs.end())
        return m_outputs[name];
    else
        IE_THROW(NotImplemented) << "Cannot get blob for " << name;
}

void MockInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& userBlob) {
    if (m_inputs.find(name) != m_inputs.end())
        m_inputs[name] = userBlob;
    else if (m_outputs.find(name) != m_outputs.end())
        m_outputs[name] = userBlob;
    else
        IE_THROW(NotImplemented) << "Cannot set blob for " << name;
}
void MockInferRequest::InferImpl() {
    const auto& model = m_compiled_model->m_model;
    ov::TensorVector inputs;
    ov::TensorVector outputs;
    for (const auto& input : m_compiled_model->m_model->inputs()) {
        const auto& blob = m_inputs[ngraph::op::util::create_ie_output_name(input)];
        inputs.emplace_back(ov::Tensor(InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision()),
                                       blob->getTensorDesc().getDims(),
                                       blob->buffer()));
    }
    for (const auto& output : m_compiled_model->m_model->outputs()) {
        const auto& blob = m_outputs[ngraph::op::util::create_ie_output_name(output.get_node()->input_value(0))];
        outputs.emplace_back(
            ov::Tensor(InferenceEngine::details::convertPrecision(blob->getTensorDesc().getPrecision()),
                       blob->getTensorDesc().getDims(),
                       blob->buffer()));
    }
    model->evaluate(outputs, inputs);
}
