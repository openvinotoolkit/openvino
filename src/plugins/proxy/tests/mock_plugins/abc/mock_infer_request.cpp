// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mock_infer_request.hpp"

#include "blob_factory.hpp"
#include "ie_ngraph_utils.hpp"
#include "mock_compiled_model.hpp"

static std::unordered_set<std::string> input_names = {"input"};
static std::unordered_set<std::string> output_names = {"sub"};

void MockInferRequest::allocate_blobs() {
    m_input = make_blob_with_precision(InferenceEngine::TensorDesc(
        InferenceEngine::details::convertPrecision(m_compiled_model->m_model->input().get_element_type()),
        m_compiled_model->m_model->input().get_shape(),
        InferenceEngine::TensorDesc::getLayoutByDims(m_compiled_model->m_model->input().get_shape())));
    m_input->allocate();
    m_output = make_blob_with_precision(InferenceEngine::TensorDesc(
        InferenceEngine::details::convertPrecision(m_compiled_model->m_model->output().get_element_type()),
        m_compiled_model->m_model->output().get_shape(),
        InferenceEngine::TensorDesc::getLayoutByDims(m_compiled_model->m_model->output().get_shape())));
    m_output->allocate();
}
InferenceEngine::Blob::Ptr MockInferRequest::GetBlob(const std::string& name) {
    if (input_names.find(name) != input_names.end())
        return m_input;
    else if (output_names.find(name) != output_names.end())
        return m_output;
    else
        IE_THROW(NotImplemented) << "Cannot get blob for " << name;
}

void MockInferRequest::SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& userBlob) {
    if (input_names.find(name) != input_names.end())
        m_input = userBlob;
    else if (output_names.find(name) != output_names.end())
        m_output = userBlob;
    else
        IE_THROW(NotImplemented) << "Cannot set blob for " << name;
}
void MockInferRequest::InferImpl() {
    const auto& model = m_compiled_model->m_model;
    ov::TensorVector inputs;
    ov::TensorVector outputs;
    inputs.emplace_back(ov::Tensor(InferenceEngine::details::convertPrecision(m_input->getTensorDesc().getPrecision()),
                                   m_input->getTensorDesc().getDims(),
                                   m_input->buffer()));
    outputs.emplace_back(
        ov::Tensor(InferenceEngine::details::convertPrecision(m_output->getTensorDesc().getPrecision()),
                   m_output->getTensorDesc().getDims(),
                   m_output->buffer()));
    model->evaluate(outputs, inputs);
}
