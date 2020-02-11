// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tensor_iterator.hpp"

#include <ie_blob.h>

#include <details/ie_exception.hpp>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::FakeTensorIterator::type_info;

op::FakeTensorIterator::FakeTensorIterator(const ngraph::NodeVector& inputs,
                                           const pugi::xml_document& xml,
                                           const std::shared_ptr<InferenceEngine::details::LayerParseParameters> params,
                                           const std::vector<ngraph::PartialShape>& output_shapes,
                                           const InferenceEngine::Blob::CPtr weights)
    : Op(inputs), m_params(params), m_output_shapes(output_shapes) {
    m_doc.reset(xml);
    m_weights = weights;
    constructor_validate_and_infer_types();
}

op::FakeTensorIterator::FakeTensorIterator(const pugi::xml_node& xml,
                                           const std::shared_ptr<InferenceEngine::details::LayerParseParameters> params,
                                           const ngraph::OutputVector& inputs,
                                           const std::vector<ngraph::PartialShape>& output_shapes,
                                           const InferenceEngine::Blob::CPtr weights)
    : Op(inputs), m_params(params), m_output_shapes(output_shapes) {
    m_doc.append_copy(xml);
    m_weights = weights;
    constructor_validate_and_infer_types();
}

void op::FakeTensorIterator::validate_and_infer_types() {
    size_t i = 0;
    for (auto& shape : m_output_shapes) {
        set_output_type(i++, get_input_element_type(0), shape);
    }
}

shared_ptr<Node> op::FakeTensorIterator::copy_with_new_args(const NodeVector& new_args) const {
    return std::make_shared<op::FakeTensorIterator>(new_args, m_doc, m_params, m_output_shapes, m_weights);
}
