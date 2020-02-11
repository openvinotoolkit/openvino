// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <pugixml.hpp>

#include <ie_format_parser.h>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class FakeTensorIterator : public Op {
public:
    static constexpr NodeTypeInfo type_info{"FakeTensorIterator", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    FakeTensorIterator(const ngraph::NodeVector & inputs,
                       const pugi::xml_document & xml,
                       const std::shared_ptr<InferenceEngine::details::LayerParseParameters> params,
                       const std::vector<ngraph::PartialShape> & output_shapes,
                       const InferenceEngine::Blob::CPtr weights);

    FakeTensorIterator(const pugi::xml_node & xml,
                       const std::shared_ptr<InferenceEngine::details::LayerParseParameters> params,
                       const ngraph::OutputVector & inputs,
                       const std::vector<ngraph::PartialShape> & output_shapes,
                       const InferenceEngine::Blob::CPtr weights);

    std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

    pugi::xml_node get_xml() { return m_doc.document_element(); }
    std::shared_ptr<InferenceEngine::details::LayerParseParameters> get_params() { return m_params; }
    InferenceEngine::Blob::CPtr get_weights() { return m_weights; }
    void validate_and_infer_types() override;

protected:
    pugi::xml_document m_doc;
    std::shared_ptr<InferenceEngine::details::LayerParseParameters> m_params;
    std::vector<ngraph::PartialShape> m_output_shapes;
    InferenceEngine::Blob::CPtr m_weights;
};

}  // namespace op
}  // namespace ngraph
