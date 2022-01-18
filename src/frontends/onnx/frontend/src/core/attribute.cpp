// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/attribute.hpp"

#include "core/graph.hpp"
#include "core/model.hpp"
#include "ngraph/log.hpp"

namespace ngraph {
namespace onnx_import {
Subgraph Attribute::get_subgraph(const Graph* parent_graph) const {
    if (m_attribute_proto->type() != ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH) {
        throw error::attribute::InvalidData{m_attribute_proto->type()};
    }

    auto model_proto = std::make_shared<ONNX_NAMESPACE::ModelProto>();

    const auto& graph = m_attribute_proto->g();
    model_proto->mutable_graph()->CopyFrom(graph);

    // set opset version and domain from the parent graph
    model_proto->mutable_opset_import()->CopyFrom(parent_graph->get_opset_imports());
    return Subgraph{model_proto, parent_graph};
}

}  // namespace onnx_import

}  // namespace ngraph
