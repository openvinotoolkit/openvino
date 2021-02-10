// Copyright (C) 2017-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "generic_ie.hpp"

#include <ie_blob.h>

#include <algorithm>
#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "blob_factory.hpp"
#include <ie_ngraph_utils.hpp>
#include "ngraph/util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

constexpr ::ngraph::NodeTypeInfo ngraph::op::GenericIE::type_info;

void ngraph::op::GenericIE::addExtension(std::shared_ptr<const ngraph::Function> func,
                                         const InferenceEngine::IShapeInferExtensionPtr& ext) {
    NodeVector nodes;

    for (auto r : func->get_results())
        nodes.emplace_back(r);
    for (auto s : func->get_sinks())
        nodes.emplace_back(s);
    for (auto param : func->get_parameters())
        nodes.emplace_back(param);

    traverse_nodes(nodes, [&](std::shared_ptr<Node> op) {
        if (auto generic = std::dynamic_pointer_cast<GenericIE>(op)) {
            generic->addExtension(ext);
        }
        if (auto ti = std::dynamic_pointer_cast<ngraph::op::TensorIterator>(op)) {
            addExtension(ti->get_body(), ext);
        }
    });
}

void ngraph::op::GenericIE::addExtension(const InferenceEngine::IShapeInferExtensionPtr& ext) {
    extensions.emplace_back(ext);
}

std::vector<InferenceEngine::IShapeInferExtensionPtr> ngraph::op::GenericIE::getExtensions(std::shared_ptr<const ngraph::Function> func) {
    for (auto& op : func->get_ops()) {
        if (auto generic = std::dynamic_pointer_cast<GenericIE>(op)) {
            return generic->getExtensions();
        }
    }
    return {};
}

std::vector<InferenceEngine::IShapeInferExtensionPtr> ngraph::op::GenericIE::getExtensions() {
    return extensions;
}

ngraph::op::GenericIE::GenericIE(const ngraph::OutputVector& inputs,
                                 const std::map<std::string, InferenceEngine::Parameter>& params_,
                                 const std::string type_, const std::vector<PortIE>& outputs_)
    : Op(inputs), params(params_), outputs(outputs_), type(type_), initialized(0) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ngraph::op::GenericIE::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    auto genNode = std::make_shared<GenericIE>(new_args, params, type, outputs);
    genNode->extensions = extensions;
    genNode->reshape = reshape;
    return genNode;
}

void ngraph::op::GenericIE::validate_and_infer_types() {
    // This function returns precision based on existing precision and
    // precision that was set in outputs vector
    auto get_precision = [this](const size_t index) -> element::Type {
        if (index >= get_output_size() ||
            get_output_element_type(index) == element::dynamic ||
            get_output_element_type(index) == element::undefined) {
            return InferenceEngine::details::convertPrecision(outputs[index].precision);
        }
        return get_output_element_type(index);
    };

    // Extensions are not loaded when we create nGraph function
    // First call: create node
    if (initialized < 1) {
        if (outputs.size())
            set_output_size(outputs.size());
        for (size_t output_index = 0; output_index < outputs.size(); output_index++) {
            set_output_type(output_index, get_precision(output_index), Shape(outputs[output_index].dims));
        }
        initialized++;
    } else if (reshape) {
        THROW_IE_EXCEPTION << "IShapeInferExtension wasn't registered for node " << get_friendly_name()
                           << " with type " << type;
    }
}

bool ngraph::op::GenericIE::visit_attributes(ngraph::AttributeVisitor& visitor) {
    for (const auto& p : params) {
        std::string name = p.first;
        std::string value = p.second;
        visitor.on_attribute(name, value);
    }
    // This is a way to pass type name to transformations::Serialize() without
    // adding plugin_api dependency on transformation library
    std::string name = "__generic_ie_type__";
    std::string value = getType();
    visitor.on_attribute(name, value);
    return true;
}
