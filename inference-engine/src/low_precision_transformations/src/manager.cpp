// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/manager.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

Manager::Manager() {
}

Manager::Manager(
    std::shared_ptr<PassConfig> pass_config,
    TransformationContext* context,
    const ILayerTransformationsManager* manager) :
    ngraph::pass::Manager(pass_config),
    context(context),
    manager(manager) {
}

std::vector<element::Type> Manager::getPrecisionsOnActivations(const Node& op) const noexcept {
    if (m_pass_list.empty()) {
        //
    }
    return { element::u8 };

    //const std::string operantionType = LowPrecisionTransformations::getType(op);
    //const std::vector<LayerTransformationPtr> transformation = transformations.find(operantionType);
    //if (transformation.empty()) {
    //    return std::vector<element::Type>();
    //}
    //std::vector<element::Type> precisions = transformation[0]->getPrecisionsOnActivations();

    //for (const auto& transform : transformation) {
    //    precisions = precisionIntersection(precisions, transform->getPrecisionsOnActivations());
    //}
    //return precisions;
}

bool Manager::isQuantized(const std::shared_ptr<Node>& target) const noexcept {
    if (manager != nullptr) {
        return manager->isQuantized(target);
    }

    const std::vector<LayerTransformationPtr> transformations = find(target->get_type_name());
    if (transformations.empty()) {
        return false;
    }

    for (const auto& transform : transformations) {
        if (!transform->isQuantized(target)) {
            return false;
        }
    }

    return true;
}

bool Manager::isPrecisionPreserved(const std::shared_ptr<Node>& target) const noexcept {
    if (manager != nullptr) {
        return manager->isPrecisionPreserved(target);
    }

    const std::vector<LayerTransformationPtr> transformations = find(target->get_type_name());
    if (transformations.empty()) {
        return false;
    }

    for (const auto& transform : transformations) {
        if (!transform->isPrecisionPreserved(target)) {
            return false;
        }
    }

    return true;
}

std::vector<LayerTransformationPtr> Manager::find(const std::string& name) const noexcept {
    const auto it = transformations.find(name);
    if (it == transformations.end()) {
        return {};
    }
    return { it->second };

    //for (auto pass : m_pass_list) {
    //    auto transformation = std::dynamic_pointer_cast<ngraph::pass::low_precision::LayerTransformation>(pass);
    //}
    //return {};
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
