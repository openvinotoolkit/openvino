// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/target_machine.hpp"

using namespace ov::snippets;
std::function<std::shared_ptr<Emitter>(const lowered::ExpressionPtr&)> TargetMachine::get(const ov::DiscreteTypeInfo& type) const {
    auto jitter = jitters.find(type);
    if (jitter == jitters.end()) {
        OPENVINO_THROW(std::string("Target code emitter is not available for ") + type.name + " operation.");
    }
    return jitter->second.first;
}

std::function<std::set<ov::element::TypeVector>(const std::shared_ptr<ov::Node>&)>
TargetMachine::get_supported_precisions(const ov::DiscreteTypeInfo& type) const {
    auto jitter = jitters.find(type);
    if (jitter == jitters.end()) {
        OPENVINO_THROW(std::string("Target code emitter is not available for ") + type.name + " operation.");
    }
    return jitter->second.second;
}

bool TargetMachine::has(const ov::DiscreteTypeInfo& type) const {
    return jitters.find(type) != jitters.end();
}
