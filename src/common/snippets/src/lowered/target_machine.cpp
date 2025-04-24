// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/target_machine.hpp"

#include "snippets/runtime_configurator.hpp"

using namespace ov::snippets;
std::function<std::shared_ptr<Emitter>(const lowered::ExpressionPtr&)> TargetMachine::get(const ov::DiscreteTypeInfo& type) const {
    auto jitter = jitters.find(type);
    OPENVINO_ASSERT(jitter != jitters.end(), "Target code emitter is not available for ", type.name, " operation.");
    return jitter->second.first;
}

std::function<std::set<ov::element::TypeVector>(const std::shared_ptr<ov::Node>&)>
TargetMachine::get_supported_precisions(const ov::DiscreteTypeInfo& type) const {
    auto jitter = jitters.find(type);
    OPENVINO_ASSERT(jitter != jitters.end(), "Supported precisions set is not available for ", type.name, " operation.");
    return jitter->second.second;
}

bool TargetMachine::has(const ov::DiscreteTypeInfo& type) const {
    return jitters.find(type) != jitters.end();
}

const std::shared_ptr<RuntimeConfigurator>& TargetMachine::get_runtime_configurator() const {
    OPENVINO_ASSERT(configurator, "RuntimeConfigurator has not been inited!");
    return configurator;
}
