// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "instance_registry.hpp"

namespace ov {
namespace auto_plugin {

DeviceExecutionContext& InstanceRegistry::add(const ModelInstanceKey& key) {
    auto& context = m_contexts[key];
    if (!context) {
        context = std::make_unique<DeviceExecutionContext>(key);
    }
    return *context;
}

DeviceExecutionContext& InstanceRegistry::get(const ModelInstanceKey& key) {
    auto* context = find(key);
    OPENVINO_ASSERT(context, "Device ", key.to_string(), " has no execution context");
    return *context;
}

DeviceExecutionContext* InstanceRegistry::find(const ModelInstanceKey& key) const {
    const auto iter = m_contexts.find(key);
    return iter == m_contexts.end() ? nullptr : iter->second.get();
}

void InstanceRegistry::for_each(const std::function<void(DeviceExecutionContext&)>& visit) const {
    for (const auto& context : m_contexts) {
        visit(*context.second);
    }
}

void InstanceRegistry::clear() {
    publish(std::nullopt);
    m_contexts.clear();
}

void InstanceRegistry::publish(const Resolver& resolve) {
    std::lock_guard<std::mutex> lock(m_publish_mutex);
    const auto active = resolve();
    m_snapshot = active ? std::make_shared<const Snapshot>(Snapshot{*active}) : nullptr;
}

void InstanceRegistry::publish(const std::optional<ModelInstanceKey>& active) {
    publish([&active] {
        return active;
    });
}

std::shared_ptr<const InstanceRegistry::Snapshot> InstanceRegistry::load() const {
    std::lock_guard<std::mutex> lock(m_publish_mutex);
    return m_snapshot;
}

std::optional<ModelInstanceKey> InstanceRegistry::active_instance() const {
    const auto snapshot = load();
    return snapshot ? std::optional<ModelInstanceKey>(snapshot->active) : std::nullopt;
}

ActiveDevice InstanceRegistry::active_device() const {
    for (auto snapshot = load(); snapshot;) {
        const auto* context = find(snapshot->active);
        auto model = context ? context->compiled_model() : SoCompiledModel{};
        // the instance is published before the previous one is released, so an empty model means it moved on
        auto latest = load();
        if (model || latest == snapshot) {
            return model ? ActiveDevice{snapshot->active, std::move(model)} : ActiveDevice{};
        }
        snapshot = std::move(latest);
    }
    return {};
}

}  // namespace auto_plugin
}  // namespace ov
