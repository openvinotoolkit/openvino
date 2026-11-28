// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>

#include "device_execution_context.hpp"

namespace ov {
namespace auto_plugin {

// the instance that answers the model level queries, with a copy of its model that stays valid after a release
struct ActiveDevice {
    ModelInstanceKey instance;
    SoCompiledModel compiled_model;
};

/**
 * @brief Owns the execution context of every model instance of a schedule, and publishes the instance that serves
 * the requests.
 *
 * The contexts are added while the schedule initializes and are never removed before it is destroyed, so their
 * address stays valid. The routing decision is published as an immutable snapshot: model level queries read it
 * without taking any lock of the scheduler (only the short publication lock of the registry), and only copy the
 * model handle of the published instance.
 */
class InstanceRegistry {
public:
    // computes the instance to publish, it runs under the publication lock so it must only read atomic state
    using Resolver = std::function<std::optional<ModelInstanceKey>()>;

    InstanceRegistry() = default;
    InstanceRegistry(const InstanceRegistry&) = delete;
    InstanceRegistry& operator=(const InstanceRegistry&) = delete;

    // idempotent; must not be called once the scheduler serves requests
    DeviceExecutionContext& add(const ModelInstanceKey& key);
    // asserts if the instance was not added
    DeviceExecutionContext& get(const ModelInstanceKey& key);
    DeviceExecutionContext* find(const ModelInstanceKey& key) const;
    void for_each(const std::function<void(DeviceExecutionContext&)>& visit) const;
    void clear();

    // concurrent publishers resolve and publish one after the other, so the last one sees the latest state
    void publish(const Resolver& resolve);
    void publish(const std::optional<ModelInstanceKey>& active);
    std::optional<ModelInstanceKey> active_instance() const;
    // empty if nothing is published or the published instance has no model
    ActiveDevice active_device() const;

private:
    struct Snapshot {
        ModelInstanceKey active;
    };
    std::shared_ptr<const Snapshot> load() const;

    std::map<ModelInstanceKey, std::unique_ptr<DeviceExecutionContext>> m_contexts;
    mutable std::mutex m_publish_mutex;
    std::shared_ptr<const Snapshot> m_snapshot;
};

}  // namespace auto_plugin
}  // namespace ov
