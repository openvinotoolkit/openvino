// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

#include "openvino/core/except.hpp"

namespace ov {
namespace tp_gpu {
class TPDeviceCoordinator;
}  // namespace tp_gpu

namespace intel_gpu {

/// \brief Maps a collective group to the coordinator that runs it.
///
/// One registry per worker graph-set; every rank of that set shares the same
/// registry. The tensor-parallel plugin fills it during `compile_model` and
/// hands it over as a property, so the graph itself carries no runtime state --
/// only the group id each collective op belongs to.
///
/// The coordinator stays an incomplete type here: this header is compiled into
/// every GPU target, including builds without the tensor-parallel plugin, and
/// must not depend on it.
///
/// Deliberately not synchronized: the registry is populated before any infer
/// request exists and is read-only afterwards.
class CollectiveCommRegistry {
public:
    void set_group(uint32_t group_id, std::shared_ptr<ov::tp_gpu::TPDeviceCoordinator> coordinator) {
        m_groups[group_id] = std::move(coordinator);
    }

    const std::shared_ptr<ov::tp_gpu::TPDeviceCoordinator>& get_group(uint32_t group_id) const {
        auto it = m_groups.find(group_id);
        OPENVINO_ASSERT(it != m_groups.end(), "No collective group registered with id ", group_id);
        return it->second;
    }

private:
    std::unordered_map<uint32_t, std::shared_ptr<ov::tp_gpu::TPDeviceCoordinator>> m_groups;
};

using CollectiveCommRegistryPtr = std::shared_ptr<CollectiveCommRegistry>;

}  // namespace intel_gpu
}  // namespace ov
