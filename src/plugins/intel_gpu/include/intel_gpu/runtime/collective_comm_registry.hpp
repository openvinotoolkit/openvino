// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>

#include "openvino/core/except.hpp"

namespace ov {
namespace tp_gpu {
class TPDeviceCoordinator;
}  // namespace tp_gpu

namespace intel_gpu {

/// \brief Holds the coordinator that runs this graph's collectives.
///
/// One registry per worker graph-set; every rank of that set shares the same
/// registry. The tensor-parallel plugin fills it during `compile_model` and
/// hands it over as a property, so the graph itself carries no runtime state.
///
/// The coordinator stays an incomplete type here: this header is compiled into
/// every GPU target, including builds without the tensor-parallel plugin, and
/// must not depend on it.
///
/// Deliberately not synchronized: the registry is populated before any infer
/// request exists and is read-only afterwards.
class CollectiveCommRegistry {
public:
    void set_coordinator(std::shared_ptr<ov::tp_gpu::TPDeviceCoordinator> coordinator) {
        m_coordinator = std::move(coordinator);
    }

    const std::shared_ptr<ov::tp_gpu::TPDeviceCoordinator>& coordinator() const {
        OPENVINO_ASSERT(m_coordinator != nullptr, "No collective coordinator registered");
        return m_coordinator;
    }

private:
    std::shared_ptr<ov::tp_gpu::TPDeviceCoordinator> m_coordinator;
};

using CollectiveCommRegistryPtr = std::shared_ptr<CollectiveCommRegistry>;

}  // namespace intel_gpu
}  // namespace ov
