// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

/**
 * @brief A graph that runs through level zero.
 * @details The generic graph interface does not know the driver API. What the pipelines need from a graph in terms of
 * command queues and events is here instead.
 */
class IZeroGraph : public IGraph {
public:
    virtual CommandQueueDesc get_command_queue_desc() const = 0;

    uint64_t get_command_queue_key() const override {
        return get_command_queue_desc().key();
    }

    virtual void set_last_submitted_event(const std::shared_ptr<Event>&, size_t) {
        OPENVINO_THROW("set_last_submitted_event not implemented");
    }
    virtual const std::shared_ptr<Event>& get_last_submitted_event(size_t) const {
        OPENVINO_THROW("get_last_submitted_event not implemented");
    }
};

/**
 * @brief Gets the level zero view of a graph.
 * @details Every graph that is given to the level zero pipelines is created by the level zero compiler adapters, so
 * this is expected to succeed. It fails with a message and not later on a missing method.
 */
inline std::shared_ptr<IZeroGraph> asZeroGraph(const std::shared_ptr<IGraph>& graph) {
    auto zeroGraph = std::dynamic_pointer_cast<IZeroGraph>(graph);
    OPENVINO_ASSERT(zeroGraph != nullptr, "The graph does not run through level zero");
    return zeroGraph;
}

}  // namespace intel_npu
