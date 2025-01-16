// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <vector>

#include "graph.h"
#include "memory_desc/cpu_memory_desc.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "post_ops.hpp"

namespace ov {
namespace intel_cpu {

template <typename Attrs>
class GraphEmitter {
public:
    // @todo use template argument instead of passing std::function
    using ensureAttrsStrategy =
        std::function<void(GraphPtr, NodePtr, Attrs, Attrs, ExecutorContext::CPtr, const std::string&)>;

    GraphEmitter(const MemoryDescArgs& descs,
                 const Attrs& attrs,
                 const PostOps& postOps,
                 const MemoryArgs& memory,
                 const ExecutorContext::CPtr context,
                 const std::string& name,
                 ensureAttrsStrategy ensureAttrs = {})
        : descs(descs),
          attrs(attrs),
          postOps(postOps),
          context(context),
          name(name),
          ensureAttrs(std::move(ensureAttrs)) {
        OPENVINO_THROW("Graph emitter is not implemented yet!");
    }

    GraphEmitter& createGraph(const MemoryDescArgs& descs,
                              const Attrs& attrs,
                              const PostOps& postOps,
                              const ExecutorContext::CPtr context) {
        OPENVINO_THROW("Not implemented yet!");
        return *this;
    }

    GraphEmitter& ensureSrcDescsMatch() {
        OPENVINO_THROW("Not implemented yet!");
        return *this;
    }

    GraphEmitter& ensureDstDescsMatch() {
        OPENVINO_THROW("Not implemented yet!");
        return *this;
    }

    GraphEmitter& ensureAttrsMatch() {
        OPENVINO_THROW("Not implemented yet!");
        return *this;
    }

    GraphEmitter& ensurePostOpsMatch() {
        OPENVINO_THROW("Not implemented yet!");
        return *this;
    }

    GraphPtr emit() {
        OPENVINO_THROW("Not implemented yet!");
        return graph;
    }

private:
    const MemoryDescArgs& descs;
    const Attrs& attrs;
    const PostOps& postOps;
    const ExecutorContext::CPtr context;
    const std::string& name;
    const ensureAttrsStrategy ensureAttrs;
    NodePtr coreNode;
    GraphPtr graph;
};

}  // namespace intel_cpu
}  // namespace ov
