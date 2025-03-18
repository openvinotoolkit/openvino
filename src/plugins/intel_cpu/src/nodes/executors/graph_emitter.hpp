// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <utility>

#include "graph.h"
#include "node.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

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
                 ExecutorContext::CPtr context,
                 const std::string& name,
                 ensureAttrsStrategy ensureAttrs = {})
        : descs(descs),
          attrs(attrs),
          postOps(postOps),
          context(std::move(context)),
          name(name),
          ensureAttrs(std::move(ensureAttrs)) {
        OPENVINO_THROW("Graph emitter is not implemented yet!");
    }

    GraphEmitter& createGraph(const MemoryDescArgs& descs,
                              const Attrs& attrs,
                              const PostOps& postOps,
                              const ExecutorContext::CPtr& context) {
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

    static MemoryDescArgs memoryDescsFromMemory(const MemoryArgs& memory) {
        MemoryDescArgs memoryDescs;
        memoryDescs.reserve(memory.size());

        for (const auto& mem : memory) {
            memoryDescs[mem.first] = mem.second->getDescPtr();
        }

        return memoryDescs;
    }

    static executor::Config<Attrs> createConfig(const MemoryArgs& memory, const Attrs& attrs, const PostOps& postOps) {
        return executor::Config<Attrs>{memoryDescsFromMemory(memory), attrs, postOps};
    }

    static ExecutorPtr fallback(const executor::Config<Attrs>& config,
                                const executor::Config<Attrs>& fallbackConfig,
                                const MemoryArgs& memory,
                                const ExecutorContext::CPtr context,
                                const std::string& name) {
        DEBUG_LOG("Falling back to graph executor for ",
                  name,
                  ". Original config: ",
                  config,
                  " new config:",
                  fallbackConfig);

        GraphEmitter<Attrs> graphEmitter(config.descs, config.attrs, config.postOps, memory, context, name);

        const auto& graphExecutor =
            graphEmitter.createGraph(fallbackConfig.descs, fallbackConfig.attrs, fallbackConfig.postOps, context)
                .ensureAttrsMatch()
                .ensureSrcDescsMatch()
                .ensureDstDescsMatch()
                .ensurePostOpsMatch()
                .emit();
        (void)graphExecutor;

        OPENVINO_THROW("Fallback logic is not implemented yet");  // return graphExecutor;
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

}  // namespace ov::intel_cpu
