// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/except.hpp"
#include "openvino/core/visibility.hpp"
#include <memory>

#include "cache/multi_cache.h"
#include "cpu_memory.h"
#include "graph_context.h"
#include "onednn/iml_type_mapper.h"
#include "memory_arguments.hpp"

namespace ov {
namespace intel_cpu {

#if defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_ARM64)
#    define OV_CPU_INSTANCE_MLAS_ARM64(...) {__VA_ARGS__},
#else
#    define OV_CPU_INSTANCE_MLAS_ARM64(...)
#endif

#if defined(OV_CPU_WITH_ACL)
#   if defined(OPENVINO_ARCH_ARM)
#       define OV_CPU_INSTANCE_ACL32(...) {__VA_ARGS__},
#   else
#       define OV_CPU_INSTANCE_ACL32(...)
#   endif
#   if defined(OPENVINO_ARCH_ARM64)
#       define OV_CPU_INSTANCE_ACL64(...) {__VA_ARGS__},
#   else
#       define OV_CPU_INSTANCE_ACL64(...)
#   endif
#   if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
#       define OV_CPU_INSTANCE_ACL(...) {__VA_ARGS__},
#   else
#       define OV_CPU_INSTANCE_ACL(...)
#   endif
#else
#   define OV_CPU_INSTANCE_ACL32(...)
#   define OV_CPU_INSTANCE_ACL64(...)
#   define OV_CPU_INSTANCE_ACL(...)
#endif

#if defined(OV_CPU_WITH_DNNL)
#    define OV_CPU_INSTANCE_DNNL(...) {__VA_ARGS__},
#else
#    define OV_CPU_INSTANCE_DNNL(...)
#endif

#if defined(OPENVINO_ARCH_X86_64)
#    define OV_CPU_INSTANCE_X64(...) {__VA_ARGS__},
#else
#    define OV_CPU_INSTANCE_X64(...)
#endif

#if defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_X86_64)
#    define OV_CPU_INSTANCE_MLAS_X64(...) {__VA_ARGS__},
#else
#    define OV_CPU_INSTANCE_MLAS_X64(...)
#endif

#if defined(OV_CPU_WITH_SHL)
#    define OV_CPU_INSTANCE_SHL(...) {__VA_ARGS__},
#else
#    define OV_CPU_INSTANCE_SHL(...)
#endif

#define OV_CPU_INSTANCE_COMMON(...) {__VA_ARGS__},

// @todo another option is to determine shape relation by executor type
enum class ShapeTolerance {
    Agnostic,
    Dependant
};

enum class ExecutorType {
    Undefined,
    Graph,
    Common,
    jit_x64,
    Dnnl,
    Acl,
    Mlas,
    jit_aarch64,
    Shl
};

enum class OperationType {
    FullyConnected,
    MatMul,
    Convolution
};

std::string ExecutorTypeToString(const ExecutorType type);
ExecutorType ExecutorTypeFromString(const std::string& typeStr);

class ExecutorContext {
public:
    using Ptr = std::shared_ptr<ExecutorContext>;
    using CPtr = std::shared_ptr<const ExecutorContext>;

    ExecutorContext(const GraphContext::CPtr graphContext,
                    const std::vector<impl_desc_type>& implPriorities,
                    std::shared_ptr<std::unordered_map<std::string, MemoryPtr>> privateWeighCache = nullptr)
        : runtimeCache(graphContext->getParamsCache()),
          scratchPads(graphContext->getScratchPads()),
          weightsCache(graphContext->getWeightsCache()),
          engine(graphContext->getEngine()),
          implPriorities(implPriorities),
          privateWeighCache(std::move(privateWeighCache)),
          numNumaNodes(graphContext->getNumNumaNodes()) {
        auto cpuStreamsExecutor = graphContext->getCPUStreamExecutor();
        curNumaNodeId = std::max(0, cpuStreamsExecutor ? cpuStreamsExecutor->get_numa_node_id() : curNumaNodeId);
    }

    MultiCachePtr getRuntimeCache() const {
        auto runtimeCachePtr = runtimeCache.lock();
        assert(runtimeCachePtr);
        return runtimeCachePtr;
    }

    DnnlScratchPadPtr getScratchPad() const {
        return scratchPads[curNumaNodeId];
    }

    std::shared_ptr<std::unordered_map<std::string, MemoryPtr>> getPrivateWeighCache() const {
        return privateWeighCache;
    }

    const dnnl::engine& getEngine() const {
        return engine;
    }

    const std::vector<impl_desc_type>& getImplPriorities() const {
        return implPriorities;
    }

    const WeightsSharing::Ptr getWeightsCache() const {
        return weightsCache;
    }

private:
    // weak_ptr is required to avoid cycle dependencies with MultiCache
    // since ExecutorContext is stored in Executor itself
    MultiCacheWeakPtr runtimeCache;
    std::vector<DnnlScratchPadPtr> scratchPads;
    WeightsSharing::Ptr weightsCache;
    const dnnl::engine& engine;
    std::vector<impl_desc_type> implPriorities;
    // @todo remove after global cache is used exclusevly
    std::shared_ptr<std::unordered_map<std::string, MemoryPtr>> privateWeighCache;
    int numNumaNodes;
    int curNumaNodeId = -1;
};

class ExecutorFactoryLegacy {
public:
    ExecutorFactoryLegacy(const ExecutorContext::CPtr context) : context(context) {}
    virtual ~ExecutorFactoryLegacy() = default;

    const ExecutorContext::CPtr context;
};

using ExecutorFactoryLegacyPtr = std::shared_ptr<ExecutorFactoryLegacy>;
using ExecutorFactoryLegacyCPtr = std::shared_ptr<const ExecutorFactoryLegacy>;

class Executor {
public:
    // returns false if the stage has failed and the executor must be rejected
    virtual bool update(const MemoryArgs& memory) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'update' method is not implemented by executor");
        return false;
    }
    virtual void execute() const {}
    // dnnl_fullyconnected 3D workaround version
    virtual void execute(const MemoryArgs& memory) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'execute' method is not implemented by executor");
    }
    // legacy version
    virtual void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'execute' method is not implemented by executor");
    }
    virtual impl_desc_type implType() const = 0;
    virtual void moveMemToNumaNode(int numaID) {
        OPENVINO_THROW_NOT_IMPLEMENTED("This version of the 'moveMemToNumaNode' method is not implemented by executor");
    }
    virtual ~Executor() = default;
};
using ExecutorPtr = std::shared_ptr<Executor>;

}  // namespace intel_cpu
}  // namespace ov
