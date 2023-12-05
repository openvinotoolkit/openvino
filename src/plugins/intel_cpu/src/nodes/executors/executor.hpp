// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cache/multi_cache.h"
#include "graph_context.h"
#include "onednn/iml_type_mapper.h"

namespace ov {
namespace intel_cpu {

#if defined(OV_CPU_WITH_MLAS) && defined(OPENVINO_ARCH_ARM64)
#define OV_CPU_INSTANCE_MLAS_ARM64(...) \
    {__VA_ARGS__},
#else
#define OV_CPU_INSTANCE_MLAS_ARM64(...)
#endif

#if defined(OV_CPU_WITH_ACL)
#define OV_CPU_INSTANCE_ACL(...) \
    {__VA_ARGS__},
#else
#define OV_CPU_INSTANCE_ACL(...)
#endif

#if defined(OV_CPU_WITH_DNNL)
#define OV_CPU_INSTANCE_DNNL(...) \
    {__VA_ARGS__},
#else
#define OV_CPU_INSTANCE_DNNL(...)
#endif

#if defined(OPENVINO_ARCH_X86_64)
#define OV_CPU_INSTANCE_X64(...) \
    {__VA_ARGS__},
#else
#define OV_CPU_INSTANCE_X64(...)
#endif

#define OV_CPU_INSTANCE_COMMON(...) \
    {__VA_ARGS__},

enum class ExecutorType {
    Undefined,
    Common,
    x64,
    Dnnl,
    Acl,
    Mlas
};

class ExecutorContext {
public:
    typedef std::shared_ptr<ExecutorContext> Ptr;
    typedef std::shared_ptr<const ExecutorContext> CPtr;

    ExecutorContext(const GraphContext::CPtr graphContext, const std::vector<impl_desc_type>& implPriorities)
        : runtimeCache(graphContext->getParamsCache()),
          scratchPad(graphContext->getScratchPad()),
          engine(graphContext->getEngine()),
          implPriorities(implPriorities) {}

    MultiCacheWeakPtr getRuntimeCache() const {
        return runtimeCache;
    }

    DnnlScratchPadPtr getScratchPad() const {
        return scratchPad;
    }

    dnnl::engine getEngine() const {
        return engine;
    }

    const std::vector<impl_desc_type>& getImplPriorities() const {
        return implPriorities;
    }

private:
    // weak_ptr is required to avoid cycle dependencies with MultiCache
    // since ExecutorContext is stored in Executor itself
    MultiCacheWeakPtr runtimeCache;
    DnnlScratchPadPtr scratchPad;
    dnnl::engine engine;
    std::vector<impl_desc_type> implPriorities;
};

class ExecutorFactory {
public:
    ExecutorFactory(const ExecutorContext::CPtr context) : context(context) {}
    virtual ~ExecutorFactory() = default;

    const ExecutorContext::CPtr context;
};

using ExecutorFactoryPtr = std::shared_ptr<ExecutorFactory>;
using ExecutorFactoryCPtr = std::shared_ptr<const ExecutorFactory>;

}   // namespace intel_cpu
}   // namespace ov
