// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "cpu_shape.h"
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_config.hpp"
#include "nodes/executors/executor_implementation.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/implementations.hpp"
#include "nodes/executors/interpolate.hpp"
#include "nodes/executors/interpolate_config.hpp"
#include "nodes/executors/interpolate_ref.hpp"
#include "memory_format_filter.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86_64)
#include <cpu/x64/cpu_isa_traits.hpp>
#include "nodes/executors/x64/interpolate_jit.hpp"
#endif

#if defined(OV_CPU_WITH_ACL)
#include "nodes/executors/acl/acl_interpolate.hpp"
#endif

namespace ov::intel_cpu {

using namespace executor;
using namespace ov::element;
using namespace dnnl::impl::cpu;
using ov::intel_cpu::any_of;

static bool isJitApplicable(const executor::Config<InterpolateAttrs>& config,
                            const MemoryFormatFilter& filter) {
#if defined(OPENVINO_ARCH_X86_64)
    const auto& attrs = config.attrs;
    
    // Check if mode is supported by JIT
    if (!any_of(attrs.mode, 
                InterpolateMode::nearest,
                InterpolateMode::linear,
                InterpolateMode::linear_onnx,
                InterpolateMode::cubic,
                InterpolateMode::bilinear_pillow,
                InterpolateMode::bicubic_pillow)) {
        return false;
    }
    
    // Check layout support
    bool isNearestLinearOrCubic = attrs.mode == InterpolateMode::nearest ||
                                  attrs.mode == InterpolateMode::linear_onnx ||
                                  attrs.mode == InterpolateMode::cubic;
    bool isPlanarLayourAndSse41 = attrs.layout != InterpolateLayoutType::planar && x64::mayiuse(x64::sse41);
    bool isAvx2AndF32 = x64::mayiuse(x64::avx2) && attrs.inPrc == ov::element::f32;
    bool isPillowMode = attrs.mode == InterpolateMode::bilinear_pillow ||
                        attrs.mode == InterpolateMode::bicubic_pillow;
    bool isByChannelLayout = attrs.layout == InterpolateLayoutType::by_channel;
    bool isNearestLinearOrCubicSupported = isNearestLinearOrCubic && (isPlanarLayourAndSse41 || isAvx2AndF32);
    bool isPillowModeSupported = isPillowMode && isByChannelLayout;

    return (isNearestLinearOrCubicSupported || isPillowModeSupported) && x64::mayiuse(x64::sse41);
#else
    return false;
#endif
}

static bool isACLApplicable(const executor::Config<InterpolateAttrs>& config,
                            const MemoryFormatFilter& filter) {
#if defined(OV_CPU_WITH_ACL)
    const auto& attrs = config.attrs;
    
    // ACL supports limited modes
    if (!any_of(attrs.mode, 
                InterpolateMode::nearest,
                InterpolateMode::linear,
                InterpolateMode::linear_onnx)) {
        return false;
    }
    
    // Check memory descriptors to ensure both source and destination have compatible layouts
    // Port 0 is the data input for interpolate
    auto srcDesc = config.descs.count(0) ? config.descs.at(0) : nullptr;
    auto dstDesc = config.descs.count(ARG_DST) ? config.descs.at(ARG_DST) : nullptr;
    
    if (!srcDesc || !dstDesc) {
        return false;
    }
    
    // ACL works best with NHWC layout
    bool isNHWC = srcDesc->hasLayoutType(LayoutType::nspc) && 
                  dstDesc->hasLayoutType(LayoutType::nspc);
    
    // For NCHW layout, we need the NCHWAsNHWC flag to handle dimension swapping
    bool canHandleNCHW = attrs.NCHWAsNHWC;
    
    return isNHWC || canHandleNCHW;
#else
    return false;
#endif
}


// Factory function to get all implementations
template <>
const std::vector<ExecutorImplementation<InterpolateAttrs>>& getImplementations<InterpolateAttrs>() {
    static const std::vector<ExecutorImplementation<InterpolateAttrs>> implementations = {
        // Register implementations with priority (highest priority first)
        
#if defined(OPENVINO_ARCH_X86_64)
        // JIT implementation using old JIT executor
        ExecutorImplementation<InterpolateAttrs>(
            "jit_interpolate",
            ExecutorType::Jit,
            OperationType::Interpolate,
            isJitApplicable,
            nullptr,  // createOptimalConfig
            nullptr,  // acceptsShape
            [](const InterpolateAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> ExecutorPtr {
#if defined(OPENVINO_ARCH_X86_64)
                auto executor = std::make_shared<JitInterpolateExecutor>(context);
                
                // Build memory descriptors - JIT expects only the data input
                std::vector<MemoryDescPtr> srcDescs;
                std::vector<MemoryDescPtr> dstDescs;
                
                // Port 0 is the data input for interpolate node
                if (memory.count(0)) {
                    srcDescs.push_back(memory.at(0)->getDescPtr());
                } else {
                    return nullptr;
                }
                
                if (memory.count(ARG_DST)) {
                    dstDescs.push_back(memory.at(ARG_DST)->getDescPtr());
                } else {
                    return nullptr;
                }
                
                // Initialize the executor with config and memory descriptors
                dnnl::primitive_attr attr;
                if (executor->init(attrs, srcDescs, dstDescs, attr)) {
                    return executor;
                }
#endif
                return nullptr;
            }
        ),
#endif
        
#if defined(OV_CPU_WITH_ACL)
        // ACL implementation
        ExecutorImplementation<InterpolateAttrs>(
            "acl_interpolate",
            ExecutorType::Acl,
            OperationType::Interpolate,
            isACLApplicable,
            nullptr,  // createOptimalConfig
            nullptr,  // acceptsShape
            [](const InterpolateAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> ExecutorPtr {
#if defined(OV_CPU_WITH_ACL)
                auto executor = std::make_shared<ACLInterpolateExecutor>(context);
                
                // Build memory descriptors - ACL expects only the data input
                std::vector<MemoryDescPtr> srcDescs;
                std::vector<MemoryDescPtr> dstDescs;
                
                // Port 0 is the data input for interpolate node
                if (memory.count(0)) {
                    srcDescs.push_back(memory.at(0)->getDescPtr());
                } else {
                    return nullptr;
                }
                
                if (memory.count(ARG_DST)) {
                    dstDescs.push_back(memory.at(ARG_DST)->getDescPtr());
                } else {
                    return nullptr;
                }
                
                // Validate that descriptors are appropriate for ACL
                if (!srcDescs[0] || !dstDescs[0]) {
                    return nullptr;
                }
                
                // ACL expects 4D tensors
                if (srcDescs[0]->getShape().getDims().size() != 4 ||
                    dstDescs[0]->getShape().getDims().size() != 4) {
                    return nullptr;
                }
                
                // Initialize the executor with config and memory descriptors
                dnnl::primitive_attr attr;
                if (executor->init(attrs, srcDescs, dstDescs, attr)) {
                    return executor;
                }
#endif
                return nullptr;
            }
        ),
#endif
        
        // Reference implementation using old reference executor (lowest priority)
        ExecutorImplementation<InterpolateAttrs>(
            "ref_interpolate",
            ExecutorType::Reference,
            OperationType::Interpolate,
            [](const executor::Config<InterpolateAttrs>& config,
               const MemoryFormatFilter& filter) -> bool {
                // Reference implementation supports all cases
                return true;
            },
            nullptr,  // createOptimalConfig
            nullptr,  // acceptsShape
            [](const InterpolateAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr& context) -> ExecutorPtr {
                // Create new reference executor
                auto executor = std::make_shared<NewRefInterpolateExecutor>(context);
                
                // Build memory descriptors - reference executor needs all inputs
                std::vector<MemoryDescPtr> srcDescs;
                std::vector<MemoryDescPtr> dstDescs;
                
                // Collect all source descriptors in port order
                for (size_t i = 0; i < 10; ++i) { // Check up to 10 ports
                    if (memory.count(i)) {
                        srcDescs.push_back(memory.at(i)->getDescPtr());
                    } else {
                        break; // Stop at first missing port
                    }
                }
                
                if (memory.count(ARG_DST)) {
                    dstDescs.push_back(memory.at(ARG_DST)->getDescPtr());
                } else {
                    return nullptr;
                }
                
                if (srcDescs.empty() || dstDescs.empty()) {
                    return nullptr;
                }
                
                // Initialize the executor with config and memory descriptors
                dnnl::primitive_attr attr;
                if (executor->init(attrs, srcDescs, dstDescs, attr)) {
                    return executor;
                }
                
                return nullptr;
            }
        )
    };
    
    return implementations;
}

}  // namespace ov::intel_cpu