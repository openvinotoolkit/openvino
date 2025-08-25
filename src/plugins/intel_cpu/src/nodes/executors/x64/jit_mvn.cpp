// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mvn.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <any>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <utility>
#include <vector>

#include "common/primitive_attr.hpp"
#include "cpu_memory.h"
#include "cpu_shape.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "nodes/kernels/x64/mlp_utils.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace ov::intel_cpu {

namespace {

struct MVNKey {
    MVNAttrs mvnAttrs;
    dnnl::primitive_attr attr;

    [[nodiscard]] size_t hash() const;
    bool operator==(const MVNKey& rhs) const;
};

size_t MVNKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, mvnAttrs.initAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.execAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.normalizeVariance_);
    seed = hash_combine(seed, mvnAttrs.epsValue_);
    seed = hash_combine(seed, mvnAttrs.epsMode_);
    seed = hash_combine(seed, mvnAttrs.src_prc.hash());
    seed = hash_combine(seed, mvnAttrs.dst_prc.hash());
    seed = hash_combine(seed, mvnAttrs.layout);
    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool MVNKey::operator==(const MVNKey& rhs) const {
    bool retVal = true;
    retVal = retVal && mvnAttrs.initAcrossChannels_ == rhs.mvnAttrs.initAcrossChannels_ &&
             mvnAttrs.execAcrossChannels_ == rhs.mvnAttrs.execAcrossChannels_ &&
             mvnAttrs.normalizeVariance_ == rhs.mvnAttrs.normalizeVariance_ &&
             mvnAttrs.epsValue_ == rhs.mvnAttrs.epsValue_ && mvnAttrs.epsMode_ == rhs.mvnAttrs.epsMode_ &&
             mvnAttrs.src_prc == rhs.mvnAttrs.src_prc && mvnAttrs.dst_prc == rhs.mvnAttrs.dst_prc &&
             mvnAttrs.layout == rhs.mvnAttrs.layout;
    retVal = retVal && *attr.get() == *rhs.attr.get();
    return retVal;
}

}  // namespace

MVNJitExecutor::MVNJitExecutor(MVNAttrs mvnAttrs, MemoryArgs memory, ExecutorContext::CPtr contextPtr)
    : attrs(std::move(mvnAttrs)),
      memoryArgs(std::move(memory)),
      context(std::move(contextPtr)),
      shape5D(attrs.shape5D) {
    // Set post-ops in dnnl::primitive_attr
    setPostOps(attrs.attr, true);

    // Create a key for caching using the attr from MVNAttrs
    MVNKey key{attrs, attrs.attr};

    auto builder = [&](const MVNKey& key) -> std::shared_ptr<legacy::MVNJitExecutorLegacy> {
        return std::make_shared<legacy::MVNJitExecutorLegacy>(key.mvnAttrs, key.attr);
    };

    // Use context's cache if available
    if (context) {
        auto cache = context->getRuntimeCache();
        auto result = cache->getOrCreate(key, builder);
        legacyJitExecutor = result.first;
    } else {
        // Fallback if no context available
        legacyJitExecutor = builder(key);
    }
}

void MVNJitExecutor::executeImpl(const MemoryArgs& memory) {
    // Extract memory pointers from MemoryArgs
    const auto* src_data = memory.at(ARG_SRC)->getDataAs<const uint8_t>();
    auto* dst_data = memory.at(ARG_DST)->getDataAs<uint8_t>();

    // Pass post-ops data to legacy executor
    // Legacy MVN expects an array of float* pointers
    const void* postOpsData = postOpsPtrArray.empty() ? nullptr : reinterpret_cast<const void*>(postOpsPtrArray.data());
    legacyJitExecutor->exec(src_data, dst_data, postOpsData, shape5D);
}

void MVNJitExecutor::setPostOps(dnnl::primitive_attr& attr, bool /*initWeights*/) {
    // Use the original DnnlPostOpsComposer approach which worked on master
    if (attrs.postOps.empty()) {
        return;
    }

    // Use DnnlPostOpsComposer to convert PostOps to dnnl format
    // For post-ops, we need to use the actual channel size and create appropriate output dimensions
    VectorDims outputDims = shape5D;
    size_t idxOC = attrs.layout == MVNLayoutType::mvn_by_channel ? outputDims.size() - 1 : 1;

    // Override the channel dimension with the actual channel size
    // This ensures post-ops scale/shift sizes match the expected channel count
    if (attrs.actualChannelSize > 0) {
        outputDims[idxOC] = attrs.actualChannelSize;
    }

    // Handle broadcasting pattern for Instance Normalization
    // Style transfer models often use [1,C,1,1] constants which need special handling
    PostOps adjustedPostOps;
    for (const auto& postOp : attrs.postOps) {
        if (postOp.type() == typeid(ScaleShiftPostOp)) {
            try {
                const auto& scaleShiftOp = std::any_cast<const ScaleShiftPostOp&>(postOp);

                std::vector<float> adjustedScales = scaleShiftOp.scales();
                std::vector<float> adjustedShifts = scaleShiftOp.shifts();

                // For MVN, if we're processing across channels, the actual output channel size might be shape5D[1]
                size_t expectedChannelSize = attrs.actualChannelSize;
                if (shape5D.size() > 1 && !attrs.execAcrossChannels_) {
                    expectedChannelSize = shape5D[1];
                }

                // Check if scales have broadcasting pattern [1,C,1,1] or just C values
                if (adjustedScales.size() != expectedChannelSize && adjustedScales.size() != 1) {
                    // This might be a flattened pattern, try to extract channel values
                    if (adjustedScales.size() == expectedChannelSize * 1 * 1) {
                        std::vector<float> channelScales;
                        for (size_t c = 0; c < expectedChannelSize; ++c) {
                            channelScales.push_back(adjustedScales[c]);
                        }
                        adjustedScales = channelScales;
                    }
                }

                // Same for shifts
                if (adjustedShifts.size() != expectedChannelSize && adjustedShifts.size() != 1) {
                    if (adjustedShifts.size() == expectedChannelSize * 1 * 1) {
                        std::vector<float> channelShifts;
                        for (size_t c = 0; c < expectedChannelSize; ++c) {
                            channelShifts.push_back(adjustedShifts[c]);
                        }
                        adjustedShifts = channelShifts;
                    }
                }

                // Update outputDims to match the actual channel size if needed
                if (expectedChannelSize != outputDims[idxOC]) {
                    outputDims[idxOC] = expectedChannelSize;
                }

                // Create new ScaleShiftPostOp with adjusted values
                adjustedPostOps.push_back(
                    std::make_any<ScaleShiftPostOp>(scaleShiftOp.type(), adjustedScales, adjustedShifts));
            } catch (const std::bad_any_cast&) {
                // Not a ScaleShiftPostOp, add as is
                adjustedPostOps.push_back(postOp);
            }
        } else {
            // Keep other post-ops as is
            adjustedPostOps.push_back(postOp);
        }
    }

    const bool isINT8 =
        (attrs.src_prc == ov::element::i8 || attrs.src_prc == ov::element::u8) && attrs.dst_prc == ov::element::i8;
    const auto outDataType = DnnlExtensionUtils::ElementTypeToDataType(attrs.dst_prc);

    // Create memory args for post-ops composer
    MemoryArgs postOpsMemoryArgs = memoryArgs;
    // Create a dummy empty bias memory descriptor if not present
    if (postOpsMemoryArgs.count(ARG_BIAS) == 0) {
        auto biasDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape{});
        postOpsMemoryArgs[ARG_BIAS] = std::make_shared<Memory>(context->getEngine(), biasDesc);
    }

    DnnlPostOpsComposer composer(adjustedPostOps.empty() ? attrs.postOps : adjustedPostOps,
                                 context->getEngine(),
                                 outputDims,
                                 idxOC,
                                 isINT8,
                                 1 << 0,  // weight scale mask per channel
                                 postOpsMemoryArgs,
                                 outDataType);

    auto primAttrs = composer.compose();
    attr = primAttrs.attr;

    // Clear previous data
    postOpsDataPtrs.clear();
    postOpsDataBuffer.clear();
    postOpsPtrArray.clear();
    postOpsMemory.clear();

    // For legacy MVN, we need to create an array of pointers
    // The legacy implementation expects an array where each element is a pointer to post-op data

    // Collect all post-ops data memory from DnnlPostOpsComposer
    for (const auto& cpuArg : primAttrs.cpuArgs) {
        // Check if this is post-op data
        if (cpuArg.first >= DNNL_ARG_ATTR_MULTIPLE_POST_OP(0)) {
            // Keep the memory alive by storing the MemoryPtr
            postOpsMemory.push_back(cpuArg.second);

            const auto* memPtr = cpuArg.second.get();
            if (memPtr && memPtr->getData()) {
                postOpsDataPtrs.push_back(memPtr->getData());
            }
        }
    }

    // For legacy MVN, we need to create an array of float* pointers
    // The legacy implementation increments by sizeof(float*) when accessing post-ops data
    if (!postOpsDataPtrs.empty()) {
        // Legacy MVN expects an array of pointers where each pointer points to the data for one post-op
        // For FakeQuantize, this should be a pointer to a buffer containing:
        // [cropLow][cropHigh][inputScale][inputShift][outputScale][outputShift]

        // Create the pointer array that legacy MVN expects
        postOpsPtrArray.clear();

        // For each post-op data pointer, add it to the array
        for (const auto& ptr : postOpsDataPtrs) {
            postOpsPtrArray.push_back(const_cast<void*>(ptr));
        }
    }
}

bool MVNJitExecutor::canReuseShapeAgnosticKernel(const VectorDims& newShape5D) {
    // Shape-agnostic kernel optimization
    // Reuses kernel if the shape is the same or only batch size changed
    if (shape5D[0] != newShape5D[0]) {
        if (shape5D[1] == newShape5D[1] && shape5D[2] == newShape5D[2] && shape5D[3] == newShape5D[3] &&
            shape5D[4] == newShape5D[4]) {
            shape5D = newShape5D;
            return true;
        }
    }
    return false;
}

bool MVNJitExecutor::supports(const MVNConfig& /*config*/) {
    // JIT implementation supports all precisions
    // The legacy implementation handles precision conversions internally
    return true;
}

}  // namespace ov::intel_cpu