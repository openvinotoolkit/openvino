// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_mvn.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>

#include "common/primitive_attr.hpp"
#include "cpu_memory.h"
#include "cpu_shape.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "nodes/executors/debug_messages.hpp"
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

using namespace dnnl::impl::primitive_hashing;

size_t hash_vector(const std::vector<float>& data) {
    size_t seed = data.size();
    for (float v : data) {
        seed = hash_combine(seed, v);
    }
    return seed;
}

size_t hash_fake_quantize_post_op(const FakeQuantizePostOp& fq) {
    size_t seed = static_cast<size_t>(fq.type());
    seed = hash_combine(seed, fq.levels());
    seed = hash_combine(seed, fq.isInputLowBroadcast());
    seed = hash_combine(seed, fq.isOutputHighBroadcast());
    seed = hash_combine(seed, hash_vector(fq.cropLow()));
    seed = hash_combine(seed, hash_vector(fq.cropHigh()));
    seed = hash_combine(seed, hash_vector(fq.inputScale()));
    seed = hash_combine(seed, hash_vector(fq.inputShift()));
    seed = hash_combine(seed, hash_vector(fq.outputScale()));
    seed = hash_combine(seed, hash_vector(fq.outputShift()));
    return seed;
}

size_t hash_scale_shift_post_op(const ScaleShiftPostOp& ss) {
    size_t seed = static_cast<size_t>(ss.type());
    seed = hash_combine(seed, hash_vector(ss.scales()));
    seed = hash_combine(seed, hash_vector(ss.shifts()));
    return seed;
}

size_t hash_activation_post_op(const ActivationPostOp& act) {
    size_t seed = static_cast<size_t>(act.type());
    seed = hash_combine(seed, act.alpha());
    seed = hash_combine(seed, act.beta());
    seed = hash_combine(seed, act.gamma());
    return seed;
}

size_t hash_post_ops(const PostOps& postOps) {
    size_t seed = postOps.size();
    for (const auto& postOp : postOps) {
        const auto& type = postOp.type();
        seed = hash_combine(seed, type.hash_code());
        if (type == typeid(FakeQuantizePostOp)) {
            seed = hash_combine(seed, hash_fake_quantize_post_op(std::any_cast<const FakeQuantizePostOp&>(postOp)));
        } else if (type == typeid(ScaleShiftPostOp)) {
            seed = hash_combine(seed, hash_scale_shift_post_op(std::any_cast<const ScaleShiftPostOp&>(postOp)));
        } else if (type == typeid(ActivationPostOp)) {
            seed = hash_combine(seed, hash_activation_post_op(std::any_cast<const ActivationPostOp&>(postOp)));
        }
    }
    return seed;
}

bool equal_fake_quantize_post_op(const FakeQuantizePostOp& lhs, const FakeQuantizePostOp& rhs) {
    return lhs.type() == rhs.type() && lhs.cropLow() == rhs.cropLow() && lhs.cropHigh() == rhs.cropHigh() &&
           lhs.inputScale() == rhs.inputScale() && lhs.inputShift() == rhs.inputShift() &&
           lhs.outputScale() == rhs.outputScale() && lhs.outputShift() == rhs.outputShift() &&
           lhs.levels() == rhs.levels() && lhs.isInputLowBroadcast() == rhs.isInputLowBroadcast() &&
           lhs.isOutputHighBroadcast() == rhs.isOutputHighBroadcast();
}

bool equal_scale_shift_post_op(const ScaleShiftPostOp& lhs, const ScaleShiftPostOp& rhs) {
    return lhs.type() == rhs.type() && lhs.scales() == rhs.scales() && lhs.shifts() == rhs.shifts();
}

bool equal_activation_post_op(const ActivationPostOp& lhs, const ActivationPostOp& rhs) {
    return lhs.type() == rhs.type() && lhs.alpha() == rhs.alpha() && lhs.beta() == rhs.beta() &&
           lhs.gamma() == rhs.gamma();
}

bool equal_post_ops(const PostOps& lhs, const PostOps& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i].type() != rhs[i].type()) {
            return false;
        }
        const auto& type = lhs[i].type();
        if (type == typeid(FakeQuantizePostOp)) {
            if (!equal_fake_quantize_post_op(std::any_cast<const FakeQuantizePostOp&>(lhs[i]),
                                             std::any_cast<const FakeQuantizePostOp&>(rhs[i]))) {
                return false;
            }
        } else if (type == typeid(ScaleShiftPostOp)) {
            if (!equal_scale_shift_post_op(std::any_cast<const ScaleShiftPostOp&>(lhs[i]),
                                           std::any_cast<const ScaleShiftPostOp&>(rhs[i]))) {
                return false;
            }
        } else if (type == typeid(ActivationPostOp)) {
            if (!equal_activation_post_op(std::any_cast<const ActivationPostOp&>(lhs[i]),
                                          std::any_cast<const ActivationPostOp&>(rhs[i]))) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

struct MVNKey {
    MVNAttrs mvnAttrs;
    dnnl::primitive_attr attr;
    ov::element::Type src_prc;
    ov::element::Type dst_prc;

    [[nodiscard]] size_t hash() const;
    bool operator==(const MVNKey& rhs) const;
};

size_t MVNKey::hash() const {
    size_t seed = 0;
    seed = hash_combine(seed, mvnAttrs.initAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.execAcrossChannels_);
    seed = hash_combine(seed, mvnAttrs.normalizeVariance_);
    seed = hash_combine(seed, mvnAttrs.epsValue_);
    seed = hash_combine(seed, mvnAttrs.epsMode_);
    seed = hash_combine(seed, mvnAttrs.layout);
    seed = hash_combine(seed, hash_post_ops(mvnAttrs.postOps));
    seed = hash_combine(seed, src_prc.hash());
    seed = hash_combine(seed, dst_prc.hash());
    seed = hash_combine(seed, get_attr_hash(*attr.get()));
    return seed;
}

bool MVNKey::operator==(const MVNKey& rhs) const {
    bool retVal = true;
    retVal = retVal && mvnAttrs.initAcrossChannels_ == rhs.mvnAttrs.initAcrossChannels_ &&
             mvnAttrs.execAcrossChannels_ == rhs.mvnAttrs.execAcrossChannels_ &&
             mvnAttrs.normalizeVariance_ == rhs.mvnAttrs.normalizeVariance_ &&
             mvnAttrs.epsValue_ == rhs.mvnAttrs.epsValue_ && mvnAttrs.epsMode_ == rhs.mvnAttrs.epsMode_ &&
             mvnAttrs.layout == rhs.mvnAttrs.layout;
    retVal = retVal && equal_post_ops(mvnAttrs.postOps, rhs.mvnAttrs.postOps);
    retVal = retVal && src_prc == rhs.src_prc && dst_prc == rhs.dst_prc && *attr.get() == *rhs.attr.get();
    return retVal;
}

}  // namespace

MVNJitExecutor::MVNJitExecutor(MVNAttrs mvnAttrs, MemoryArgs memory, ExecutorContext::CPtr contextPtr)
    : attrs(std::move(mvnAttrs)),
      memoryArgs(std::move(memory)),
      context(std::move(contextPtr)),
      shape5D(attrs.shape5D) {
    // Do not finalize kernel here: MVN node finalizes attrs (layout, exec flags)
    // in prepareParams(). We'll construct/reuse the legacy JIT in update().
}

void MVNJitExecutor::executeImpl(const MemoryArgs& memory) {
    const auto srcMem = memory.at(ARG_SRC);
    const auto dstMem = memory.at(ARG_DST);
    const auto* src_data = srcMem->getDataAs<const uint8_t>();
    auto* dst_data = dstMem->getDataAs<uint8_t>();
    const void* postOpsData = postOpsPtrArray.empty() ? nullptr : reinterpret_cast<const void*>(postOpsPtrArray.data());
    legacyJitExecutor->exec(src_data, dst_data, postOpsData, shape5D);
}

void MVNJitExecutor::setPostOps(dnnl::primitive_attr& attr, bool /*initWeights*/) {
    // Use the original DnnlPostOpsComposer approach which worked on master
    if (attrs.postOps.empty()) {
        return;
    }

    // Use DnnlPostOpsComposer to convert PostOps to dnnl format
    // For post-ops, we need to use the actual channel size and proper channel axis
    VectorDims outputDims = shape5D;

    // Derive logical channel axis (idxOC) consistent with MVN::prepareParams mapping and layout
    size_t idxOC = 1;  // default (N, C, D, H, W)
    if (attrs.layout == MVNLayoutType::mvn_by_channel) {
        // Channel-last (NHWC-like): channel is the last dim in our 5D mapping
        idxOC = outputDims.size() - 1;  // 4
    } else if (attrs.layout == MVNLayoutType::mvn_planar) {
        if (!attrs.execAcrossChannels_) {
            // Low-rank across-channels transformed cases
            // 1D across: {1,1,1,1,C} -> channel at index 4
            if (outputDims.size() == 5 && outputDims[0] == 1 && outputDims[1] == 1 && outputDims[2] == 1 &&
                outputDims[4] == outputDims[4]) {
                idxOC = 4;
            }
            // 2D across: {1,N,1,C,1} -> channel at index 3
            else if (outputDims.size() == 5 && outputDims[0] == 1 && outputDims[2] == 1) {
                idxOC = 3;
            } else {
                idxOC = 1;
            }
        } else {
            idxOC = 1;
        }
    } else {  // mvn_block
        idxOC = 1;
    }

    // Override the channel dimension with the actual channel size to match composer expectations
    if (attrs.actualChannelSize > 0 && idxOC < outputDims.size()) {
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

                // Expected channel size must align with logical channel dim
                // computed in MVN::prepareParams (attrs.actualChannelSize)
                size_t expectedChannelSize = attrs.actualChannelSize;

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

    const auto& src_prc = memoryArgs.at(ARG_SRC_0)->getDesc().getPrecision();
    const auto& dst_prc = memoryArgs.at(ARG_DST)->getDesc().getPrecision();
    const bool isINT8 = (src_prc == ov::element::i8 || src_prc == ov::element::u8) && dst_prc == ov::element::i8;
    const auto outDataType = DnnlExtensionUtils::ElementTypeToDataType(dst_prc);

    // Create memory args for post-ops composer
    MemoryArgs postOpsMemoryArgs = memoryArgs;
    // Create a dummy empty bias memory descriptor if not present
    if (postOpsMemoryArgs.count(ARG_BIAS) == 0) {
        auto biasDesc = std::make_shared<CpuBlockedMemoryDesc>(ov::element::f32, Shape{});
        postOpsMemoryArgs[ARG_BIAS] = std::make_shared<Memory>(context->getEngine(), biasDesc);
    }

    // Prefer legacy-style depthwise/quantization post-ops for MVN JIT to match legacy injectors
    DnnlPostOpsComposer composer(adjustedPostOps.empty() ? attrs.postOps : adjustedPostOps,
                                 context->getEngine(),
                                 outputDims,
                                 idxOC,
                                 isINT8,
                                 1 << 0,  // weight scale mask per channel
                                 postOpsMemoryArgs,
                                 outDataType,
                                 std::vector<float>{},
                                 PostOpsMode::ForcedLegacy,
                                 false);

    auto primAttrs = composer.compose();
    attr = primAttrs.attr;

    // Build legacy-compatible pointer table for post-ops data in the same order as attrs.postOps
    postOpsPtrArray.clear();
    postOpsMemory.clear();
    postOpsDataBuffer.clear();

    const auto* attrInternal = attr.get();
    if (attrInternal != nullptr) {
        const auto& postOps = attrInternal->post_ops_;
        for (int idx = 0; idx < postOps.len(); ++idx) {
            const auto& entry = postOps.entry_[idx];
            const int argKey = DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1;

            auto memoryIt = primAttrs.cpuArgs.find(argKey);
            MemoryPtr postOpMemory = (memoryIt != primAttrs.cpuArgs.end()) ? memoryIt->second : MemoryPtr{};

            if (entry.is_quantization()) {
                OPENVINO_ASSERT(postOpMemory, "Quantization post-op memory is not set");
                postOpsMemory.push_back(postOpMemory);
                auto* base = postOpMemory->getDataAs<float>();
                for (size_t offsetIdx = 0; offsetIdx < dnnl_post_ops::entry_t::quantization_t::fields_count;
                     ++offsetIdx) {
                    postOpsPtrArray.push_back(static_cast<void*>(base + entry.quantization.offset[offsetIdx]));
                }
            } else if (entry.is_depthwise()) {
                OPENVINO_ASSERT(postOpMemory, "Depthwise post-op memory is not set");
                postOpsMemory.push_back(postOpMemory);
                auto* base = postOpMemory->getDataAs<float>();
                postOpsPtrArray.push_back(static_cast<void*>(base + entry.depthwise.offset[entry.depthwise.scales]));
                postOpsPtrArray.push_back(static_cast<void*>(base + entry.depthwise.offset[entry.depthwise.shifts]));
            }
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

bool MVNJitExecutor::supports(const MVNConfig& config) {
    const auto& srcDesc = config.descs.at(ARG_SRC_0);
    const auto& dstDesc = config.descs.at(ARG_DST);

    VERIFY(srcDesc, UNSUPPORTED_BY_EXECUTOR);
    VERIFY(dstDesc, UNSUPPORTED_BY_EXECUTOR);
    VERIFY(!srcDesc->empty(), UNSUPPORTED_BY_EXECUTOR);
    VERIFY(!dstDesc->empty(), UNSUPPORTED_BY_EXECUTOR);

    return true;
}

static VectorDims to5D(const VectorDims& dims, const MVNAttrs& attrs) {
    VectorDims out;
    const size_t rank = dims.size();
    switch (rank) {
    case 0:
        out = {1, 1, 1, 1, 1};
        break;
    case 1:
        out = attrs.initAcrossChannels_ ? VectorDims{1, 1, 1, 1, dims[0]} : VectorDims{1, dims[0], 1, 1, 1};
        break;
    case 2:
        out = attrs.initAcrossChannels_ ? VectorDims{1, dims[0], 1, dims[1], 1} : VectorDims{dims[0], dims[1], 1, 1, 1};
        break;
    case 3:
        out = {dims[0], dims[1], 1, dims[2], 1};
        break;
    case 4:
        out = {dims[0], dims[1], 1, dims[2], dims[3]};
        break;
    default:
        out = {dims[0], dims[1], dims[2], dims[3], dims[4]};
        break;
    }
    return out;
}

bool MVNJitExecutor::update(const MemoryArgs& memory) {
    memoryArgs = memory;

    // 1) Refresh shape mapping from current input dims
    VectorDims in_dims;
    if (auto it = memory.find(ARG_SRC_0); it != memory.end()) {
        in_dims = it->second->getStaticDims();
        shape5D = to5D(in_dims, attrs);
    }

    // 2) Derive runtime layout from actual memory descriptor
    //    This reflects the selected PD (ncsp/nspc/blocked)
    if (auto it = memory.find(ARG_SRC_0); it != memory.end()) {
        auto md = it->second->getDescPtr();
        if (md->hasLayoutType(LayoutType::nspc)) {
            attrs.layout = MVNLayoutType::mvn_by_channel;
        } else if (md->hasLayoutType(LayoutType::ncsp)) {
            attrs.layout = MVNLayoutType::mvn_planar;
        } else {
            attrs.layout = MVNLayoutType::mvn_block;
        }
    }

    // 3) Harmonize execAcrossChannels_ for low-rank across-cases to match node::transformTo5DCase
    const size_t inRank = in_dims.size();
    if ((inRank == 1 || inRank == 2) && attrs.initAcrossChannels_) {
        // node::transformTo5DCase forces per-channel execution for low ranks
        attrs.execAcrossChannels_ = false;
    } else {
        attrs.execAcrossChannels_ = attrs.initAcrossChannels_;
    }

    // 4) Compute actual channel size for post-ops composition
    //    Default channel index in 5D is 1 (N,C,D,H,W)
    size_t chIndex5D = 1;
    if (attrs.layout == MVNLayoutType::mvn_planar) {
        if (inRank == 1 && attrs.initAcrossChannels_) {
            chIndex5D = 4;  // {1,1,1,1,C}
        } else if (inRank == 2 && attrs.initAcrossChannels_) {
            chIndex5D = 3;  // {1,N,1,C,1}
        } else {
            chIndex5D = 1;  // {N,C,...}
        }
    } else if (attrs.layout == MVNLayoutType::mvn_by_channel) {
        chIndex5D = 4;  // NHWC-like channel index
    } else {            // mvn_block
        chIndex5D = 1;
    }
    attrs.actualChannelSize = shape5D.size() > chIndex5D ? shape5D[chIndex5D] : 1;

    // 5) Compose post-ops with updated dims/layout and rebuild/reuse legacy JIT by key
    dnnl::primitive_attr computedAttr;
    setPostOps(computedAttr, true);

    const auto& src_prc = memoryArgs.at(ARG_SRC_0)->getDesc().getPrecision();
    const auto& dst_prc = memoryArgs.at(ARG_DST)->getDesc().getPrecision();
    MVNKey key{attrs, computedAttr, src_prc, dst_prc};

    auto builder = [&](const MVNKey& k) -> std::shared_ptr<legacy::MVNJitExecutorLegacy> {
        return std::make_shared<legacy::MVNJitExecutorLegacy>(k.mvnAttrs, k.attr, k.src_prc, k.dst_prc);
    };

    if (context) {
        auto cache = context->getRuntimeCache();
        auto result = cache->getOrCreate(key, builder);
        legacyJitExecutor = result.first;
    } else {
        legacyJitExecutor = builder(key);
    }

    return true;
}

}  // namespace ov::intel_cpu
