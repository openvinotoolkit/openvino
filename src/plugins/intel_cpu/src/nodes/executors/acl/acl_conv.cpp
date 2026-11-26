// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_conv.hpp"

#include <arm_compute/core/CoreTypes.h>
#include <arm_compute/core/Error.h>
#include <arm_compute/core/QuantizationInfo.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "acl_utils.hpp"
#include "cpu_parallel.hpp"
#include "cpu_shape.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/acl/acl_common_executor.hpp"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/convolution_config.hpp"
#include "nodes/executors/debug_messages.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_ARM64)
#    include "nodes/kernels/aarch64/jit_sdot_convolution.hpp"
#    include "utils/arm_isa_support.h"
#endif

namespace ov::intel_cpu {

#if defined(OPENVINO_ARCH_ARM64)
namespace {

class Oc1DotConvolution : public arm_compute::IFunction {
public:
    Oc1DotConvolution(const ACLTensors& tensors,
                      const arm_compute::PadStrideInfo& padding,
                      std::shared_ptr<CpuParallel> parallel)
        : m_src(tensors[ACL_SRC_0].get()),
          m_weights(tensors[ACL_WEI].get()),
          m_bias(tensors[ACL_BIAS].get()),
          m_dst(tensors[ACL_DST].get()),
          m_parallel(std::move(parallel)),
          m_channels(m_src->info()->dimension(0)),
          m_unsigned_input(m_src->info()->data_type() == arm_compute::DataType::QASYMM8),
          m_zero_point(m_src->info()->quantization_info().uniform().offset),
          m_kernel(m_channels, m_unsigned_input),
          m_pad_top(padding.pad_top()),
          m_pad_left(padding.pad_left()),
          m_scale(m_weights->info()->quantization_info().uniform().scale) {}

    void run() override {
        const auto* src = reinterpret_cast<const int8_t*>(m_src->buffer());
        const auto* weights = reinterpret_cast<const int8_t*>(m_weights->buffer());
        auto* dst = reinterpret_cast<float*>(m_dst->buffer());
        const float bias = m_bias ? *reinterpret_cast<const float*>(m_bias->buffer()) : 0.F;
        const auto input_w = m_src->info()->dimension(1);
        const auto input_h = m_src->info()->dimension(2);
        const auto output_w = m_dst->info()->dimension(1);
        const auto output_h = m_dst->info()->dimension(2);
        const auto batch = m_src->info()->dimension(3);
        std::array<uint8_t, 128> padding{};
        padding.fill(static_cast<uint8_t>(m_zero_point));
        const auto* padding_bytes = reinterpret_cast<const int8_t*>(padding.data());
        // SDOT reads U8 bytes rebased by 128; restore (input - zero_point) in the accumulator.
        const int32_t correction =
            m_unsigned_input ? (128 - m_zero_point) * std::accumulate(weights, weights + 9 * m_channels, 0) : 0;

        m_parallel->parallel_for(batch * output_h, [&](size_t row) {
            constexpr size_t tile = 64;
            std::array<const int8_t*, tile * 9> taps;
            const auto y = row % output_h;
            const auto* image = src + (row / output_h) * input_h * input_w * m_channels;
            for (size_t x = 0; x < output_w; x += tile) {
                const auto count = std::min(tile, output_w - x);
                for (size_t ky = 0; ky < 3; ++ky) {
                    const auto iy = y + ky - m_pad_top;
                    const auto* line = iy < input_h ? image + iy * input_w * m_channels : nullptr;
                    for (size_t dx = 0; dx < count; ++dx) {
                        for (size_t kx = 0; kx < 3; ++kx) {
                            const auto ix = x + dx + kx - m_pad_left;
                            taps[dx * 9 + ky * 3 + kx] = line && ix < input_w ? line + ix * m_channels : padding_bytes;
                        }
                    }
                }
                m_kernel(taps.data(), weights, dst + row * output_w + x, count, correction, m_scale, bias);
            }
        });
    }

private:
    arm_compute::Tensor* m_src;
    arm_compute::Tensor* m_weights;
    arm_compute::Tensor* m_bias;
    arm_compute::Tensor* m_dst;
    std::shared_ptr<CpuParallel> m_parallel;
    size_t m_channels;
    bool m_unsigned_input;
    int32_t m_zero_point;
    aarch64::Oc1DotKernel m_kernel;
    size_t m_pad_top;
    size_t m_pad_left;
    float m_scale;
};

}  // namespace

bool ACLConvolutionExecutor::supportsOc1JitGeometry(const VectorDims& src,
                                                    const VectorDims& weights,
                                                    const VectorDims& dst,
                                                    const std::vector<size_t>& strides,
                                                    const std::vector<size_t>& dilations,
                                                    const std::vector<ptrdiff_t>& padsBegin,
                                                    const std::vector<ptrdiff_t>& padsEnd) {
    const auto nonnegative = [](ptrdiff_t pad) {
        return pad >= 0;
    };
    return hasArmISASupport(ArmISA::DOTPROD) && src.size() == 4 && weights.size() == 4 && dst.size() == 4 &&
           src[0] == dst[0] && src[1] >= 4 && src[1] <= 128 && src[1] % 4 == 0 && weights[0] == 1 &&
           weights[1] == src[1] && weights[2] == 3 && weights[3] == 3 && dst[1] == 1 && dst[3] >= 16 &&
           dst[2] * dst[3] >= 1024 && strides == std::vector<size_t>{1, 1} && dilations == std::vector<size_t>{0, 0} &&
           padsBegin.size() == 2 && padsEnd.size() == 2 &&
           std::all_of(padsBegin.begin(), padsBegin.end(), nonnegative) &&
           std::all_of(padsEnd.begin(), padsEnd.end(), nonnegative);
}
#endif

ACLConvolutionExecutor::ACLConvolutionExecutor(const ConvAttrs& attrs,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context)
    : cpuParallel(context->getCpuParallel()),
      srcZeroPoint(attrs.inputZeroPoint),
      weightScale(attrs.dqScales) {
    aclTensorAttrs.hasLayoutTypeNHWC = memory.at(ARG_SRC)->getDescPtr()->hasLayoutType(LayoutType::nspc);

    MemoryDescPtr srcMemPtr = memory.at(ARG_SRC_0)->getDescPtr();
    MemoryDescPtr weiMemPtr = memory.at(ARG_WEI)->getDescPtr();
    MemoryDescPtr dstMemPtr = memory.at(ARG_DST)->getDescPtr();

    Shape weiShape = weiMemPtr->getShape();
    Shape srcShape = srcMemPtr->getShape();
    Shape dstShape = dstMemPtr->getShape();

    const auto with_groups = static_cast<int>(weiShape.getRank() == srcShape.getRank() + 1);
    const int kh = weiShape.getDims()[with_groups + srcShape.getRank() - 2];
    const int kw = weiShape.getDims()[with_groups + srcShape.getRank() - 1];
    const int oc = dstShape.getDims()[1];

#if defined(OPENVINO_ARCH_ARM64)
    const bool signedInput = srcMemPtr->getPrecision() == ov::element::i8 &&
                             attrs.inputZeroPointsType == ZeroPointsType::None && srcZeroPoint == 0;
    const bool unsignedInput = srcMemPtr->getPrecision() == ov::element::u8 &&
                               attrs.inputZeroPointsType == ZeroPointsType::PerTensorUnsignedOc1Jit &&
                               srcZeroPoint >= 0 && srcZeroPoint <= 255;
    useOc1Jit = (signedInput || unsignedInput) && !attrs.isGrouped && attrs.constantWeights && attrs.postOps.empty() &&
                weightScale.size() <= 1 &&
                supportsOc1JitGeometry(srcShape.getDims(),
                                       weiShape.getDims(),
                                       dstShape.getDims(),
                                       attrs.stride,
                                       attrs.dilation,
                                       attrs.paddingL,
                                       attrs.paddingR) &&
                weiMemPtr->getPrecision() == ov::element::i8 && dstMemPtr->getPrecision() == ov::element::f32 &&
                aclTensorAttrs.hasLayoutTypeNHWC && weiMemPtr->hasLayoutType(LayoutType::nspc) &&
                dstMemPtr->hasLayoutType(LayoutType::nspc);
#endif
    weightsInfo = arm_compute::WeightsInfo(false, kw, kh, oc, false, arm_compute::WeightFormat::UNSPECIFIED);
    auto paddingLeft = (attrs.paddingL.size() >= 2U) ? attrs.paddingL[1] : attrs.paddingL[0];
    auto paddingRight = (attrs.paddingR.size() >= 2U) ? attrs.paddingR[1] : attrs.paddingR[0];
    auto paddingTop = (attrs.paddingL.size() >= 2U) ? attrs.paddingL[0] : 0;
    auto paddingBottom = (attrs.paddingR.size() >= 2U) ? attrs.paddingR[0] : 0;
    padStrideInfo = arm_compute::PadStrideInfo(attrs.stride[0],
                                               attrs.stride[1],
                                               paddingLeft,
                                               paddingRight,
                                               paddingTop,
                                               paddingBottom,
                                               arm_compute::DimensionRoundingType::FLOOR);
    dilation = arm_compute::Size2D(attrs.dilation[1] + 1, attrs.dilation[0] + 1);
    // Instead of checking ==1 ==2 branch, iterate throught the vector
    for (const auto& postOp : attrs.postOps) {
        if (const auto* const activation = std::any_cast<ActivationPostOp>(&postOp)) {
            activationLayerInfo = getActivationLayerInfo(convertToEltwiseAlgorithm(activation->type()),
                                                         activation->alpha(),
                                                         activation->beta(),
                                                         activation->gamma());
        } else if (const auto* const fq = std::any_cast<FakeQuantizePostOp>(&postOp)) {
            fqInputScale = fq->inputScale();
            fqInputShift = fq->inputShift();
            fqOutputScale = fq->outputScale();
            fqOutputShift = fq->outputShift();

            // ACL destination quantization supports only per-tensor scale/shift.
            // For per-channel FQ input scales, move channel-wise ratios to weights scales.
            if (fqInputScale.size() > 1) {
                const auto baseScale = fqInputScale.front();
                const bool hasValidBaseScale = std::fabs(baseScale) > std::numeric_limits<float>::epsilon();
                const bool hasUniformShift = isPerTensorDataWithTolerance(fqInputShift, 0.00005F);

                if (hasValidBaseScale && hasUniformShift) {
                    std::vector<float> scaleRatios(fqInputScale.size(), 1.0F);
                    for (std::size_t i = 0; i < fqInputScale.size(); i++) {
                        scaleRatios[i] = fqInputScale[i] / baseScale;
                    }

                    multiplyAndBroadcastScales(weightScale, scaleRatios, fqInputScale.size());
                    fqInputScale = {baseScale};
                    if (fqInputShift.size() > 1) {
                        fqInputShift = {fqInputShift.front()};
                    }
                }
            }

            foldFakeQuantizeOutputShift(fqInputShift, fqOutputScale, fqOutputShift);
            fqOutputShift.clear();
        } else {
            OPENVINO_THROW("ACLConvolutionExecutor: the executor supports FakeQuantize and Activation post ops only");
        }
    }
}

bool ACLConvolutionExecutor::supports(const ConvConfig& config) {
    VERIFY(config.attrs.postOps.size() <= 2U, UNSUPPORTED_BY_EXECUTOR);

    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);

    VERIFY(aclSupported({srcDesc, weiDesc, dstDesc}), UNSUPPORTED_ACL_COMMON_PRECONDITION);

    // ACL GemmConv2d supports 4D activations and 4D weight only
    VERIFY(srcDesc->getShape().getRank() == 4 && weiDesc->getShape().getRank() == 4, UNSUPPORTED_BY_EXECUTOR);
    // isQuantized verifies whether src is u8/i8, weights is i8 and FQ is fused if dst is u8/i8
    // the last requirement is due to ACL int32 accumulation that needs to be requantized by non-trivial scales
    const bool hasQuantizationPostOp =
        std::any_of(config.attrs.postOps.begin(), config.attrs.postOps.end(), [](const std::any& op) {
            return std::any_cast<FakeQuantizePostOp>(&op) != nullptr;
        });
    const bool isQuantizedU8 = srcDesc->getPrecision() == ov::element::u8 &&
                               any_of(weiDesc->getPrecision(), ov::element::u8, ov::element::i8) &&
                               dstDesc->getPrecision() == ov::element::u8 && hasQuantizationPostOp;
    const bool isQuantizedI8 = srcDesc->getPrecision() == ov::element::i8 &&
                               weiDesc->getPrecision() == ov::element::i8 &&
                               dstDesc->getPrecision() == ov::element::i8 && hasQuantizationPostOp;
    bool useUnsignedJit = false;
#if defined(OPENVINO_ARCH_ARM64)
    useUnsignedJit =
        srcDesc->getPrecision() == ov::element::u8 && config.attrs.constantWeights && !config.attrs.isGrouped &&
        config.attrs.postOps.empty() && config.attrs.inputZeroPointsType == ZeroPointsType::PerTensorUnsignedOc1Jit &&
        config.attrs.inputZeroPoint >= 0 && config.attrs.inputZeroPoint <= 255 && srcDesc->getShape().isStatic() &&
        weiDesc->getShape().isStatic() && dstDesc->getShape().isStatic() &&
        supportsOc1JitGeometry(srcDesc->getShape().getDims(),
                               weiDesc->getShape().getDims(),
                               dstDesc->getShape().getDims(),
                               config.attrs.stride,
                               config.attrs.dilation,
                               config.attrs.paddingL,
                               config.attrs.paddingR);
#endif
    const bool isQuantizedI8DstF32 = (srcDesc->getPrecision() == ov::element::i8 || useUnsignedJit) &&
                                     weiDesc->getPrecision() == ov::element::i8 &&
                                     dstDesc->getPrecision() == ov::element::f32;

    VERIFY(isQuantizedU8 || isQuantizedI8 || isQuantizedI8DstF32, UNSUPPORTED_BY_EXECUTOR);
    VERIFY(!isQuantizedI8DstF32 || config.attrs.dqScales.size() <= 1, UNSUPPORTED_PER_CHANNEL_QUANTIZATION);
    if (config.attrs.withBias) {
        const auto biasPrecision = config.descs.at(ARG_BIAS)->getPrecision();
        if (isQuantizedI8DstF32) {
            VERIFY(biasPrecision == ov::element::f32, UNSUPPORTED_BIAS_PRECISIONS);
        } else {
            VERIFY(biasPrecision == ov::element::i32, UNSUPPORTED_BIAS_PRECISIONS);
        }
    }

    return true;
}

arm_compute::Status ACLConvolutionExecutor::validateTensorsInfo(const ACLInfos& aclMemoryInfos) {
    // Note: LPT propagate dequantization scales from src and weights on conv output, and the result scale
    // is applied as weight scale. So quantization configuration forms in the following way:
    // - src quantization info is always trivial
    // - weights: scale is equal to result dequantization scale after Convolution propagated by LPT
    //            shift is not supported
    // - destination: scale is formed based on requantization FakeQuantize parameters: scale = 1.0 / input scale
    //                shift = input shift. Skipped for f32 dst (dequantize-to-float path).
    aclMemoryInfos[ACLArgs::ACL_SRC_0]->set_quantization_info(arm_compute::QuantizationInfo(1.0F, srcZeroPoint));
    aclMemoryInfos[ACLArgs::ACL_WEI]->set_quantization_info(
        weightScale.empty() ? arm_compute::QuantizationInfo(1.0F) : arm_compute::QuantizationInfo(weightScale));
    if (useOc1Jit) {
        return {};
    }
    if (aclMemoryInfos[ACLArgs::ACL_DST]->data_type() != arm_compute::DataType::F32) {
        const auto dstPrecision = aclMemoryInfos[ACLArgs::ACL_DST]->data_type() == arm_compute::DataType::QASYMM8_SIGNED
                                      ? ov::element::i8
                                      : ov::element::u8;
        aclMemoryInfos[ACLArgs::ACL_DST]->set_quantization_info(
            getDstQuantizationInfo(fqInputScale, fqInputShift, dstPrecision));
    }

    return arm_compute::NEConvolutionLayer::validate(aclMemoryInfos[ACLArgs::ACL_SRC_0].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_WEI].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_BIAS].get(),
                                                     aclMemoryInfos[ACLArgs::ACL_DST].get(),
                                                     padStrideInfo,
                                                     weightsInfo,
                                                     dilation,
                                                     activationLayerInfo,
                                                     false,  // enable fast math
                                                     1);     // num_groups
}

ACLFunction ACLConvolutionExecutor::configureFunction(const ACLTensors& aclMemoryTensors) {
#if defined(OPENVINO_ARCH_ARM64)
    if (useOc1Jit) {
        return std::make_unique<Oc1DotConvolution>(aclMemoryTensors, padStrideInfo, cpuParallel);
    }
#endif
    auto neConv = std::make_unique<arm_compute::NEConvolutionLayer>();

    neConv->configure(aclMemoryTensors[ACLArgs::ACL_SRC_0].get(),
                      aclMemoryTensors[ACLArgs::ACL_WEI].get(),
                      aclMemoryTensors[ACLArgs::ACL_BIAS].get(),
                      aclMemoryTensors[ACLArgs::ACL_DST].get(),
                      padStrideInfo,
                      weightsInfo,
                      dilation,
                      activationLayerInfo,
                      false,  // enable fast math
                      1);     // num_groups
    return neConv;
}

}  // namespace ov::intel_cpu
