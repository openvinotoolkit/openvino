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
#include <cstring>
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

#if defined(OPENVINO_ARCH_ARM64) && defined(__APPLE__)
#    include <xbyak_aarch64/xbyak_aarch64.h>

#    include "utils/arm_isa_support.h"
#endif

namespace ov::intel_cpu {

#if defined(OPENVINO_ARCH_ARM64) && defined(__APPLE__)
namespace {

constexpr size_t oc16Channels = 16;
constexpr size_t oc16FilterElements = 27;
constexpr size_t oc16PackedElements = 28;

bool supportsU8Oc16Jit(const ConvAttrs& attrs,
                       const MemoryDesc& src,
                       const MemoryDesc& weights,
                       const MemoryDesc& dst) {
    if (!hasArmISASupport(ArmISA::DOTPROD) || attrs.isGrouped || !attrs.constantWeights || attrs.withBias ||
        !attrs.postOps.empty() || !attrs.dqScales.empty() || attrs.inputZeroPoint != 0 ||
        attrs.inputZeroPointsType != ZeroPointsType::None || src.getPrecision() != ov::element::u8 ||
        weights.getPrecision() != ov::element::i8 || dst.getPrecision() != ov::element::f32 ||
        !src.getShape().isStatic() || !weights.getShape().isStatic() || !dst.getShape().isStatic()) {
        return false;
    }

    const auto& input = src.getShape().getStaticDims();
    const auto& filter = weights.getShape().getStaticDims();
    const auto& output = dst.getShape().getStaticDims();
    return input.size() == 4 && output.size() == 4 && filter == VectorDims{oc16Channels, 3, 3, 3} && input[1] == 3 &&
           output[1] == oc16Channels && input[0] == output[0] && input[2] == output[2] && input[3] == output[3] &&
           output[3] >= 16 && output[2] * output[3] >= 1024 && attrs.stride == std::vector<size_t>{1, 1} &&
           attrs.dilation == std::vector<size_t>{0, 0} && attrs.paddingL == std::vector<ptrdiff_t>{1, 1} &&
           attrs.paddingR == std::vector<ptrdiff_t>{1, 1};
}

class U8Oc16DotKernel : public Xbyak_aarch64::CodeGenerator {
public:
    using Function = void (*)(const uint8_t*, const int8_t*, float*, size_t, const int32_t*);

    U8Oc16DotKernel() {
        using namespace Xbyak_aarch64;
        movi(v5.b, 128);
        const auto loop = L();
        for (size_t block = 0; block < 4; ++block) {
            ldr(QReg(16 + block), ptr(x4, static_cast<int32_t>(16 * block)));
        }
        for (size_t k = 0; k < oc16PackedElements / 4; ++k) {
            ldr(w5, ptr(x0, static_cast<int32_t>(4 * k)));
            dup(v0.s, w5);
            eor(v0.b, v0.b, v5.b);
            for (size_t block = 0; block < 4; ++block) {
                ldr(q1, ptr(x1, static_cast<int32_t>(64 * k + 16 * block)));
                sdot(VReg4S(16 + block), v1.b, v0.b);
            }
        }
        for (size_t block = 0; block < 4; ++block) {
            const auto acc = VReg4S(16 + block);
            scvtf(acc, acc);
            str(QReg(16 + block), ptr(x2, static_cast<int32_t>(16 * block)));
        }
        add(x0, x0, oc16PackedElements);
        add(x2, x2, oc16Channels * sizeof(float));
        subs(x3, x3, 1);
        b(NE, loop);
        ret();
        ready();
    }
};

class U8Oc16DotConvolution : public arm_compute::IFunction {
public:
    U8Oc16DotConvolution(const ACLTensors& tensors, std::shared_ptr<CpuParallel> parallel)
        : m_src(tensors[ACL_SRC_0].get()),
          m_weights(tensors[ACL_WEI].get()),
          m_dst(tensors[ACL_DST].get()),
          m_parallel(std::move(parallel)) {}

    void run() override {
        const auto* src = m_src->buffer();
        const auto* weights = reinterpret_cast<const int8_t*>(m_weights->buffer());
        auto* dst = reinterpret_cast<float*>(m_dst->buffer());
        const auto width = m_src->info()->dimension(1);
        const auto height = m_src->info()->dimension(2);
        const auto batch = m_src->info()->dimension(3);
        std::array<int8_t, oc16PackedElements * oc16Channels> packedWeights{};
        std::array<int32_t, oc16Channels> corrections{};
        // Pack four bytes per SDOT lane for four output channels at a time.
        for (size_t oc = 0; oc < oc16Channels; ++oc) {
            for (size_t k = 0; k < oc16FilterElements; ++k) {
                const auto weight = weights[oc * oc16FilterElements + k];
                packedWeights[(k / 4) * 64 + oc * 4 + k % 4] = weight;
                corrections[oc] += 128 * weight;
            }
        }
        const auto kernel = m_kernel.getCode<U8Oc16DotKernel::Function>();
        m_parallel->parallel_for(batch * height, [&](size_t row) {
            constexpr size_t tile = 64;
            std::array<uint8_t, tile * oc16PackedElements> patches;
            const auto y = row % height;
            const auto* image = src + (row / height) * width * height * 3;
            for (size_t x = 0; x < width; x += tile) {
                const auto count = std::min(tile, width - x);
                for (size_t dx = 0; dx < count; ++dx) {
                    auto* patch = patches.data() + dx * oc16PackedElements;
                    const auto column = x + dx;
                    for (size_t ky = 0; ky < 3; ++ky) {
                        const auto iy = y + ky - 1;
                        if (iy >= height) {
                            std::fill_n(patch + ky * 9, 9, 0);
                        } else {
                            const auto* line = image + iy * width * 3;
                            if (column > 0 && column + 1 < width) {
                                std::memcpy(patch + ky * 9, line + (column - 1) * 3, 9);
                            } else {
                                auto* patchRow = patch + ky * 9;
                                std::fill_n(patchRow, 9, 0);
                                const bool left = column == 0;
                                std::memcpy(patchRow + (left ? 3 : 0), line + (left ? 0 : (column - 1) * 3), 6);
                            }
                        }
                    }
                    patch[oc16FilterElements] = 0;
                }
                kernel(patches.data(),
                       packedWeights.data(),
                       dst + (row * width + x) * oc16Channels,
                       count,
                       corrections.data());
            }
        });
    }

private:
    arm_compute::Tensor* m_src;
    arm_compute::Tensor* m_weights;
    arm_compute::Tensor* m_dst;
    std::shared_ptr<CpuParallel> m_parallel;
    U8Oc16DotKernel m_kernel;
};

class Oc1DotKernel : public Xbyak_aarch64::CodeGenerator {
public:
    using Function = void (*)(const int8_t* const*, const int8_t*, float*, size_t, int32_t, float, float);

    Oc1DotKernel(size_t channels, bool unsignedInput) {
        using namespace Xbyak_aarch64;
        if (unsignedInput) {
            movi(v7.b, 128);
        }
        const auto loop = L();
        for (int i = 16; i < 20; ++i) {
            movi(VReg16B(i), 0);
        }
        for (size_t tap = 0; tap < 9; ++tap) {
            ldr(x5, ptr(x0, static_cast<int32_t>(tap * sizeof(void*))));
            for (size_t c = 0; c < channels;) {
                const auto bytes = channels - c >= 16 ? 16 : channels - c >= 8 ? 8 : 4;
                add(x6, x1, static_cast<uint32_t>(tap * channels + c));
                const auto input = ptr(x5, static_cast<int32_t>(c));
                if (bytes == 16) {
                    ldr(q2, input);
                    ldr(q3, ptr(x6));
                } else if (bytes == 8) {
                    ldr(d2, input);
                    ldr(d3, ptr(x6));
                } else {
                    ldr(s2, input);
                    ldr(s3, ptr(x6));
                }
                if (unsignedInput) {
                    eor(v2.b, v2.b, v7.b);
                }
                sdot(VReg4S(16 + tap % 4), v2.b, v3.b);
                c += bytes;
            }
        }
        add(v16.s, v16.s, v17.s);
        add(v18.s, v18.s, v19.s);
        add(v16.s, v16.s, v18.s);
        addv(s2, v16.s);
        fmov(w5, s2);
        add(w5, w5, w4);
        scvtf(s2, w5);
        fmul(s2, s2, s0);
        fadd(s2, s2, s1);
        str(s2, post_ptr(x2, sizeof(float)));
        add(x0, x0, 9 * sizeof(void*));
        subs(x3, x3, 1);
        b(NE, loop);
        ret();
        ready();
    }
};

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
        const auto kernel = m_kernel.getCode<Oc1DotKernel::Function>();
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
                kernel(taps.data(), weights, dst + row * output_w + x, count, correction, m_scale, bias);
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
    Oc1DotKernel m_kernel;
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

#if defined(OPENVINO_ARCH_ARM64) && defined(__APPLE__)
    useU8Oc16Jit = aclTensorAttrs.hasLayoutTypeNHWC && weiMemPtr->hasLayoutType(LayoutType::nspc) &&
                   dstMemPtr->hasLayoutType(LayoutType::nspc) &&
                   supportsU8Oc16Jit(attrs, *srcMemPtr, *weiMemPtr, *dstMemPtr);
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
#if defined(OPENVINO_ARCH_ARM64) && defined(__APPLE__)
    if (supportsU8Oc16Jit(config.attrs, *srcDesc, *weiDesc, *dstDesc)) {
        return true;
    }
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
    if (useOc1Jit || useU8Oc16Jit) {
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
#if defined(OPENVINO_ARCH_ARM64) && defined(__APPLE__)
    if (useU8Oc16Jit) {
        return std::make_unique<U8Oc16DotConvolution>(aclMemoryTensors, cpuParallel);
    }
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
