// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_postops_composer.h"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <common/c_types_map.hpp>
#include <common/primitive_attr.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_shape.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/dnnl/dnnl_post_op_data.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "utils/cpp/to_underlying.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

DnnlPostOpsComposer::DnnlPostOpsComposer(const PostOps& postOps,
                                         const dnnl::engine& engine,
                                         const VectorDims& outputDims,
                                         const size_t indexOfOutputChannelDim,
                                         const bool isInt8,
                                         const int weiScaleMaskPerChannel,
                                         const MemoryArgs& memory,
                                         const dnnl::memory::data_type outDataType,
                                         const std::vector<float>& legacyDqScales,
                                         PostOpsMode postOpsMode,
                                         bool useLegacyZeroPoints,
                                         dnnl::post_ops ops)
    : engine(engine),
      postOps(postOps),
      outputDims(outputDims),
      idxOC(indexOfOutputChannelDim),
      isINT8(isInt8),
      weightScaleMaskPerChannel(weiScaleMaskPerChannel),
      outDataType(outDataType),
      postOpsMode(postOpsMode),
      useLegacyZeroPoints(useLegacyZeroPoints),
      ops(std::move(ops)) {
    OPENVINO_ASSERT(idxOC < outputDims.size());
    OC = outputDims[idxOC];
    dimsPerOC = dimsPerTensor = VectorDims(outputDims.size(), 1);
    dimsPerOC[idxOC] = OC;

    const auto& DQScales = !legacyDqScales.empty() ? legacyDqScales : getDeQuantizedScales(memory);
    // generalise dq scales, so extra logic is necessary here.
    if (isINT8) {
        wei_scale_values = DQScales.empty() ? std::vector<float>{1.0} : DQScales;
        wei_scale_mask = wei_scale_values.size() > 1 ? weiScaleMaskPerChannel : 0;
        dst_scale_val = 1.0;

        // set the DQscale into attr weight scale before appending any post-ops.
        updateWeiScales();
        // If having the bias, attr weight scale can't be updated for further ops-ops optimization.
        // ONEDNN 3.x quantization for scheme: QuantizedInput * QuantizedWeight * DQScale + Bias.
        const bool hasBias = !memory.at(ARG_BIAS)->getDesc().empty();
        weightScaleAvailable = !hasBias;
    } else if (!DQScales.empty()) {
        // DQ scale is fused but swiching back to non-INT8 for execution in some cases.
        DEBUG_LOG("Set DQ scales for None-INT8, scale size ", DQScales.size());
        appendScale(DQScales, false, true);
    }

    if (useLegacyZeroPoints) {
        appendZeroPointsLegacy(memory);
    } else {
        appendZeroPoints(memory);
    }

    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
}

static dnnl::algorithm convertToOneDnn(const ActivationPostOp::Type type) {
    switch (type) {
    case ActivationPostOp::Type::relu:
        return dnnl::algorithm::eltwise_relu;
    case ActivationPostOp::Type::tanh:
        return dnnl::algorithm::eltwise_tanh;
    case ActivationPostOp::Type::elu:
        return dnnl::algorithm::eltwise_elu;
    case ActivationPostOp::Type::square:
        return dnnl::algorithm::eltwise_square;
    case ActivationPostOp::Type::abs:
        return dnnl::algorithm::eltwise_abs;
    case ActivationPostOp::Type::sqrt:
        return dnnl::algorithm::eltwise_sqrt;
    case ActivationPostOp::Type::soft_relu:
        return dnnl::algorithm::eltwise_soft_relu;
    case ActivationPostOp::Type::logistic:
        return dnnl::algorithm::eltwise_logistic;
    case ActivationPostOp::Type::exp:
        return dnnl::algorithm::eltwise_exp;
    case ActivationPostOp::Type::gelu_erf:
        return dnnl::algorithm::eltwise_gelu_erf;
    case ActivationPostOp::Type::gelu_tanh:
        return dnnl::algorithm::eltwise_gelu_tanh;
    case ActivationPostOp::Type::clip:
        return dnnl::algorithm::eltwise_clip;
    case ActivationPostOp::Type::swish:
        return dnnl::algorithm::eltwise_swish;
    case ActivationPostOp::Type::hardswish:
        return dnnl::algorithm::eltwise_hardswish;
    case ActivationPostOp::Type::mish:
        return dnnl::algorithm::eltwise_mish;
    case ActivationPostOp::Type::hsigmoid:
        return dnnl::algorithm::eltwise_hsigmoid;
    case ActivationPostOp::Type::round_half_to_even:
        return dnnl::algorithm::eltwise_round_half_to_even;
    case ActivationPostOp::Type::round_half_away_from_zero:
        return dnnl::algorithm::eltwise_round_half_away_from_zero;
    case ActivationPostOp::Type::linear:
    case ActivationPostOp::Type::powerstatic:  // actually eltwise_pow + eltwise_linear
        return dnnl::algorithm::eltwise_linear;
    default:
        return dnnl::algorithm::undef;
    }

    return dnnl::algorithm::undef;
}

bool DnnlPostOpsComposer::appendAttrPostOps(const ActivationPostOp& postOp,
                                            bool isLastPostOp,
                                            [[maybe_unused]] bool allowBinary) {
    if (postOp.type() == ActivationPostOp::Type::powerstatic) {
        const auto& scale = postOp.beta();
        const auto& shift = postOp.gamma();
        if (scale != 1.0F && shift != 0.0F) {
            return appendLinear({scale}, {shift}, isLastPostOp, allowBinary);
        }

        if (scale != 1.0F) {  // Multiply if has scales
            return appendScale({scale}, isLastPostOp, allowBinary);
        }

        if (shift != 0.0F) {  // Add only if has shifts
            return appendShift({shift}, allowBinary);
        }

        return true;
    }

    if (postOp.type() == ActivationPostOp::Type::linear) {
        appendLinear({postOp.alpha()}, {postOp.beta()}, isLastPostOp);
    } else {
        const dnnl::algorithm alg = convertToOneDnn(postOp.type());
        appendEltwise(alg, postOp.alpha(), postOp.beta());
    }

    return true;
}

bool DnnlPostOpsComposer::appendAttrPostOps(const ScaleShiftPostOp& postOp, bool isLastPostOp, bool allowBinary) {
    const auto& shifts = postOp.shifts();
    const auto& scales = postOp.scales();

    switch (postOp.type()) {
    case ScaleShiftPostOp::Type::add:
    case ScaleShiftPostOp::Type::subtract:
        return appendShift(shifts, allowBinary);
    case ScaleShiftPostOp::Type::divide:
    case ScaleShiftPostOp::Type::multiply:
        return appendScale(scales, isLastPostOp, allowBinary);
    case ScaleShiftPostOp::Type::muladd:
        return appendLinear(scales, shifts, isLastPostOp, allowBinary);
    case ScaleShiftPostOp::Type::prelu:
        if (!allowBinary) {
            return false;
        }
        appendBinary(dnnl::algorithm::binary_prelu, scales);
        break;
    default:
        OPENVINO_THROW(to_underlying(postOp.type()), " as post operation is not supported");
    }

    return true;
}

static float roundHalfToEven(float f) {
    const float RHAFZ = std::round(f);  // r is round-half-away-from-zero
    const float d = RHAFZ - f;          // f + d -> RHAFZ
    if (none_of(d, 0.5F, -0.5F)) {
        return RHAFZ;
    }

    // already even +/-1.5 -> +/-2
    if (std::fmod(RHAFZ, 2.0F) == 0.0F) {
        return RHAFZ;
    }

    // +/-2.5 -> +/-3, but we need it to to +/-2
    // RHAFZ (f+d) goes the wrong way, should be (f-d)
    return f - d;
}

struct OptimizedFormula {
    std::vector<float> isc;
    std::vector<float> ish;
    std::vector<float> osc;
    std::vector<float> osh;
    std::vector<float> clo;
    std::vector<float> chi;

    void shrinkLength() {
        auto _do_shrink = [](std::vector<float>& v) {
            if (v.size() <= 1) {
                return;
            }
            auto ref = v[0];
            if (std::all_of(v.cbegin(), v.cend(), [&](float val) {
                    return val == ref;
                })) {
                v.resize(1);
            }
        };
        _do_shrink(isc);
        _do_shrink(ish);
        _do_shrink(clo);
        _do_shrink(chi);
        _do_shrink(osc);
        _do_shrink(osh);
    }
};

static OptimizedFormula updateOptimizedFormula(const FakeQuantizePostOp& postOp, bool do_rounding) {
    OptimizedFormula f;
    const auto& inputScale = postOp.inputScale();
    const auto& inputShift = postOp.inputShift();
    const auto& cropLow = postOp.cropLow();
    const auto& cropHigh = postOp.cropHigh();
    const auto& outputScale = postOp.outputScale();
    const auto& outputShift = postOp.outputShift();

    DEBUG_LOG("\t ---- Original formula ----");
    DEBUG_LOG("\t    cropLow =[", printable(cropLow), "]");
    DEBUG_LOG("\t   cropHigh =[", printable(cropHigh), "]");
    DEBUG_LOG("\t inputScale =[", printable(inputScale), "]");
    DEBUG_LOG("\t inputShift =[", printable(inputShift), "]");
    DEBUG_LOG("\t outputScale=[", printable(outputScale), "]");
    DEBUG_LOG("\t outputShift=[", printable(outputShift), "]");

    auto isPerTensor =
        [](const std::vector<float>& v, float ref, const float zero_thr = std::numeric_limits<float>::min()) {
            return std::all_of(v.cbegin(), v.cend(), [&](float val) {
                return abs(val - ref) < zero_thr;
            });
        };
    size_t OC = std::max({inputScale.size(),
                          inputShift.size(),
                          cropLow.size(),
                          cropHigh.size(),
                          outputScale.size(),
                          outputShift.size()});

    OPENVINO_ASSERT(any_of(inputScale.size(), 1U, OC));
    OPENVINO_ASSERT(any_of(inputShift.size(), 1U, OC));
    OPENVINO_ASSERT(any_of(cropLow.size(), 1U, OC));
    OPENVINO_ASSERT(any_of(cropHigh.size(), 1U, OC));
    OPENVINO_ASSERT(any_of(outputScale.size(), 1U, OC));
    OPENVINO_ASSERT(any_of(outputShift.size(), 1U, OC));

    // WA: a per-Tensor input shift may little drift away randomly
    //     from it's orginal value when FQ was fused with any
    //     preceding per-channel multiply and create a false
    //     per-channel input shift, this threshold was chosen carefully
    //     to recorver the per-Tensor nature w/o mistaking a real
    //     per-channel FQ.
    if (isPerTensor(inputShift, inputShift[0], 0.00005F)) {
        f.ish.resize(OC);
        for (auto& v : f.ish) {
            v = inputShift[0];
        }
    } else {
        f.ish = inputShift;
    }
    f.clo = cropLow;
    f.chi = cropHigh;
    f.isc = inputScale;
    f.osc = outputScale;
    f.osh = outputShift;

    if (f.clo.size() == 1) {
        f.clo.resize(OC, f.clo[0]);
    }
    if (f.chi.size() == 1) {
        f.chi.resize(OC, f.chi[0]);
    }
    if (f.isc.size() == 1) {
        f.isc.resize(OC, f.isc[0]);
    }
    if (f.ish.size() == 1) {
        f.ish.resize(OC, f.ish[0]);
    }

    for (size_t i = 0; i < OC; i++) {
        auto& clo = f.clo[i];
        auto& chi = f.chi[i];
        auto& isc = f.isc[i];
        auto& ish = f.ish[i];
        const auto& osc = f.osc[f.osc.size() == 1 ? 0 : i];
        const auto& osh = f.osh[f.osh.size() == 1 ? 0 : i];

        clo = roundHalfToEven(clo * isc + ish);
        chi = roundHalfToEven(chi * isc + ish);
        if (clo > chi) {
            std::swap(clo, chi);
        }

        if (!do_rounding) {
            // when no rounding is needed, outputScale/outputShift can be
            // merged with inputScale/inputShift with updated cropLow/cropHigh
            clo = clo * osc + osh;
            chi = chi * osc + osh;
            if (clo > chi) {
                std::swap(clo, chi);
            }

            //  crop(x*isc + ish, a, b)*osc + osh
            //  crop(x*isc*osc + ish*osc + osh, a', b')
            isc = isc * osc;
            ish = ish * osc + osh;
        }
    }

    if (!do_rounding) {
        f.osc.clear();
        f.osh.clear();
    }

    f.shrinkLength();

    if (f.osc.size() == 1 && f.osc[0] == 1.0F && f.osh.size() == 1 && f.osh[0] == std::trunc(f.osh[0])) {
        // if outputScale == 1.0f and outputShift is interger, it can be further optimized
        //   x = clip2(round(x * inputScale + ish),c2lo,c2hi)*osc + osh
        //     = clip2(round(x * inputScale + ish),c2lo,c2hi) + osh
        //     = clip2(round(x * inputScale + ish) + osh, c2lo+osh,c2hi+osh)
        //     = clip2(round(x * inputScale + ish + osh), c2lo+osh,c2hi+osh)
        for (auto& v : f.ish) {
            v += f.osh[0];
        }
        for (auto& v : f.clo) {
            v += f.osh[0];
        }
        for (auto& v : f.chi) {
            v += f.osh[0];
        }
        f.osc.clear();
        f.osh.clear();
    }

    // we can save an additional eltwise linear for negligible shift
    if (all_of(1U, f.ish.size(), f.clo.size(), f.chi.size())) {
        auto range = (f.chi[0] - f.clo[0]);
        if (abs(f.ish[0]) < range * 0.00001F) {
            f.ish[0] = 0.0F;
        }
    }

    return f;
}

bool DnnlPostOpsComposer::appendAttrPostOps(const FakeQuantizePostOp& postOp,
                                            bool isLastPostOp,
                                            bool doRounding,
                                            bool allowBinary) {
    DEBUG_LOG("isLastPostOp=",
              isLastPostOp,
              ", outDataType=",
              outDataType,
              ", allowBinary=",
              allowBinary,
              ", doRounding=",
              doRounding);

    const auto f = updateOptimizedFormula(postOp, doRounding);

    DEBUG_LOG("\t ---- Optimized formula ----");
    DEBUG_LOG("\t inputScale =[", printable(f.isc), "]");
    DEBUG_LOG("\t inputShift =[", printable(f.ish), "]");
    DEBUG_LOG("\t    cropLow =[", printable(f.clo), "]");
    DEBUG_LOG("\t   cropHigh =[", printable(f.chi), "]");
    DEBUG_LOG("\toutputScale =[", printable(f.osc), "]");
    DEBUG_LOG("\toutputShift =[", printable(f.osh), "]");

    // when FQ is last postOps and output data type is u8/s8
    // round & clip2 can be further optimized since saturation will be performed by oneDNN by default
    bool skipRoundClipOutputLinear = false;
    if (isLastPostOp && (postOp.levels() == 256) && f.clo.size() == 1 && f.chi.size() == 1 && f.osc.empty() &&
        f.osh.empty()) {
        if (outDataType == dnnl::memory::data_type::u8 && f.clo[0] <= 0.0F && f.chi[0] >= 255.0F) {
            skipRoundClipOutputLinear = true;
        }
        if (outDataType == dnnl::memory::data_type::s8 && f.clo[0] <= -128.0F && f.chi[0] >= 127.0F) {
            skipRoundClipOutputLinear = true;
        }
    }

    // return false before committing any change to DnnlPostOpsComposer
    if (!allowBinary) {
        if (f.ish.size() > 1) {
            return false;
        }
        if (!skipRoundClipOutputLinear) {
            if (f.clo.size() > 1 || f.chi.size() > 1) {
                return false;
            }
            if (f.osc.size() > 1 || f.osh.size() > 1) {
                return false;
            }
        }
    }

    if (!appendLinear(f.isc, f.ish, isLastPostOp && skipRoundClipOutputLinear, allowBinary)) {
        return false;
    }

    if (skipRoundClipOutputLinear) {
        return true;
    }

    if (doRounding) {
        appendRoundHTE();
    }
    appendClip(f.clo, f.chi);

    appendLinear(f.osc, f.osh, isLastPostOp, allowBinary);

    return true;
}

void DnnlPostOpsComposer::updateWeiScales() {
    if (wei_scale_mask == 0 && wei_scale_values[0] == 1.0F) {
        return;
    }

    DEBUG_LOG("Set weight scales mask ", "DNNL_ARG: ", DNNL_ARG_WEIGHTS, " mask: ", wei_scale_mask);
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_scale_mask);

    DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, Shape({wei_scale_values.size()}));
    auto mem = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(mem->getData(), wei_scale_values.data(), wei_scale_values.size() * sizeof(float));
    cpuArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = mem;
    dnnlArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = mem->getPrimitive();
}

void DnnlPostOpsComposer::updateDestScales() {
    if (dst_scale_val == 1.0F) {
        return;
    }

    DEBUG_LOG("Set dest scale mask ", "DNNL_ARG: ", DNNL_ARG_DST, " mask: ", 0);
    attr.set_scales_mask(DNNL_ARG_DST, 0);

    DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, Shape({1}));
    auto mem = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(mem->getData(), &dst_scale_val, sizeof(float));
    cpuArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = mem;
    dnnlArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = mem->getPrimitive();
}

void DnnlPostOpsComposer::appendBinary(const dnnl::algorithm alg, const std::vector<float>& data) {
    VectorDims* pdims = &dimsPerTensor;
    if (data.size() > 1) {
        OPENVINO_ASSERT(data.size() == OC, "data size: ", data.size(), " OC: ", OC);
        pdims = &dimsPerOC;
    }

    DEBUG_LOG("Append binary post op with algorithm: ", convert_to_c(alg), " Shape: ", Shape(*pdims));

    DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, Shape(*pdims));
    ops.append_binary(alg, memoryDesc.getDnnlDesc());
    // copy the data as args
    auto mem = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(mem->getData(), data.data(), data.size() * sizeof(float));

    cpuArgs[DNNL_ARG_ATTR_MULTIPLE_POST_OP(ops.len() - 1) | DNNL_ARG_SRC_1] = mem;
    dnnlArgs[DNNL_ARG_ATTR_MULTIPLE_POST_OP(ops.len() - 1) | DNNL_ARG_SRC_1] = mem->getPrimitive();
}

void DnnlPostOpsComposer::appendEltwise(const dnnl::algorithm alg, float alpha, float beta) {
    DEBUG_LOG("Append eltwise post op with algorithm: ", convert_to_c(alg), " alpha: ", alpha, " beta: ", beta);
    ops.append_eltwise(alg, alpha, beta);
}

void DnnlPostOpsComposer::appendSum(float scale, int32_t zeroPoint, ov::element::Type dataType) {
    DEBUG_LOG("Append sum post op with scale: ", scale, " zero point: ", zeroPoint, " data type: ", dataType);
    auto sum_data_type = DnnlExtensionUtils::ElementTypeToDataType(dataType);

    ops.append_sum(scale, zeroPoint, sum_data_type);
}

void DnnlPostOpsComposer::appendRoundHTE() {
    appendEltwise(dnnl::algorithm::eltwise_round_half_to_even, 0, 0);
}

bool DnnlPostOpsComposer::appendScale(const std::vector<float>& scale, bool isLastPostOp, bool allowBinary) {
    OPENVINO_ASSERT(any_of(scale.size(), OC, 1U));

    bool fuseIntoWeiScale = false;
    // Use dest scale when last post-ops is per-tensor quantization.
    if ((isINT8 && isLastPostOp && scale.size() == 1)) {
        dst_scale_val = 1.0F / scale[0];
        updateDestScales();
        return true;
    }
    if (weightScaleAvailable) {
        // oneDNN v3.* weight scale can also be used in the further optimization patterns.
        //  there are so many possible optimizations can be done, for example:
        //
        //  we can switch the existing postOps's order to take
        //  advantage of output scale if it's available:
        //     relu(x)*scale = relu(x*scale)
        //  or we can fuse it into previous one as long as they are
        //  compatible in shape
        //     x*A*s = x*(A*s)
        //  or even with add:
        //     (x*A + B)*s = x*(A*s) + (B*s)
        //  or we can combine these two tricks:
        //     relu(x*A)*s = relu(x*(A*s))
        //
        //  we cannot implement all of them, so we just add the one
        //  that we observed in real models.
        if ((ops.len() == 0)) {
            fuseIntoWeiScale = true;
        }

        // relu(x)*s = relu(x*s)
        // prelu(x)*s = prelu(x*s)
        if (ops.len() == 1) {
            auto& cur_op = ops.get()->entry_[0];
            if ((cur_op.kind == dnnl::impl::primitive_kind::eltwise && cur_op.eltwise.alg == dnnl_eltwise_relu) ||
                (cur_op.kind == dnnl::impl::primitive_kind::binary && cur_op.binary.alg == dnnl_binary_prelu)) {
                fuseIntoWeiScale = true;
            }
        }

        // (x + dst[:])*s = (x*s + s*dst[:])
        if (all_of(1, static_cast<int>(scale.size()), ops.len())) {
            auto& cur_op = ops.get()->entry_.back();
            if (cur_op.kind == dnnl::impl::primitive_kind::sum) {
                cur_op.sum.scale *= scale[0];
                fuseIntoWeiScale = true;
            }
        }
    }
    if (fuseIntoWeiScale) {
        if (scale.size() > 1) {
            if (wei_scale_mask == 0) {
                wei_scale_values.resize(scale.size(), wei_scale_values[0]);
            } else {
                OPENVINO_ASSERT(wei_scale_values.size() == OC);
            }

            for (Dim j = 0; j < OC; j++) {
                wei_scale_values[j] *= scale[j];
            }
        } else {
            for (float& wei_scale_value : wei_scale_values) {
                wei_scale_value *= scale[0];
            }
        }

        if (wei_scale_values.size() == 1) {
            wei_scale_mask = 0;
        } else {
            wei_scale_mask = weightScaleMaskPerChannel;
        }

        updateWeiScales();

        return true;
    }

    // final fallback
    if (scale.size() == 1) {
        appendEltwise(dnnl::algorithm::eltwise_linear, scale[0], 0);
    } else {
        // this check returns before committing any changes
        if (!allowBinary) {
            return false;
        }
        appendBinary(dnnl::algorithm::binary_mul, scale);
    }
    return true;
}

bool DnnlPostOpsComposer::appendShift(const std::vector<float>& shift, bool allowBinary) {
    if (shift.size() == 1) {
        if (shift[0] != 0.0F) {
            appendEltwise(dnnl::algorithm::eltwise_linear, 1.0F, shift[0]);
        }
    } else {
        if (!allowBinary) {
            return false;
        }
        appendBinary(dnnl::algorithm::binary_add, shift);
    }
    return true;
}

bool DnnlPostOpsComposer::appendLinear(const std::vector<float>& scale,
                                       const std::vector<float>& shift,
                                       bool isLastPostOp,
                                       bool allowBinary) {
    if (all_of(1U, scale.size(), shift.size())) {
        if (shift[0] == 0.0F) {
            return appendScale(scale, isLastPostOp, allowBinary);
        }
        appendEltwise(dnnl::algorithm::eltwise_linear, scale[0], shift[0]);

    } else {
        // return before committing any changes
        if (!allowBinary && shift.size() > 1) {
            return false;
        }

        if (!scale.empty()) {
            if (!appendScale(scale, isLastPostOp && shift.empty(), allowBinary)) {
                return false;
            }
        }
        if (!shift.empty()) {
            if (!appendShift(shift, allowBinary)) {
                return false;
            }
        }
    }
    return true;
}

void DnnlPostOpsComposer::appendClip(const std::vector<float>& low, const std::vector<float>& high) {
    if (all_of(1U, low.size(), high.size())) {
        appendEltwise(dnnl::algorithm::eltwise_clip, low[0], high[0]);
    } else if (low.size() == 1) {
        OPENVINO_ASSERT(high.size() == OC);
        appendEltwise(dnnl::algorithm::eltwise_clip, low[0], std::numeric_limits<float>::max());
        if (!high.empty()) {
            appendBinary(dnnl::algorithm::binary_min, high);
        }
    } else if (high.size() == 1) {
        OPENVINO_ASSERT(low.size() == OC);
        appendEltwise(dnnl::algorithm::eltwise_clip, -std::numeric_limits<float>::max(), high[0]);
        if (!low.empty()) {
            appendBinary(dnnl::algorithm::binary_max, low);
        }
    } else {
        if (!low.empty()) {
            OPENVINO_ASSERT(low.size() == OC);
            appendBinary(dnnl::algorithm::binary_max, low);
        }
        if (!high.empty()) {
            OPENVINO_ASSERT(high.size() == OC);
            appendBinary(dnnl::algorithm::binary_min, high);
        }
    }
}

void DnnlPostOpsComposer::appendDepthwiseConvolution(int inH,
                                                     int inW,
                                                     int kerH,
                                                     int kerW,
                                                     int strH,
                                                     int strW,
                                                     dnnl::memory::data_type inDataType) {
    DEBUG_LOG("Append DW convolution");
    ops.append_dw_conv(inH, inW, kerH, kerW, strH, strW, dnnl::memory::convert_to_c(inDataType));
}

void DnnlPostOpsComposer::appendZeroPoints(const MemoryArgs& memory) {
    if (const auto arg = memory.find(ARG_ATTR_ZERO_POINTS | ARG_SRC_3); arg != memory.end()) {
        const auto mem = arg->second;
        attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
    }
}

void DnnlPostOpsComposer::appendZeroPointsLegacy(const MemoryArgs& memory) {
    const auto mask = 1 << idxOC;  // through C dim

    auto numElements = [](const MemoryCPtr& mem) {
        return mem->getDesc().getShape().getElementsCount();
    };

    if (const auto arg = memory.find(ARG_ATTR_ZERO_POINTS | ARG_SRC); arg != memory.end()) {
        const auto mem = arg->second;
        attr.set_input_zero_points(numElements(mem), mask);
    }

    if (const auto arg = memory.find(ARG_ATTR_ZERO_POINTS | ARG_WEI); arg != memory.end()) {
        const auto mem = arg->second;
        attr.set_weights_zero_points(numElements(mem), mask);
    }

    if (const auto arg = memory.find(ARG_ATTR_ZERO_POINTS | ARG_DST); arg != memory.end()) {
        const auto mem = arg->second;
        attr.set_output_compensations(numElements(mem), mask);
    }
}

static MemoryPtr prepackDecompressionParams(const MemoryCPtr& paramsPtr,
                                            bool needTranspose,
                                            ov::element::Type dstPrc,
                                            const dnnl::engine& engine) {
    auto shape = paramsPtr->getShape().getStaticDims();
    if (all_of(1U, shape.size(), shape[0])) {
        shape.push_back(1);
    }

    OPENVINO_ASSERT(any_of(shape.size(), 2U, 3U),
                    "DnnlPostOpsComposer cannot prepack decompression params with invalid shape");

    // weights without batch: (OC, G)
    // weights with batch: (B, OC, G)
    const size_t OC = shape[shape.size() - 2];
    const size_t G = shape[shape.size() - 1];

    Shape dstShape = Shape({OC, G});

    DnnlBlockedMemoryDesc dstMemoryDesc(dstShape,
                                        DnnlExtensionUtils::ElementTypeToDataType(dstPrc),
                                        dnnl::memory::format_tag::io);
    auto dstMem = std::make_shared<Memory>(engine, dstMemoryDesc);
    auto srcFormat = needTranspose ? dnnl::memory::format_tag::oi : dnnl::memory::format_tag::io;

    DnnlBlockedMemoryDesc srcMemoryDesc(
        dstShape,
        DnnlExtensionUtils::ElementTypeToDataType(paramsPtr->getDescPtr()->getPrecision()),
        srcFormat);
    auto srcMem = std::make_shared<Memory>(engine, srcMemoryDesc, paramsPtr->getData());

    dstMem->load(*srcMem, true, false);
    return dstMem;
}

static dnnl::memory::dims getGroupDims(const VectorDims& weiDims, const VectorDims& scaleDims) {
    if (all_of(1U, scaleDims[0], scaleDims[1])) {
        return {};
    }

    int N = weiDims[weiDims.size() - 2];
    int K = weiDims[weiDims.size() - 1];
    dnnl::memory::dim groupN = N / scaleDims[0];
    dnnl::memory::dim groupK = K / scaleDims[1];

    return {groupK, groupN};
}

static int getMask(const VectorDims& weiDims, const dnnl::memory::dims& groupDims) {
    const int maskN = 1 << (weiDims.size() - 1);
    const int maskK = 1 << (weiDims.size() - 2);
    int N = weiDims[weiDims.size() - 2];
    int K = weiDims[weiDims.size() - 1];
    int mask = 0;
    if (!groupDims.empty() && groupDims[1] != N) {
        mask += maskN;
    }
    if (!groupDims.empty() && groupDims[0] != K) {
        mask += maskK;
    }

    return mask;
}

void DnnlPostOpsComposer::appendDecompressionScales(const MemoryCPtr& scales_ptr,
                                                    bool needTranspose,
                                                    ov::element::Type dstPrecision,
                                                    const VectorDims& weiDims) {
    if (scales_ptr == nullptr) {
        return;
    }

    auto scaleMem = prepackDecompressionParams(scales_ptr, needTranspose, dstPrecision, engine);
    auto groupDims = getGroupDims(weiDims, scaleMem->getStaticDims());
    auto mask = getMask(weiDims, groupDims);

    attr.set_scales(DNNL_ARG_WEIGHTS, mask, groupDims, DnnlExtensionUtils::ElementTypeToDataType(dstPrecision));
    cpuArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = std::move(scaleMem);
    dnnlArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
        cpuArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS]->getPrimitive();
}

void DnnlPostOpsComposer::appendDecompressionZeroPoints(const MemoryCPtr& zero_points_ptr,
                                                        bool needTranspose,
                                                        ov::element::Type dstPrecision,
                                                        const VectorDims& weiDims) {
    if (zero_points_ptr == nullptr) {
        return;
    }

    auto zeroPointsMem = prepackDecompressionParams(zero_points_ptr, needTranspose, dstPrecision, engine);
    auto groupDims = getGroupDims(weiDims, zeroPointsMem->getStaticDims());
    auto mask = getMask(weiDims, groupDims);

    attr.set_zero_points(DNNL_ARG_WEIGHTS, mask, groupDims, DnnlExtensionUtils::ElementTypeToDataType(dstPrecision));
    cpuArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = zeroPointsMem;
    dnnlArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = zeroPointsMem->getPrimitive();
}

void DnnlPostOpsComposer::appendDecompressionScalesLegacy(const MemoryCPtr& scales_ptr,
                                                          bool needTranspose,
                                                          ov::element::Type dstPrecision) {
    if (scales_ptr == nullptr) {
        return;
    }

    auto scalesMem = prepackDecompressionParams(scales_ptr, needTranspose, dstPrecision, engine);
    attr.set_scales_dims(DNNL_ARG_WEIGHTS,
                         DnnlExtensionUtils::convertToDnnlDims(scalesMem->getStaticDims()),
                         DnnlExtensionUtils::ElementTypeToDataType(dstPrecision));
    cpuArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = std::move(scalesMem);
    dnnlArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
        cpuArgs[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS]->getPrimitive();
}

void DnnlPostOpsComposer::appendDecompressionZeroPointsLegacy(const MemoryCPtr& zero_points_ptr,
                                                              bool needTranspose,
                                                              ov::element::Type dstPrecision) {
    if (zero_points_ptr == nullptr) {
        return;
    }

    auto zeroPointsMem = prepackDecompressionParams(zero_points_ptr, needTranspose, dstPrecision, engine);
    attr.set_zero_points_dims(DNNL_ARG_WEIGHTS,
                              DnnlExtensionUtils::convertToDnnlDims(zeroPointsMem->getStaticDims()),
                              DnnlExtensionUtils::ElementTypeToDataType(dstPrecision));
    cpuArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = zeroPointsMem;
    dnnlArgs[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = zeroPointsMem->getPrimitive();
}

void DnnlPostOpsComposer::setDynamicQuantizationParams(uint64_t groupSize) {
    attr.set_src_dyn_quant_params(groupSize);
}

void DnnlPostOpsComposer::appendAttrPostOpsLegacy(const ActivationPostOp& postOp) {
    // powerstatic is not really an activation function but have similar semantics:
    // d = s^alpha + s*beta + gamma
    if (postOp.type() == ActivationPostOp::Type::powerstatic) {
        ops.append_eltwise(dnnl::algorithm::eltwise_linear, postOp.beta(), postOp.gamma());
        if (postOp.alpha() != 1.0F) {
            ops.append_eltwise(dnnl::algorithm::eltwise_pow, 1.0F, postOp.alpha());
        }
        return;
    }
    // for the rest of activation functions 'alpha' is usually a scale and 'beta' is a shift
    ops.append_eltwise(convertToOneDnn(postOp.type()), postOp.alpha(), postOp.beta());
}

void DnnlPostOpsComposer::appendAttrPostOpsLegacy(const ScaleShiftPostOp& postOp) {
    size_t channelSize = OC;

    size_t depthwiseDataSize = 2 * channelSize;
    std::vector<float> depthwiseData;
    const auto& scales = postOp.scales();
    const auto& shifts = postOp.shifts();

    depthwiseData.insert(depthwiseData.end(), scales.begin(), scales.end());
    if (scales.size() == 1) {
        depthwiseData.resize(channelSize, depthwiseData.back());
    } else if (scales.size() != channelSize) {
        OPENVINO_THROW("failed due to scales data size inconsistency");
    }
    depthwiseData.insert(depthwiseData.end(), shifts.begin(), shifts.end());
    if (shifts.empty()) {
        // in case of Prelu algorithm scales data is always empty
        depthwiseData.resize(2 * channelSize, 0);
    } else if (shifts.size() == 1) {
        depthwiseData.resize(2 * channelSize, depthwiseData.back());
    } else if (shifts.size() != channelSize) {
        OPENVINO_THROW("failed due to shifts data size inconsistency");
    }

    // always align for legacy scale/shift post ops
    constexpr int bufferAlignment = 16;
    int bufferPaddingSize = rnd_up(channelSize, bufferAlignment) - channelSize;
    depthwiseData.resize(depthwiseDataSize + bufferPaddingSize, 0);

    std::array<size_t, 2> offsets = {0};
    offsets[1] = offsets[0] + channelSize;

    // @todo legacy depthwise post ops are kept for now for performance reasons
    switch (postOp.type()) {
    case ScaleShiftPostOp::Type::add:
    case ScaleShiftPostOp::Type::subtract:
    case ScaleShiftPostOp::Type::multiply:
    case ScaleShiftPostOp::Type::divide:
    case ScaleShiftPostOp::Type::muladd:
        ops.append_depthwise(dnnl::algorithm::depthwise_scale_shift, offsets);
        break;
    case ScaleShiftPostOp::Type::prelu:
        ops.append_depthwise(dnnl::algorithm::depthwise_prelu, offsets);
        break;
    default:
        OPENVINO_THROW("as post operation is not supported");
    }

    DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, {depthwiseData.size()});
    auto memory = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(memory->getData(), depthwiseData.data(), depthwiseData.size() * sizeof(float));

    cpuArgs[DNNL_ARG_ATTR_MULTIPLE_POST_OP(ops.len() - 1) | DNNL_ARG_SRC_1] = memory;
}

void DnnlPostOpsComposer::appendAttrPostOpsLegacy(const FakeQuantizePostOp& postOp) {
    // try to map fakeQuantizeNode using output scale & eltwise first
    // if failed, fallback to append_quantization()

    // oneDNN quantization_injectors assumes that quantization data memory is always aligned on 16
    // by length of AVX512 vector register which is also enough for AVX2 and SSE42 implementations.
    // Otherwise it can lead to buffer over-read and performance penalties due to denormals.
    const size_t bufferAlignment = 16;

    if (postOp.type() == FakeQuantizePostOp::Type::binarization) {
        const auto realAxisSize = OC;
        const auto axisPaddedSize = rnd_up(realAxisSize, bufferAlignment);

        std::vector<float> binarizationThresholds;
        std::vector<uint32_t> binarizationOutputMask;

        binarizationThresholds.resize(axisPaddedSize, 0);
        binarizationOutputMask.resize(axisPaddedSize, 0);

        if (postOp.isInputLowBroadcast()) {
            std::fill(binarizationThresholds.begin() + 1,
                      binarizationThresholds.begin() + realAxisSize,
                      binarizationThresholds[0]);
            std::fill(binarizationThresholds.begin() + realAxisSize, binarizationThresholds.end(), 0.F);
        }
        if (postOp.isOutputHighBroadcast()) {
            std::fill(binarizationOutputMask.begin() + 1,
                      binarizationOutputMask.begin() + realAxisSize,
                      binarizationOutputMask[0]);
            std::fill(binarizationThresholds.begin() + realAxisSize, binarizationThresholds.end(), 0.F);
        }

        ops.append_binarization(dnnl::algorithm::binarization_depthwise,
                                reinterpret_cast<const float*>(binarizationThresholds.data()),
                                reinterpret_cast<const float*>(binarizationOutputMask.data()));
        return;
    }

    dnnl::algorithm alg = postOp.type() == FakeQuantizePostOp::Type::quantization_only
                              ? dnnl::algorithm::quantization_quantize
                              : dnnl::algorithm::quantization_quantize_dequantize;

    const auto& cropLow = postOp.cropLow();
    const auto& cropHigh = postOp.cropHigh();
    const auto& inputScale = postOp.inputScale();
    const auto& inputShift = postOp.inputShift();
    const auto& outputScale = postOp.outputScale();
    const auto& outputShift = postOp.outputShift();

    const size_t cropLowSize = cropLow.size();
    const size_t cropHighSize = cropHigh.size();
    const size_t inputScaleSize = inputScale.size();
    const size_t inputShiftSize = inputShift.size();
    const size_t outputScaleSize = outputScale.size();
    const size_t outputShiftSize = outputShift.size();

    std::array<bool, 6> per_channel = {cropLowSize > 1,
                                       cropHighSize > 1,
                                       inputScaleSize > 1,
                                       inputShiftSize > 1,
                                       outputScaleSize > 1,
                                       outputShiftSize > 1};

    std::array<bool, 6> all_default = {false};
    all_default[0] = std::all_of(cropLow.cbegin(), cropLow.cend(), [](float val) {
        return val == 0.F;
    });
    all_default[1] = std::all_of(cropHigh.cbegin(), cropHigh.cend(), [](float val) {
        return val == 0.F;
    });
    all_default[2] = std::all_of(inputScale.cbegin(), inputScale.cend(), [](float val) {
        return val == 1.F;
    });
    all_default[3] = std::all_of(inputShift.cbegin(), inputShift.cend(), [](float val) {
        return val == 0.F;
    });
    all_default[4] = std::all_of(outputScale.cbegin(), outputScale.cend(), [](float val) {
        return val == 1.F;
    });
    all_default[5] = std::all_of(outputShift.cbegin(), outputShift.cend(), [](float val) {
        return val == 0.F;
    });

    std::array<size_t, 6> offsets = {0};
    offsets[1] = offsets[0] + cropLowSize;
    offsets[2] = offsets[1] + cropHighSize;
    offsets[3] = offsets[2] + inputScaleSize;
    offsets[4] = offsets[3] + inputShiftSize;
    offsets[5] = offsets[4] + outputScaleSize;

    std::vector<float> quantizationData;
    quantizationData.insert(quantizationData.end(), cropLow.begin(), cropLow.end());
    quantizationData.insert(quantizationData.end(), cropHigh.begin(), cropHigh.end());
    quantizationData.insert(quantizationData.end(), inputScale.begin(), inputScale.end());
    quantizationData.insert(quantizationData.end(), inputShift.begin(), inputShift.end());
    quantizationData.insert(quantizationData.end(), outputScale.begin(), outputScale.end());
    quantizationData.insert(quantizationData.end(), outputShift.begin(), outputShift.end());

    DnnlBlockedMemoryDesc memoryDesc(ov::element::f32, {quantizationData.size()});
    auto memory = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(memory->getData(), quantizationData.data(), quantizationData.size() * sizeof(float));
    ops.append_quantization(alg, per_channel, all_default, offsets);

    cpuArgs[DNNL_ARG_ATTR_MULTIPLE_POST_OP(ops.len() - 1) | DNNL_ARG_SRC_1] = memory;
}

DnnlPrimitiveAttrs DnnlPostOpsComposer::compose() {
    DEBUG_LOG("Composing dnnl postops");
    for (size_t i = 0; i < postOps.size(); ++i) {
        const auto& postOp = postOps[i];
        bool isLastPostOp = (i == (postOps.size() - 1));

        if (const auto* const activation = std::any_cast<ActivationPostOp>(&postOp)) {
            switch (postOpsMode) {
            case PostOpsMode::Original:
                appendAttrPostOps(*activation, isLastPostOp, true);
                break;
            case PostOpsMode::Legacy:
                // legacy depthwise post ops often outperform binary post ops
                // first try to make do with original post ops without binary
                if (appendAttrPostOps(*activation, isLastPostOp, false)) {
                    DEBUG_LOG("Append as original post op without binary");
                    continue;
                }
                // fallback to legacy if failed
                appendAttrPostOpsLegacy(*activation);
                break;
            case PostOpsMode::ForcedLegacy:
                appendAttrPostOpsLegacy(*activation);
                break;
            }

            continue;
        }

        if (const auto* const ss = std::any_cast<ScaleShiftPostOp>(&postOp)) {
            switch (postOpsMode) {
            case PostOpsMode::Original:
                appendAttrPostOps(*ss, isLastPostOp, true);
                break;
            case PostOpsMode::Legacy:
                // legacy depthwise post ops often outperform binary post ops
                // first try to make do with original post ops without binary
                if (appendAttrPostOps(*ss, isLastPostOp, false)) {
                    DEBUG_LOG("Append as original post op without binary");
                    continue;
                }
                // fallback to legacy if failed
                appendAttrPostOpsLegacy(*ss);
                break;
            case PostOpsMode::ForcedLegacy:
                appendAttrPostOpsLegacy(*ss);
                break;
            }
            continue;
        }

        if (const auto* const fq = std::any_cast<FakeQuantizePostOp>(&postOp)) {
            // drop rounding one special residual pattern
            // TODO: validate this unsafe optimization
            auto doRounding = [&]() {
                bool hasSubsequentSum = false;
                bool hasSubsequentFQ = false;
                for (size_t j = i + 1; j < postOps.size(); j++) {
                    const auto& nextNode = postOps[j];

                    if (typeid(SumPostOp) == nextNode.type()) {
                        hasSubsequentSum = true;
                    }

                    if (typeid(FakeQuantizePostOp) == nextNode.type()) {
                        hasSubsequentFQ = true;
                    }
                }

                return !(hasSubsequentSum && hasSubsequentFQ);
            };

            auto round = i == 0 ? doRounding() : true;
            switch (postOpsMode) {
            case PostOpsMode::Original:
                DEBUG_LOG("Append as original post op");
                appendAttrPostOps(*fq, isLastPostOp, round, true);
                break;
            case PostOpsMode::Legacy:
                // legacy depthwise post ops often outperform binary post ops
                // first try to make do with original post ops without binary
                if (appendAttrPostOps(*fq, isLastPostOp, round, false)) {
                    DEBUG_LOG("Append as original post op without binary");
                    continue;
                }
                DEBUG_LOG("Append as legacy post op");
                appendAttrPostOpsLegacy(*fq);
                break;
            case PostOpsMode::ForcedLegacy:
                DEBUG_LOG("Append as legacy post op");
                appendAttrPostOpsLegacy(*fq);
                break;
            }
            continue;
        }

        if (const auto* const sum = std::any_cast<SumPostOp>(&postOp)) {
            appendSum(sum->scale(), sum->zeroPoint(), sum->dataType());
            continue;
        }

        if (const auto* const conv = std::any_cast<DepthwiseConvolutionPostOp>(&postOp)) {
            appendDepthwiseConvolution(conv->ih(),
                                       conv->iw(),
                                       conv->kernel()[1],
                                       conv->kernel()[0],
                                       conv->strides()[1],
                                       conv->strides()[0],
                                       dnnl::memory::data_type::f32);
            continue;
        }

        OPENVINO_THROW("Fusing of operation to postOp is not implemented");
    }

    attr.set_post_ops(ops);

    for (const auto& args : cpuArgs) {
        dnnlArgs[args.first] = args.second->getPrimitive();
    }

    return {attr, dnnlArgs, cpuArgs, useLegacyZeroPoints};
}

}  // namespace ov::intel_cpu
