// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_postops_composer.h"

#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_attr.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_types.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/executors/common/common_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

DnnlPostOpsComposer::DnnlPostOpsComposer(const PostOps& postOps,
                                         const dnnl::engine& engine,
                                         const VectorDims& outputDims,
                                         const size_t indexOfOutputChannelDim,
                                         const bool isInt8,
                                         const int weiScaleMaskPerChannel,
                                         const MemoryArgs& memory,
                                         const dnnl::memory::data_type outDataType)
    : engine(engine),
      postOps(postOps),
      outputDims(outputDims),
      idxOC(indexOfOutputChannelDim),
      isINT8(isInt8),
      weightScaleMaskPerChannel(weiScaleMaskPerChannel),
      outDataType(outDataType) {
    OPENVINO_ASSERT(idxOC < outputDims.size());
    OC = outputDims[idxOC];
    dimsPerOC = dimsPerTensor = VectorDims(outputDims.size(), 1);
    dimsPerOC[idxOC] = OC;

    const auto& DQScales = getDeQuantizedScales(memory);
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
    // @todo does not look good to set scratchpad mode here
    // but the reason is that oneDNN's primitive_attr structure is basically huge config structure
    // which hold everything
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
        return dnnl::algorithm::eltwise_linear;
    }

    return dnnl::algorithm::undef;
}

bool DnnlPostOpsComposer::appendAttrPostOps(const ActivationPostOp& postOp, bool isLastPostOp, bool allowBinary) {
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
    case ScaleShiftPostOp::Type::powerstatic:
        if (scales[0] != 1.0f && shifts[0] != 0.0f) {
            return appendLinear(scales, shifts, isLastPostOp, allowBinary);
        } else if (scales[0] != 1.0f) {  // Multiply if has scales
            return appendScale(scales, isLastPostOp, allowBinary);
        } else if (shifts[0] != 0.0f) {  // Add only if has shifts
            return appendShift(shifts, allowBinary);
        }
        break;
    case ScaleShiftPostOp::Type::prelu:
        if (!allowBinary) {
            return false;
        }
        appendBinary(dnnl::algorithm::binary_prelu, scales);
        break;
    default:
        OPENVINO_THROW(postOp.type(), " as post operation is not supported");
    }

    return true;
}

static float roundHalfToEven(float f) {
    const float RHAFZ = std::round(f);  // r is round-half-away-from-zero
    const float d = RHAFZ - f;          // f + d -> RHAFZ
    if ((d != 0.5f) && (d != -0.5f)) {
        return RHAFZ;
    }

    // already even +/-1.5 -> +/-2
    if (std::fmod(RHAFZ, 2.0f) == 0.0f) {
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

    OPENVINO_ASSERT(inputScale.size() == 1 || inputScale.size() == OC);
    OPENVINO_ASSERT(inputShift.size() == 1 || inputShift.size() == OC);
    OPENVINO_ASSERT(cropLow.size() == 1 || cropLow.size() == OC);
    OPENVINO_ASSERT(cropHigh.size() == 1 || cropHigh.size() == OC);
    OPENVINO_ASSERT(outputScale.size() == 1 || outputScale.size() == OC);
    OPENVINO_ASSERT(outputShift.size() == 1 || outputShift.size() == OC);

    // WA: a per-Tensor input shift may little drift away randomly
    //     from it's orginal value when FQ was fused with any
    //     preceding per-channel multiply and create a false
    //     per-channel input shift, this threshold was chosen carefully
    //     to recorver the per-Tensor nature w/o mistaking a real
    //     per-channel FQ.
    if (isPerTensor(inputShift, inputShift[0], 0.00005f)) {
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

    if (f.osc.size() == 1 && f.osc[0] == 1.0f && f.osh.size() == 1 && f.osh[0] == std::trunc(f.osh[0])) {
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
    if (f.ish.size() == 1 && f.clo.size() == 1 && f.chi.size() == 1) {
        auto range = (f.chi[0] - f.clo[0]);
        if (abs(f.ish[0]) < range * 0.00001f) {
            f.ish[0] = 0.0f;
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
        if (outDataType == dnnl::memory::data_type::u8 && f.clo[0] <= 0.0f && f.chi[0] >= 255.0f) {
            skipRoundClipOutputLinear = true;
        }
        if (outDataType == dnnl::memory::data_type::s8 && f.clo[0] <= -128.0f && f.chi[0] >= 127.0f) {
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
    if (wei_scale_mask == 0 && wei_scale_values[0] == 1.0f) {
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
    if (dst_scale_val == 1.0f) {
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
        OPENVINO_ASSERT(data.size() == OC);
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

void DnnlPostOpsComposer::appendRoundHTE() {
    appendEltwise(dnnl::algorithm::eltwise_round_half_to_even, 0, 0);
}

bool DnnlPostOpsComposer::appendScale(const std::vector<float>& scale, bool isLastPostOp, bool allowBinary) {
    OPENVINO_ASSERT(scale.size() == OC || scale.size() == 1);

    bool fuseIntoWeiScale = false;
    // Use dest scale when last post-ops is per-tensor quantization.
    if ((isINT8 && isLastPostOp && scale.size() == 1)) {
        dst_scale_val = 1.0f / scale[0];
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
        if (scale.size() == 1 && ops.len() == 1) {
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
        if (shift[0] != 0.0f) {
            appendEltwise(dnnl::algorithm::eltwise_linear, 1.0f, shift[0]);
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
    if (scale.size() == 1 && shift.size() == 1) {
        if (shift[0] == 0.0f) {
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
    if (low.size() == 1 && high.size() == 1) {
        appendEltwise(dnnl::algorithm::eltwise_clip, low[0], high[0]);
    } else if (low.size() == 1) {
        OPENVINO_ASSERT(high.size() == OC);
        appendEltwise(dnnl::algorithm::eltwise_clip, low[0], std::numeric_limits<float>::max());
        if (high.size() > 0) {
            appendBinary(dnnl::algorithm::binary_min, high);
        }
    } else if (high.size() == 1) {
        OPENVINO_ASSERT(low.size() == OC);
        appendEltwise(dnnl::algorithm::eltwise_clip, -std::numeric_limits<float>::max(), high[0]);
        if (low.size() > 0) {
            appendBinary(dnnl::algorithm::binary_max, low);
        }
    } else {
        if (low.size() > 0) {
            OPENVINO_ASSERT(low.size() == OC);
            appendBinary(dnnl::algorithm::binary_max, low);
        }
        if (high.size() > 0) {
            OPENVINO_ASSERT(high.size() == OC);
            appendBinary(dnnl::algorithm::binary_min, high);
        }
    }
}

static MemoryPtr prepackDecompressionParams(const MemoryCPtr& paramsPtr,
                                            bool needTranspose,
                                            ov::element::Type dstPrc,
                                            const dnnl::engine& engine) {
    auto shape = paramsPtr->getShape().getStaticDims();
    if (shape.size() == 1 && shape[0] == 1) {
        shape.push_back(1);
    }

    if (shape.size() != 2 && shape.size() != 3) {
        OPENVINO_THROW("DnnlPostOpsComposer cannot prepack decompression params with invalid shape");
    }

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
    if (scaleDims[0] == 1 && scaleDims[1] == 1) {
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

DnnlPrimitiveAttrs DnnlPostOpsComposer::compose() {
    for (size_t i = 0; i < postOps.size(); ++i) {
        const auto& postOp = postOps[i];
        bool isLastPostOp = (i == (postOps.size() - 1));
        // @todo replace dynamic cast with an interface for appending to DNNL postops
        if (const auto activation = std::dynamic_pointer_cast<ActivationPostOp>(postOp)) {
            appendAttrPostOps(*activation, isLastPostOp);
            continue;
        }

        if (const auto ss = std::dynamic_pointer_cast<ScaleShiftPostOp>(postOp)) {
            appendAttrPostOps(*ss, isLastPostOp);
            continue;
        }

        if (const auto fq = std::dynamic_pointer_cast<FakeQuantizePostOp>(postOp)) {
            // drop rounding one special residual pattern
            // TODO: validate this unsafe optimization
            auto doRounding = [&]() {
                return true;
            };

            auto round = i == 0 ? doRounding() : true;
            appendAttrPostOps(*fq, isLastPostOp, round);
            continue;
        }

        OPENVINO_THROW("Fusing of operation to postOp is not implemented");
    }

    attr.set_post_ops(ops);

    for (const auto& args : cpuArgs) {
        dnnlArgs[args.first] = args.second->getPrimitive();
    }

    return {attr, dnnlArgs, cpuArgs};
}

}  // namespace ov::intel_cpu
