// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_postops_composer.h"
#include <oneapi/dnnl/dnnl_types.h>

#include <common/primitive_attr.hpp>
#include "utils/debug_capabilities.h"

namespace ov {
namespace intel_cpu {

DnnlPostOpsComposer::DnnlPostOpsComposer(const dnnl::engine& engine,
                                        dnnl::primitive_attr& attr,
                                        dnnl::post_ops& ops,
                                        std::unordered_map<int, MemoryPtr>& args,
                                        const VectorDims& outputDims,
                                        int indexOfOutputChannelDim,
                                        bool isInt8,
                                        const int weiScaleMaskPerChannel,
                                        const std::vector<float>& DQScales,
                                        bool hasBias)
    : engine(engine),
      attr(attr),
      ops(ops),
      args(args),
      outputDims(outputDims),
      idxOC(indexOfOutputChannelDim),
      isINT8(isInt8),
      weightScaleMaskPerChannel(weiScaleMaskPerChannel) {
    IE_ASSERT(idxOC >= 0 && static_cast<size_t>(idxOC) < outputDims.size());
    OC = outputDims[idxOC];
    dimsPerOC = dimsPerTensor = VectorDims(outputDims.size(), 1);
    dimsPerOC[idxOC] = OC;

    if (isINT8) {
        wei_scale_values = DQScales.empty() ? std::vector<float>{1.0} : DQScales;
        wei_scale_mask = wei_scale_values.size() > 1 ? weiScaleMaskPerChannel : 0;
        dst_scale_val = 1.0;

        //set the DQscale into attr weight scale before appending any post-ops.
        updateWeiScales();
        //If having the bias, attr weight scale can't be updated for further ops-ops optimization.
        //ONEDNN 3.x quantization for scheme: QuantizedInput * QuantizedWeight * DQScale + Bias.
        weightScaleAvailable = !hasBias;
    } else if (!DQScales.empty()) {
        // DQ scale is fused but swiching back to non-INT8 for execution in some cases.
        DEBUG_LOG("Set DQ scales for None-INT8, scale size ", DQScales.size());
        appendScale(DQScales, false, true);
    }
}

void DnnlPostOpsComposer::updateWeiScales() {
    if (wei_scale_mask == 0 && wei_scale_values[0] == 1.0f)
        return;

    DEBUG_LOG("Set weight scales mask ", "DNNL_ARG: ", DNNL_ARG_WEIGHTS, " mask: ", wei_scale_mask);
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, wei_scale_mask);

    DnnlBlockedMemoryDesc memoryDesc(InferenceEngine::Precision::FP32, Shape({wei_scale_values.size()}));
    auto mem = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(mem->getData(), wei_scale_values.data(), wei_scale_values.size() * sizeof(float));
    args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = mem;
}

void DnnlPostOpsComposer::updateDestScales() {
    if (dst_scale_val == 1.0f)
        return;

    DEBUG_LOG("Set dest scale mask ", "DNNL_ARG: ", DNNL_ARG_DST, " mask: ", 0);
    attr.set_scales_mask(DNNL_ARG_DST, 0);

    DnnlBlockedMemoryDesc memoryDesc(InferenceEngine::Precision::FP32, Shape({1}));
    auto mem = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(mem->getData(), &dst_scale_val, sizeof(float));
    args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST] = mem;
}

void DnnlPostOpsComposer::appendBinary(const dnnl::algorithm alg, const std::vector<float>& data) {
    VectorDims* pdims = &dimsPerTensor;
    if (data.size() > 1) {
        IE_ASSERT(data.size() == OC);
        pdims = &dimsPerOC;
    }

    DEBUG_LOG("Append binary post op with algorithm: ", convert_to_c(alg));

    DnnlBlockedMemoryDesc memoryDesc(InferenceEngine::Precision::FP32, Shape(*pdims));
    ops.append_binary(alg, memoryDesc.getDnnlDesc());

    // copy the data as args
    auto mem = std::make_shared<Memory>(engine, memoryDesc);
    memcpy(mem->getData(), data.data(), data.size() * sizeof(float));
    args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(ops.len() - 1) | DNNL_ARG_SRC_1] = mem;
}

void DnnlPostOpsComposer::appendEltwise(const dnnl::algorithm alg, float alpha, float beta) {
    DEBUG_LOG("Append eltwise post op with algorithm: ", convert_to_c(alg));
    ops.append_eltwise(alg, alpha, beta);
}

void DnnlPostOpsComposer::appendRoundHTE() {
    appendEltwise(dnnl::algorithm::eltwise_round_half_to_even, 0, 0);
}

bool DnnlPostOpsComposer::appendScale(const std::vector<float>& scale, bool isLastPostOp, bool allowBinary) {
    IE_ASSERT(scale.size() == OC || scale.size() == 1);

    bool fuseIntoWeiScale = false;
    // Use dest scale when last post-ops is per-tensor quantization.
    if ((isINT8 && isLastPostOp && scale.size() == 1)) {
        dst_scale_val = 1.0 / scale[0];
        updateDestScales();
        return true;
    }
    if (weightScaleAvailable) {
        //oneDNN v3.* weight scale can also be used in the further optimization patterns.
        // there are so many possible optimizations can be done, for example:
        //
        // we can switch the existing postOps's order to take
        // advantage of output scale if it's available:
        //    relu(x)*scale = relu(x*scale)
        // or we can fuse it into previous one as long as they are
        // compatible in shape
        //    x*A*s = x*(A*s)
        // or even with add:
        //    (x*A + B)*s = x*(A*s) + (B*s)
        // or we can combine these two tricks:
        //    relu(x*A)*s = relu(x*(A*s))
        //
        // we cannot implement all of them, so we just add the one
        // that we observed in real models.
        if ((ops.len() == 0))
            fuseIntoWeiScale = true;

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
            if (wei_scale_mask == 0)
                wei_scale_values.resize(scale.size(), wei_scale_values[0]);
            else
                IE_ASSERT(wei_scale_values.size() == OC);

            for (Dim j = 0; j < OC; j++)
                wei_scale_values[j] *= scale[j];
        } else {
            for (size_t j = 0; j < wei_scale_values.size(); j++)
                wei_scale_values[j] *= scale[0];
        }

        if (wei_scale_values.size() == 1)
            wei_scale_mask = 0;
        else
            wei_scale_mask = weightScaleMaskPerChannel;
        updateWeiScales();
        return true;
    }

    // final fallback
    if (scale.size() == 1) {
        appendEltwise(dnnl::algorithm::eltwise_linear, scale[0], 0);
    } else {
        // this check returns before committing any changes
        if (!allowBinary)
            return false;
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
        if (!allowBinary)
            return false;
        appendBinary(dnnl::algorithm::binary_add, shift);
    }
    return true;
}

bool DnnlPostOpsComposer::appendLinear(const std::vector<float>& scale,
                                       const std::vector<float>& shift,
                                       bool isLastPostOp,
                                       bool allowBinary) {
    if (scale.size() == 1 && shift.size() == 1) {
        if (shift[0] == 0.0f)
            return appendScale(scale, isLastPostOp, allowBinary);
        else
            appendEltwise(dnnl::algorithm::eltwise_linear, scale[0], shift[0]);
    } else {
        // return before committing any changes
        if (!allowBinary && shift.size() > 1)
            return false;

        if (!scale.empty()) {
            if (!appendScale(scale, isLastPostOp && shift.empty(), allowBinary))
                return false;
        }
        if (!shift.empty()) {
            if (!appendShift(shift, allowBinary))
                return false;
        }
    }
    return true;
}

void DnnlPostOpsComposer::appendClip(const std::vector<float>& low, const std::vector<float>& high) {
    if (low.size() == 1 && high.size() == 1) {
        appendEltwise(dnnl::algorithm::eltwise_clip, low[0], high[0]);
    } else if (low.size() == 1) {
        IE_ASSERT(high.size() == OC);
        appendEltwise(dnnl::algorithm::eltwise_clip, low[0], std::numeric_limits<float>::max());
        if (high.size() > 0)
            appendBinary(dnnl::algorithm::binary_min, high);
    } else if (high.size() == 1) {
        IE_ASSERT(low.size() == OC);
        appendEltwise(dnnl::algorithm::eltwise_clip, -std::numeric_limits<float>::max(), high[0]);
        if (low.size() > 0)
            appendBinary(dnnl::algorithm::binary_max, low);
    } else {
        if (low.size() > 0) {
            IE_ASSERT(low.size() == OC);
            appendBinary(dnnl::algorithm::binary_max, low);
        }
        if (high.size() > 0) {
            IE_ASSERT(high.size() == OC);
            appendBinary(dnnl::algorithm::binary_min, high);
        }
    }
}

MemoryPtr DnnlPostOpsComposer::prepackDecompressionParams(const MemoryCPtr& params_ptr, bool needTranspose) {
    const auto shape = params_ptr->getShape().getStaticDims();
    MemoryPtr mem;

    auto params_data = static_cast<float*>(params_ptr->getData());

    if (needTranspose) {
        VectorDims dnnlShape = {shape[0], shape[1]};
        DnnlBlockedMemoryDesc memoryDesc(InferenceEngine::Precision::FP32, Shape(dnnlShape));
        mem = std::make_shared<Memory>(engine, memoryDesc);
        auto memory_buf = static_cast<float*>(mem->getData());

        // oi -> io
        for (size_t oc = 0; oc < dnnlShape[0]; oc++) {
            for (size_t ic = 0; ic < dnnlShape[1]; ic++) {
                memory_buf[ic * dnnlShape[0] + oc] = params_data[oc * dnnlShape[1] + ic];
            }
        }
    } else {
        VectorDims dnnlShape = {shape[shape.size() - 1], shape[0]};
        DnnlBlockedMemoryDesc memoryDesc(InferenceEngine::Precision::FP32, Shape(dnnlShape));
        mem = std::make_shared<Memory>(engine, memoryDesc);
        auto memory_buf = static_cast<float*>(mem->getData());
        const size_t elements_count = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

        // io -> io
        size_t dstIdx = 0;
        for (size_t oc = 0; oc < elements_count; oc++) {
            memory_buf[dstIdx] = params_data[oc];
            dstIdx++;
        }
    }

    return mem;
}

void DnnlPostOpsComposer::appendDecompressionScales(const MemoryCPtr& scales_ptr, bool needTranspose) {
    if (scales_ptr == nullptr)
        return;

    auto scalesMem = prepackDecompressionParams(scales_ptr, needTranspose);
    attr.set_scales_dims(DNNL_ARG_WEIGHTS, DnnlExtensionUtils::convertToDnnlDims(scalesMem->getStaticDims()));
    args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = scalesMem;
}

void DnnlPostOpsComposer::appendDecompressionZeroPoints(const MemoryCPtr& zero_points_ptr, bool needTranspose) {
    if (zero_points_ptr == nullptr)
        return;

    auto zeroPointsMem = prepackDecompressionParams(zero_points_ptr, needTranspose);
    attr.set_zero_points_dims(DNNL_ARG_WEIGHTS, DnnlExtensionUtils::convertToDnnlDims(zeroPointsMem->getStaticDims()));
    args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = zeroPointsMem;
}

}  // namespace intel_cpu
}  // namespace ov
