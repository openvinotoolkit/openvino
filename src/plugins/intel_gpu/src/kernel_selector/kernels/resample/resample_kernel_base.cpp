// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace {
int getAxisIndex(kernel_selector::InterpolateAxis axis) {
    switch (axis) {
    case kernel_selector::InterpolateAxis::BATCH:
        return 0;
    case kernel_selector::InterpolateAxis::FEATURE:
        return 1;
    case kernel_selector::InterpolateAxis::Z:
        return 2;
    case kernel_selector::InterpolateAxis::Y:
        return 3;
    case kernel_selector::InterpolateAxis::X:
        return 4;
    default:
        return 0;
    }
}
}  // namespace

namespace kernel_selector {

size_t ResampleKernelBase::GetFeatureBlockSize(const resample_params& params) const {
    const size_t max_size = 32;
    const size_t min_size = 4;
    size_t feature_block_size = 1;
    std::vector<size_t> preferred_sizes = { 32, 16, 8 };
    for (auto& s : preferred_sizes)
        if (params.outputs[0].Feature().v % s == 0)
            return s;
    if (params.outputs[0].Feature().v < max_size)
        return params.outputs[0].Feature().v;
    for (size_t f = 1; f <= params.outputs[0].Feature().v && f <= max_size; f++)
        if (params.outputs[0].Feature().v % f == 0)
            feature_block_size = f;
    return std::max(feature_block_size, min_size);
}

ResampleKernelBase::DispatchData ResampleKernelBase::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData dispatchData;
    auto in_layout = arg.inputs[0].GetLayout();
    auto out_layout = arg.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& out = arg.outputs[0];

    if (arg.resampleType == ResampleType::NEAREST_NEIGHBOR) {
        dispatchData.gws = { out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v };
        dims_by_gws = {{ Tensor::DataChannelName::X },
                       { Tensor::DataChannelName::Y, Tensor::DataChannelName::Z },
                       { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
    } else if ( (arg.resampleType == ResampleType::BILINEAR_INTERP || arg.resampleType == ResampleType::LINEAR_ONNX) &&
                out.Dimentions() <= 4 ) {
        dispatchData.gws = { Align(out.X().v, 32), out.Y().v, out.Batch().v };
        dims_by_gws = {{ Tensor::DataChannelName::X },
                       { Tensor::DataChannelName::Y },
                       { Tensor::DataChannelName::BATCH }};
    } else if (arg.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        dispatchData.gws = { out.X().v * out.Y().v, CeilDiv(out.Feature().v, GetFeatureBlockSize(arg)), out.Batch().v * out.Z().v };
        dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                       { Tensor::DataChannelName::FEATURE },
                       { Tensor::DataChannelName::Z, Tensor::DataChannelName::BATCH }};
    } else {
        dispatchData.gws = { out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v };
        dims_by_gws = {{ Tensor::DataChannelName::X },
                       { Tensor::DataChannelName::Y, Tensor::DataChannelName::Z },
                       { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo, in_layout, out_layout, dims_by_gws);

    if ((arg.resampleType == ResampleType::BILINEAR_INTERP || arg.resampleType == ResampleType::LINEAR_ONNX) &&
        out.Dimentions() <= 4) {
        dispatchData.lws[0] = 32;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

bool ResampleKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::RESAMPLE) {
        return false;
    }

    const resample_params& params = static_cast<const resample_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs.size() == 0) {
        return false;
    }

    const auto& input = params.inputs[0];
    if ((input.GetDType() == Datatype::UINT8 || input.GetDType() == Datatype::INT8) &&
        params.resampleType != ResampleType::NEAREST_NEIGHBOR &&
        params.resampleType != ResampleType::CAFFE_BILINEAR_INTERP &&
        params.resampleType != ResampleType::BILINEAR_INTERP &&
        params.resampleType != ResampleType::LINEAR_ONNX)
        return false;

    return true;
}

JitConstants ResampleKernelBase::GetJitConstants(const resample_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];
    auto pads_begin = params.pads_begin;
    auto pads_end = params.pads_end;
    if (pads_begin.size() == 4)
        pads_begin.insert(std::next(pads_begin.begin(), 2), 0);
    if (pads_end.size() == 4)
        pads_end.insert(std::next(pads_end.begin(), 2), 0);

    const auto b_size_padded = pads_begin[0] + input.Batch().v + pads_end[0];
    const auto f_size_padded = pads_begin[1] + input.Feature().v + pads_end[1];
    const auto x_size_padded = pads_begin[4] + input.X().v + pads_end[4];
    const auto y_size_padded = pads_begin[3] + input.Y().v + pads_end[3];
    const auto z_size_padded = pads_begin[2] + input.Z().v + pads_end[2];
    const auto out_b_size_padded = output.Batch().v;
    const auto out_f_size_padded = output.Feature().v;
    const auto out_x_size_padded = output.X().v;
    const auto out_y_size_padded = output.Y().v;
    const auto out_z_size_padded = output.Z().v;
    std::vector<float> scales(5);
    std::vector<int32_t> axesUsed(5, 0);
    bool paddingUsed = false;
    for (size_t i = 0; i < pads_begin.size(); ++i) {
        paddingUsed |= (pads_begin[i] != 0 || pads_end[i] != 0);
    }

    scales[0] = static_cast<float>(b_size_padded) / static_cast<float>(out_b_size_padded);
    scales[1] = static_cast<float>(f_size_padded) / static_cast<float>(out_f_size_padded);
    scales[4] = static_cast<float>(x_size_padded) / static_cast<float>(out_x_size_padded);
    scales[3] = static_cast<float>(y_size_padded) / static_cast<float>(out_y_size_padded);
    scales[2] = static_cast<float>(z_size_padded) / static_cast<float>(out_z_size_padded);

    for (std::size_t i = 0; i < params.axes.size(); i++) {
        int idx = getAxisIndex(params.axes[i]);
        axesUsed[idx] = 1;
        if (params.shapeCalculationMode == kernel_selector::ShapeCalculationMode::SCALES)
            scales[idx] = 1.f / params.scales[i];
    }
    for (size_t i = 0; i < scales.size(); ++i) {
        if (scales[i] != 1.f)
            axesUsed[i] = 1;
    }

    jit.AddConstants({
        MakeJitConstant(toString(params.resampleType), ""),
        MakeJitConstant(toString(params.nearestMode), ""),
        MakeJitConstant(toString(params.coordTransMode), ""),
        MakeJitConstant("SCALES", scales),
        MakeJitConstant("PADS_BEGIN", pads_begin),
        MakeJitConstant("PADS_END", pads_end),
        MakeJitConstant("PADDING_USED", static_cast<int>(paddingUsed)),
        MakeJitConstant("AXES_USED", axesUsed),
        MakeJitConstant("KERNEL_W", 2),
        MakeJitConstant("ANTIALIAS", params.antialias),
        MakeJitConstant("CUBE_COEFF", params.cube_coeff),
    });

    if (params.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        if (axesUsed[0] == 1) jit.AddConstant(MakeJitConstant("AXES_USED_B", 1));
        if (axesUsed[1] == 1) jit.AddConstant(MakeJitConstant("AXES_USED_F", 1));
        if (axesUsed[2] == 1) jit.AddConstant(MakeJitConstant("AXES_USED_Z", 1));
        if (axesUsed[3] == 1) jit.AddConstant(MakeJitConstant("AXES_USED_Y", 1));
        if (axesUsed[4] == 1) jit.AddConstant(MakeJitConstant("AXES_USED_X", 1));

        jit.AddConstants({
            MakeJitConstant("PADDED_B", b_size_padded),
            MakeJitConstant("PADDED_F", f_size_padded),
            MakeJitConstant("PADDED_X", x_size_padded),
            MakeJitConstant("PADDED_Y", y_size_padded),
            MakeJitConstant("PADDED_Z", z_size_padded),
        });
    }

    size_t feature_block_size = GetFeatureBlockSize(params);

    if (params.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        jit.AddConstant(MakeJitConstant("FEATURE_BLOCK_SIZE", feature_block_size));
        if (params.outputs[0].Feature().v % feature_block_size != 0) {
            jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));
            jit.AddConstant(MakeJitConstant("FEATURE_LEFTOVER", params.outputs[0].Feature().v % feature_block_size));
        }
    }

    if (params.resampleType == ResampleType::BILINEAR_INTERP || params.resampleType == ResampleType::LINEAR_ONNX) {
        if (params.outputs[0].X().v % 32 != 0) {
            jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));
        }
    }

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (output.GetDType() != Datatype::F16 && output.GetDType() != Datatype::F32) {
        jit.AddConstant(MakeJitConstant("RTE_OUTPUT", 1));
    }

    return jit;
}

KernelsData ResampleKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<resample_params>(params);
    resample_params& newParams = *static_cast<resample_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    return {kd};
}

Datatype ResampleKernelBase::GetAccumulatorType(const resample_params& params) const {
    auto in_dt = params.inputs[0].GetDType();

    if (params.resampleType == ResampleType::NEAREST_NEIGHBOR)
        return in_dt;

    return Datatype::F32;
}

}  // namespace kernel_selector
