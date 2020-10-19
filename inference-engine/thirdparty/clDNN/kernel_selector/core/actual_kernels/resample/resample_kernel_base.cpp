// Copyright (c) 2019-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "resample_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <iostream>

namespace {
int getAxisIndex(kernel_selector::InterpolateAxis axis) {
    switch(axis) {
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
        if (params.output.Feature().v % s == 0)
            return s;
    if (params.output.Feature().v < max_size)
        return params.output.Feature().v;
    for (size_t f = 1; f <= params.output.Feature().v && f <= max_size; f++)
        if (params.output.Feature().v % f == 0)
            feature_block_size = f;
    return std::max(feature_block_size, min_size);
}

ResampleKernelBase::DispatchData ResampleKernelBase::SetDefault(const kernel_selector::resample_params &arg) const {
    DispatchData dispatchData;
    const auto& out = arg.output;

    if (arg.resampleType == ResampleType::NEAREST_NEIGHBOR)
        dispatchData.gws = { out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v };
    else if (arg.resampleType == ResampleType::BILINEAR_INTERP || arg.resampleType == ResampleType::LINEAR_ONNX)
        dispatchData.gws = { Align(out.X().v, 32), out.Y().v, out.Batch().v };
    else if (arg.resampleType == ResampleType::CAFFE_BILINEAR_INTERP)
        dispatchData.gws = { out.X().v * out.Y().v, CeilDiv(out.Feature().v, GetFeatureBlockSize(arg)), out.Batch().v * out.Z().v };
    else
        dispatchData.gws = { out.X().v, out.Y().v * out.Z().v, out.Feature().v * out.Batch().v };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo);

    if (arg.resampleType == ResampleType::BILINEAR_INTERP || arg.resampleType == ResampleType::LINEAR_ONNX) {
        dispatchData.lws[0] = 32;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    dispatchData.efficiency = FORCE_PRIORITY_7;

    return dispatchData;
}

bool ResampleKernelBase::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::RESAMPLE || o.GetType() != KernelType::RESAMPLE) {
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
        params.resampleType != ResampleType::BILINEAR_INTERP)
        return false;

    return true;
}

JitConstants ResampleKernelBase::GetJitConstants(const resample_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& input = params.inputs[0];
    const auto& output = params.output;
    const auto align_corners = params.align_corners;
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

    if (align_corners) {
        scales[0] = (out_b_size_padded) > 1
                        ? static_cast<float>(b_size_padded - 1) / static_cast<float>(out_b_size_padded - 1)
                        : 0.0f;
        scales[1] = (out_f_size_padded) > 1
                        ? static_cast<float>(f_size_padded - 1) / static_cast<float>(out_f_size_padded - 1)
                        : 0.0f;
        scales[4] = (out_x_size_padded) > 1
                        ? static_cast<float>(x_size_padded - 1) / static_cast<float>(out_x_size_padded - 1)
                        : 0.0f;
        scales[3] = (out_y_size_padded) > 1
                        ? static_cast<float>(y_size_padded - 1) / static_cast<float>(out_y_size_padded - 1)
                        : 0.0f;
        scales[2] = (out_z_size_padded) > 1
                        ? static_cast<float>(z_size_padded - 1) / static_cast<float>(out_z_size_padded - 1)
                        : 0.0f;
    } else {
        scales[0] = static_cast<float>(b_size_padded) / static_cast<float>(out_b_size_padded);
        scales[1] = static_cast<float>(f_size_padded) / static_cast<float>(out_f_size_padded);
        scales[4] = static_cast<float>(x_size_padded) / static_cast<float>(out_x_size_padded);
        scales[3] = static_cast<float>(y_size_padded) / static_cast<float>(out_y_size_padded);
        scales[2] = static_cast<float>(z_size_padded) / static_cast<float>(out_z_size_padded);
    }
    for (const auto& it : params.axesAndScales) {
        int idx = getAxisIndex(it.first);
        axesUsed[idx] = 1;
        if (params.shapeCalculationMode == kernel_selector::ShapeCalculationMode::SCALES)
            scales[idx] = 1.f / it.second;
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
        MakeJitConstant("PADDING_USED", (int)paddingUsed),
        MakeJitConstant("AXES_USED", axesUsed),
        MakeJitConstant("ALIGN_CORNERS", align_corners),
        MakeJitConstant("KERNEL_W", 2),
        MakeJitConstant("ANTIALIAS", params.antialias),
        MakeJitConstant("CUBE_COEFF", params.cube_coeff),
    });

    size_t feature_block_size = GetFeatureBlockSize(params);

    if (params.resampleType == ResampleType::CAFFE_BILINEAR_INTERP) {
        jit.AddConstant(MakeJitConstant("FEATURE_BLOCK_SIZE", feature_block_size));
        if (params.output.Feature().v % feature_block_size != 0) {
            jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));
            jit.AddConstant(MakeJitConstant("FEATURE_LEFTOVER", params.output.Feature().v % feature_block_size));
        }
    }

    if (params.resampleType == ResampleType::BILINEAR_INTERP || params.resampleType == ResampleType::LINEAR_ONNX) {
        if (params.output.X().v % 32 != 0) {
            jit.AddConstant(MakeJitConstant("LEFTOVERS", 1));
        }
    }

    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    return jit;
}

KernelsData ResampleKernelBase::GetCommonKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<resample_params>(params);
    resample_params& newParams = *static_cast<resample_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = dispatchData.efficiency;

    return {kd};
}

Datatype ResampleKernelBase::GetAccumulatorType(const resample_params& params) const {
    auto in_dt = params.inputs[0].GetDType();
    auto out_dt = params.output.GetDType();

    if (params.resampleType == ResampleType::NEAREST_NEIGHBOR)
        return in_dt;

    auto smaller_fp_type = [](const Datatype& current, const Datatype& candidate) -> Datatype {
        if (candidate != Datatype::F32 || candidate != Datatype::F16)
            return current;

        return BytesPerElement(candidate) < BytesPerElement(current) ? candidate : current;
    };

    Datatype fp_type = Datatype::F32;
    fp_type = smaller_fp_type(fp_type, in_dt);
    fp_type = smaller_fp_type(fp_type, out_dt);

    return fp_type;
}

}  // namespace kernel_selector
