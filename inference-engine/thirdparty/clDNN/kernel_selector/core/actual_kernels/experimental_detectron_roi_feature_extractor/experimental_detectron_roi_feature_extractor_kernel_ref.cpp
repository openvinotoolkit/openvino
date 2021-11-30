// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_roi_feature_extractor_kernel_ref.h"
#include <algorithm>
#include <string>

namespace kernel_selector {
namespace {
    ExperimentalDetectronROIFeatureExtractorRef::DispatchData SetDefault(const experimental_detectron_roi_feature_extractor_params& params) {
        ExperimentalDetectronROIFeatureExtractorRef::DispatchData dispatch_data;

        dispatch_data.gws[0] = params.output.LogicalSize();
        dispatch_data.gws[1] = 1;
        dispatch_data.gws[2] = 1;

        dispatch_data.lws[0] = std::min(std::max(dispatch_data.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
        while (dispatch_data.gws[0] % dispatch_data.lws[0] != 0) {
            --dispatch_data.lws[0];
        }
        dispatch_data.lws[1] = 1;
        dispatch_data.lws[2] = 1;

        return dispatch_data;
    }

    const std::string common_level_name = "level_";

    std::string GetInputLevelParams(std::size_t levels_num) {
        std::string result = "const __global INPUT1_TYPE* " + common_level_name + "1";
        std::string idx = "";
        for (std::size_t i = 1; i < levels_num; i++) {
            idx = std::to_string(i + 1);
            result += ", const __global INPUT" + idx + "_TYPE* " + common_level_name + idx;
        }
        return result;
    }

    const std::string level_ptrs = "level_ptrs";

    std::string GetDefinedLevelPtrs(std::size_t levels_num) {
        std::string result = "const __global INPUT1_TYPE* " + level_ptrs + "[" + std::to_string(levels_num) + "] = {" + common_level_name + "1";
        for (std::size_t i = 1; i < levels_num; i++) {
            result += ", " + common_level_name + std::to_string(i + 1);
        }
        result += "}";
        return result;
    }

    const std::string spatial_scales = "spatial_scales";

    std::string GetDefinedSpatialScales(const std::vector<int64_t>& scales, std::size_t levels_num) {
        std::string result = "__constant float " + spatial_scales + "[" + std::to_string(levels_num) + "] = {" + std::to_string(1.0f / scales[0]);
        for (std::size_t i = 1; i < levels_num; i++) {
            result += ", " + std::to_string(1.0f / scales[i]);
        }
        result += "}";
        return result;
    }

    const std::string level_sizes = "level_sizes";

    std::string GetDefinedLevelSizes(std::size_t levels_num) {
    std::string result = "__constant int " + level_sizes + "[" + std::to_string(3 * levels_num) +"] = {INPUT1_SIZE_Y, INPUT1_SIZE_X, INPUT1_OFFSET";
    std::string idx = "";
    for (std::size_t i = 1; i < levels_num; i++) {
        idx = std::to_string(i + 1);
        result += " ,INPUT" + idx + "_SIZE_Y, INPUT" + idx + "_SIZE_X, INPUT" + idx + "_OFFSET";
    }
    result += "}";
    return result;
}
}  // namespace

JitConstants ExperimentalDetectronROIFeatureExtractorRef::GetJitConstants(const experimental_detectron_roi_feature_extractor_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    const std::size_t levels_num = params.number_of_inputs - 1;

    jit.AddConstants({MakeJitConstant("POOLED_HEIGHT", params.pooled_height),
                      MakeJitConstant("POOLED_WIDTH", params.pooled_width),
                      MakeJitConstant("SAMPLING_RATIO", params.sampling_ratio),
                      MakeJitConstant("IS_ALIGNED", params.aligned),
                      MakeJitConstant("NUM_PYRAMID_LEVELS", levels_num),
                      MakeJitConstant("INPUT_LEVEL_PARAMS", GetInputLevelParams(levels_num)),
                      MakeJitConstant("LEVEL_PTRS", level_ptrs),
                      MakeJitConstant("DEFINE_LEVEL_PTRS", GetDefinedLevelPtrs(levels_num)),
                      MakeJitConstant("SPATIAL_SCALES", spatial_scales),
                      MakeJitConstant("DEFINE_SPATIAL_SCALES", GetDefinedSpatialScales(params.pyramid_scales, levels_num)),
                      MakeJitConstant("LEVEL_SIZES", level_sizes),
                      MakeJitConstant("DEFINE_LEVEL_SIZES", GetDefinedLevelSizes(levels_num))});

    return jit;
}

KernelsData ExperimentalDetectronROIFeatureExtractorRef::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::EXPERIMENTAL_DETECTRON_ROI_FEATURE_EXTRACTOR);
    const experimental_detectron_roi_feature_extractor_params& org_params = static_cast<const experimental_detectron_roi_feature_extractor_params&>(params);

    if (!org_params.activations.empty()) {
        return {};
    }

    DispatchData dispatch_data = SetDefault(org_params);
    KernelData kd = KernelData::Default<experimental_detectron_roi_feature_extractor_params>(params);

    auto cldnn_jit = GetJitConstants(org_params);
    auto entry_point = GetEntryPoint(kernelName, org_params.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point, "", false, false, org_params.number_of_inputs);
    return {kd};
}

KernelsPriority ExperimentalDetectronROIFeatureExtractorRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}

ParamsKey ExperimentalDetectronROIFeatureExtractorRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::F16);
    key.EnableInputDataType(Datatype::F32);
    key.EnableOutputDataType(Datatype::F16);
    key.EnableOutputDataType(Datatype::F32);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDifferentTypes();
    return key;
}

}  // namespace kernel_selector
