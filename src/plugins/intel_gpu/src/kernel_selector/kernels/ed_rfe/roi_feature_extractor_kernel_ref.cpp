// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_feature_extractor_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <algorithm>
#include <string>

namespace kernel_selector {
namespace {
    ExperimentalDetectronROIFeatureExtractorRef::DispatchData SetDefault(const experimental_detectron_roi_feature_extractor_params& params) {
        ExperimentalDetectronROIFeatureExtractorRef::DispatchData dispatch_data;
        auto in_layout = params.inputs[0].GetLayout();
        auto out_layout = params.outputs[0].GetLayout();

        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                         { Tensor::DataChannelName::FEATURE },
                                                                         { Tensor::DataChannelName::BATCH }};

        dispatch_data.gws = {params.outputs[0].X().v * params.outputs[0].Y().v, params.outputs[0].Feature().v, params.outputs[0].Batch().v};

        dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

        return dispatch_data;
    }

    const std::string common_level_name = "level_";

    std::string GetInputLevelParams(size_t levels_num) {
        std::string result = "const __global INPUT1_TYPE* " + common_level_name + "1";
        std::string idx = "";
        for (size_t i = 1; i < levels_num; i++) {
            idx = std::to_string(i + 1);
            result += ", const __global INPUT" + idx + "_TYPE* " + common_level_name + idx;
        }
        return result;
    }

    std::string GetDefinedLevelPtrs(size_t levels_num) {
        std::string result = "(const __global INPUT1_TYPE*[]){" + common_level_name + "1";
        for (size_t i = 1; i < levels_num; i++) {
            result += ", " + common_level_name + std::to_string(i + 1);
        }
        result += "}";
        return result;
    }

    std::string GetDefinedSpatialScales(const std::vector<int64_t>& scales, size_t levels_num) {
        std::string result = "(float[]){" + std::to_string(1.0f / scales[0]);
        for (size_t i = 1; i < levels_num; i++) {
            result += ", " + std::to_string(1.0f / scales[i]);
        }
        result += "}";
        return result;
    }

    std::string GetDefinedLevelSizes(size_t levels_num) {
        std::string result = "(__private size_t[]){INPUT1_SIZE_Y, INPUT1_SIZE_X, INPUT1_OFFSET";
        std::string idx = "";
        for (size_t i = 1; i < levels_num; i++) {
            idx = std::to_string(i + 1);
            result += " ,INPUT" + idx + "_SIZE_Y, INPUT" + idx + "_SIZE_X, INPUT" + idx + "_OFFSET";
        }
        result += "}";
        return result;
    }

    std::string GetIndexCalculationFuncs(size_t levels_num) {
        std::string result = "if (level == 0) { idx = INPUT1_GET_INDEX(0, c, y, x); }";
        std::string idx = "";
        for (size_t i = 1; i < levels_num; i++) {
            idx = std::to_string(i + 1);
            result += " else if (level == " + std::to_string(i) + ") { idx = INPUT" + idx + "_GET_INDEX(0, c, y, x); }";
        }
        return result;
    }
}  // namespace

JitConstants ExperimentalDetectronROIFeatureExtractorRef::GetJitConstants(const experimental_detectron_roi_feature_extractor_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    const size_t levels_num = params.number_of_inputs - 1;

    jit.AddConstants({MakeJitConstant("POOLED_HEIGHT", params.pooled_height),
                      MakeJitConstant("POOLED_WIDTH", params.pooled_width),
                      MakeJitConstant("SAMPLING_RATIO", params.sampling_ratio),
                      MakeJitConstant("IS_ALIGNED", params.aligned),
                      MakeJitConstant("NUM_PYRAMID_LEVELS", levels_num),
                      MakeJitConstant("INPUT_LEVEL_PARAMS", GetInputLevelParams(levels_num)),
                      MakeJitConstant("LEVEL_PTRS", GetDefinedLevelPtrs(levels_num)),
                      MakeJitConstant("SPATIAL_SCALES", GetDefinedSpatialScales(params.pyramid_scales, levels_num)),
                      MakeJitConstant("LEVEL_SIZES", GetDefinedLevelSizes(levels_num)),
                      MakeJitConstant("LEVELS_IDX_CALC_FUNCS", GetIndexCalculationFuncs(levels_num))});

    return jit;
}

KernelsData ExperimentalDetectronROIFeatureExtractorRef::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::EXPERIMENTAL_DETECTRON_ROI_FEATURE_EXTRACTOR);
    const experimental_detectron_roi_feature_extractor_params& org_params = static_cast<const experimental_detectron_roi_feature_extractor_params&>(params);

    if (!org_params.activations.empty()) {
        return {};
    }

    DispatchData dispatch_data = SetDefault(org_params);
    KernelData kd = KernelData::Default<experimental_detectron_roi_feature_extractor_params>(params);

    auto cldnn_jit = GetJitConstants(org_params);
    auto entry_point = GetEntryPoint(kernelName, org_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point, "", false, false, static_cast<int>(org_params.number_of_inputs));
    return {kd};
}

KernelsPriority ExperimentalDetectronROIFeatureExtractorRef::GetKernelsPriority(const Params& /*params*/) const {
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
