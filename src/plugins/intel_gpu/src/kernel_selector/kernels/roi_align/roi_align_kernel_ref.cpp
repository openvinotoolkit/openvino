// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "roi_align_kernel_ref.h"
#include <kernel_selector_utils.h>

namespace kernel_selector {

ParamsKey ROIAlignKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnablePoolType(PoolType::MAX);
    k.EnablePoolType(PoolType::AVG);
    return k;
}

namespace {

ROIAlignKernelRef::DispatchData SetDefault(const roi_align_params& params) {
    ROIAlignKernelRef::DispatchData dispatchData;
    // Determine global work sizes.
    dispatchData.gws[0] = params.outputs[0].LogicalSize();
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

} // anonymous namespace

KernelsData ROIAlignKernelRef::GetKernelsData(const Params &params) const {
    if (!Validate(params)) {
        return {};
    }
    KernelData kernel_data = KernelData::Default<roi_align_params>(params);
    roi_align_params &new_params = dynamic_cast<roi_align_params&>(*kernel_data.params.get());
    auto dispatch_data = SetDefault(new_params);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);
    auto roi_align_specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, roi_align_specific_jit, entry_point);
    FillCLKernelData(kernel_data.kernels[0], dispatch_data, params.engineInfo,
            kernelName, jit, entry_point);
    kernel_data.kernels[0].params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    kernel_data.kernels[0].params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});

    return {kernel_data};
}

float ROIAlignKernelRef::GetKernelsPriority(const Params &params) const {
    return FORCE_PRIORITY_1;
}

bool ROIAlignKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType::ROI_ALIGN) {
        return false;
    }

    const roi_align_params &params = static_cast<const roi_align_params&>(p);
    if (params.inputs.size() != 3)
        return false;

    return true;
}

JitConstants ROIAlignKernelRef::GetJitConstants(const roi_align_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("SPATIAL_SCALE", params.spatial_scale));
    jit.AddConstant(MakeJitConstant("SAMPLING_RATIO", params.sampling_ratio));

    if (params.pooling_mode == PoolType::MAX) {
        jit.AddConstant(MakeJitConstant("MAX_POOL", true));
    } else if (params.pooling_mode == PoolType::AVG) {
        jit.AddConstant(MakeJitConstant("AVG_POOL", true));
    }

    if (params.aligned_mode == roi_aligned_mode::ASYMMETRIC) {
        jit.AddConstant(MakeJitConstant("OFFSET_SRC", 0.f));
        jit.AddConstant(MakeJitConstant("OFFSET_DST", 0.f));
        jit.AddConstant(MakeJitConstant("MIN_SIZE", 1.0f));
    } else if (params.aligned_mode == roi_aligned_mode::HALF_PIXEL_FOR_NN) {
        jit.AddConstant(MakeJitConstant("OFFSET_SRC", 0.f));
        jit.AddConstant(MakeJitConstant("OFFSET_DST", -0.5f));
        jit.AddConstant(MakeJitConstant("MIN_SIZE", 0.f));
    } else if (params.aligned_mode == roi_aligned_mode::HALF_PIXEL) {
        jit.AddConstant(MakeJitConstant("OFFSET_SRC", 0.5f));
        jit.AddConstant(MakeJitConstant("OFFSET_DST", -0.5f));
        jit.AddConstant(MakeJitConstant("MIN_SIZE", 0.f));
    }

    jit.AddConstant(MakeJitConstant("X1", "x1"));
    jit.AddConstant(MakeJitConstant("Y1", "y1"));
    jit.AddConstant(MakeJitConstant("ROI_WIDTH", "roi_width"));
    jit.AddConstant(MakeJitConstant("ROI_HEIGHT", "roi_height"));
    jit.AddConstant(MakeJitConstant("SAMPLE_X", "sample_x"));
    jit.AddConstant(MakeJitConstant("SAMPLE_Y", "sample_y"));

    if (params.rotated_mode == false) {
        const char* prepare_roi_macro =
            R"( \
        const INPUT1_TYPE X1 = (roi_ptr[0] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST; \
        const INPUT1_TYPE Y1 = (roi_ptr[1] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST; \
        const INPUT1_TYPE x2 = (roi_ptr[2] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST; \
        const INPUT1_TYPE y2 = (roi_ptr[3] + (INPUT1_TYPE)OFFSET_SRC) * (INPUT1_TYPE)SPATIAL_SCALE + (INPUT1_TYPE)OFFSET_DST; \
        const INPUT1_TYPE ROI_WIDTH = MAX(x2 - X1, (INPUT1_TYPE)MIN_SIZE); \
        const INPUT1_TYPE ROI_HEIGHT = MAX(y2 - Y1, (INPUT1_TYPE)MIN_SIZE);
        )";
        jit.AddConstant(MakeJitConstant("PREPARE_ROI(roi_ptr)", prepare_roi_macro));

        const char* transform_macro =
            R"( \
        INPUT1_TYPE SAMPLE_X = pre_sample_x; \
        INPUT1_TYPE SAMPLE_Y = pre_sample_y; 
        )";

        jit.AddConstant(MakeJitConstant("TRANSFORM_POINT_TO_IMAGE_SPACE(pre_sample_x,pre_sample_y)", transform_macro));
    } else {
        jit.AddConstant(MakeJitConstant("CLOCKWISE", params.clockwise));

        const char* prepare_roi_macro =
            R"( \
        const INPUT1_TYPE center_x =   roi_ptr[0] * (INPUT1_TYPE)SPATIAL_SCALE - (INPUT1_TYPE)0.5f; \
        const INPUT1_TYPE center_y =   roi_ptr[1] * (INPUT1_TYPE)SPATIAL_SCALE - (INPUT1_TYPE)0.5f; \
        const INPUT1_TYPE ROI_WIDTH =  roi_ptr[2] * (INPUT1_TYPE)SPATIAL_SCALE; \
        const INPUT1_TYPE ROI_HEIGHT = roi_ptr[3] * (INPUT1_TYPE)SPATIAL_SCALE; \
        INPUT1_TYPE angle = roi_ptr[4];                         \
                                                                \
        if (CLOCKWISE) {                                        \
            angle = -angle;                                     \
        }                                                       \
        const INPUT1_TYPE cos_angle = cos(angle);               \
        const INPUT1_TYPE sin_angle = sin(angle);               \
                                                                \
        const INPUT1_TYPE X1 = -ROI_WIDTH / (INPUT1_TYPE)2.0f;  \
        const INPUT1_TYPE Y1 = -ROI_HEIGHT / (INPUT1_TYPE)2.0f;
        )";
        jit.AddConstant(MakeJitConstant("PREPARE_ROI(roi_ptr)", prepare_roi_macro));

        const char* transform_macro =
            R"( \
            INPUT1_TYPE SAMPLE_Y = pre_sample_y * cos_angle - pre_sample_x * sin_angle + center_y; \
            INPUT1_TYPE SAMPLE_X = pre_sample_y * sin_angle + pre_sample_x * cos_angle + center_x;
        )";

        jit.AddConstant(MakeJitConstant("TRANSFORM_POINT_TO_IMAGE_SPACE(pre_sample_x,pre_sample_y)", transform_macro));
    }

    return jit;
}

}  // namespace kernel_selector
