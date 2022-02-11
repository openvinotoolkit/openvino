// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_items_class_kernel_base.h"
#include <vector>

namespace kernel_selector {
ParamsKey SoftmaxItemsClassKernelBase::GetDefaultSupportedKey() {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::fb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::f);
    k.EnableOutputLayout(DataLayout::f);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableSoftmaxDim(SoftmaxDim::X);
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::Z);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableSoftmaxDim(SoftmaxDim::BATCH);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

std::vector<size_t> SoftmaxItemsClassKernelBase::GetSoftmaxDimGlobalSizes(SoftmaxDim dim, const DataTensor& out) {
    switch (dim) {
        case SoftmaxDim::X:
            return {out.Y().v * out.Z().v, out.Feature().v, out.Batch().v};
        case SoftmaxDim::Y:
            return {out.X().v * out.Z().v, out.Feature().v, out.Batch().v};
        case SoftmaxDim::Z:
            return {out.X().v * out.Y().v, out.Feature().v, out.Batch().v};
        case SoftmaxDim::FEATURE:
            return {out.X().v * out.Z().v, out.Y().v, out.Batch().v};
        case SoftmaxDim::BATCH:
            return {out.X().v * out.Z().v, out.Y().v, out.Feature().v};
        default:
            return {};
    }
}

JitConstants SoftmaxItemsClassKernelBase::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = SoftmaxKernelBase::GetJitConstants(params, dispatchData);

    switch (params.dim) {
        case SoftmaxDim::X:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH", "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_OTHER0_SIZE", "INPUT0_SIZE_Y"),
                MakeJitConstant("INPUT0_OTHER1_PITCH", "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_OTHER2_PITCH", "INPUT0_Z_PITCH"),
                MakeJitConstant("INPUT0_OTHER3_PITCH", "INPUT0_BATCH_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH", "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM", "INPUT0_SIZE_X"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH", "OUTPUT_Y_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH", "OUTPUT_FEATURE_PITCH"),
                MakeJitConstant("OUTPUT_OTHER2_PITCH", "OUTPUT_Z_PITCH"),
                MakeJitConstant("OUTPUT_OTHER3_PITCH", "OUTPUT_BATCH_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH", "OUTPUT_X_PITCH"),
            });
            idx_order = {"other3", "other1", ndims == 5 ? "other2" : "0", "other0", "cls"};
            break;
        case SoftmaxDim::Y:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH", "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_OTHER0_SIZE", "INPUT0_SIZE_X"),
                MakeJitConstant("INPUT0_OTHER1_PITCH", "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_OTHER2_PITCH", "INPUT0_Z_PITCH"),
                MakeJitConstant("INPUT0_OTHER3_PITCH", "INPUT0_BATCH_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH", "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM", "INPUT0_SIZE_Y"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH", "OUTPUT_X_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH", "OUTPUT_FEATURE_PITCH"),
                MakeJitConstant("OUTPUT_OTHER2_PITCH", "OUTPUT_Z_PITCH"),
                MakeJitConstant("OUTPUT_OTHER3_PITCH", "OUTPUT_BATCH_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH", "OUTPUT_Y_PITCH"),
            });
            idx_order = {"other3", "other1", ndims == 5 ? "other2" : "0", "cls", "other0"};
            break;
        case SoftmaxDim::Z:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH", "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_OTHER0_SIZE", "INPUT0_SIZE_X"),
                MakeJitConstant("INPUT0_OTHER1_PITCH", "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_OTHER2_PITCH", "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_OTHER3_PITCH", "INPUT0_BATCH_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH", "INPUT0_Z_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM", "INPUT0_SIZE_Z"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH", "OUTPUT_X_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH", "OUTPUT_FEATURE_PITCH"),
                MakeJitConstant("OUTPUT_OTHER2_PITCH", "OUTPUT_Y_PITCH"),
                MakeJitConstant("OUTPUT_OTHER3_PITCH", "OUTPUT_BATCH_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH", "OUTPUT_Z_PITCH"),
            });
            idx_order = {"other3", "other1", "cls", "other2", "other0"};
            break;
        case SoftmaxDim::FEATURE:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH", "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_OTHER0_SIZE", "INPUT0_SIZE_X"),
                MakeJitConstant("INPUT0_OTHER1_PITCH", "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_OTHER2_PITCH", "INPUT0_Z_PITCH"),
                MakeJitConstant("INPUT0_OTHER3_PITCH", "INPUT0_BATCH_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH", "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM", "INPUT0_FEATURE_NUM"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH", "OUTPUT_X_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH", "OUTPUT_Y_PITCH"),
                MakeJitConstant("OUTPUT_OTHER2_PITCH", "OUTPUT_Z_PITCH"),
                MakeJitConstant("OUTPUT_OTHER3_PITCH", "OUTPUT_BATCH_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH", "OUTPUT_FEATURE_PITCH"),
            });
            idx_order = {"other3", "cls", ndims == 5 ? "other2" : "0", "other1", "other0"};
            break;
        case SoftmaxDim::BATCH:
            jit.AddConstants({
                MakeJitConstant("INPUT0_OTHER0_PITCH", "INPUT0_X_PITCH"),
                MakeJitConstant("INPUT0_OTHER0_SIZE", "INPUT0_SIZE_X"),
                MakeJitConstant("INPUT0_OTHER1_PITCH", "INPUT0_Y_PITCH"),
                MakeJitConstant("INPUT0_OTHER2_PITCH", "INPUT0_Z_PITCH"),
                MakeJitConstant("INPUT0_OTHER3_PITCH", "INPUT0_FEATURE_PITCH"),
                MakeJitConstant("INPUT0_CLASS_PITCH", "INPUT0_BATCH_PITCH"),
                MakeJitConstant("INPUT0_CLASS_NUM", "INPUT0_BATCH_NUM"),
                MakeJitConstant("OUTPUT_OTHER0_PITCH", "OUTPUT_X_PITCH"),
                MakeJitConstant("OUTPUT_OTHER1_PITCH", "OUTPUT_Y_PITCH"),
                MakeJitConstant("OUTPUT_OTHER2_PITCH", "OUTPUT_Z_PITCH"),
                MakeJitConstant("OUTPUT_OTHER3_PITCH", "OUTPUT_FEATURE_PITCH"),
                MakeJitConstant("OUTPUT_CLASS_PITCH", "OUTPUT_BATCH_PITCH"),
            });
            idx_order = {"cls", "other3", ndims == 5 ? "other2" : "0", "other1", "other0"};
            break;
        default:
            break;
    }

    // TODO: W/A - currently using low precision accumulator type. (for testing only)
    if (params.output.GetDType() == Datatype::F16) {
        jit.AddConstant(MakeJitConstant("ACCUMULATOR_TYPE", "half"));
    }

    return jit;
}
}  // namespace kernel_selector
