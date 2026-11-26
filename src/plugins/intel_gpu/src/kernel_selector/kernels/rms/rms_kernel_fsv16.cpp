// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_kernel_fsv16.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {
static constexpr size_t subgroup_size = 16;

namespace {
struct Fsv16Tiling {
    size_t xblk = 1;
    size_t fsplit = 1;
    size_t blocks_per_sg = 1;
    bool cache = true;
};

Fsv16Tiling get_fsv16_tiling(const rms_params& params) {
    constexpr size_t max_cached_per_lane = 64;
    constexpr size_t max_fsplit = 16;
    const auto& in = params.inputs[0];
    Fsv16Tiling t;
    const size_t x = in.X().v;
    // XBLK is the largest of 8/4/2/1 dividing X, so a tile never crosses a row end. Spatial padding (X
    // included) needs no check: in b_fs_(z)yx_fsv16 the X pitch is always 16 elements, padding only moves
    // where a row starts (applied by INPUT0_GET_INDEX/OUTPUT_GET_INDEX), so the XBLK columns of one fsv16
    // block remain XBLK * 16 contiguous elements that a single BLOCK_READN/BLOCK_WRITEN covers.
    for (size_t cand : {8, 4, 2, 1}) {
        if (x % cand == 0) {
            t.xblk = cand;
            break;
        }
    }
    const size_t fblocks = in.Feature().v / subgroup_size;
    const size_t tiles = in.Batch().v * in.Z().v * in.Y().v * (x / t.xblk);
    const size_t target_sgs = std::max<size_t>(params.engineInfo.computeUnitsCount, 1) * 32;
    const size_t max_split = std::min({max_fsplit, fblocks, params.engineInfo.maxWorkGroupSize / subgroup_size});

    auto per_sg = [&](size_t split) {
        return (fblocks + split - 1) / split;
    };
    size_t split = 1;
    while (split * 2 <= max_split && per_sg(split) * t.xblk > max_cached_per_lane) {
        split *= 2;
    }
    while (split * 2 <= max_split && tiles * split < target_sgs) {
        split *= 2;
    }
    t.fsplit = split;
    t.blocks_per_sg = per_sg(split);
    t.cache = t.blocks_per_sg * t.xblk <= max_cached_per_lane;
    return t;
}
}  // namespace

ParamsKey RMSKernelFsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::BF16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::BF16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F8E4M3);
    k.EnableOutputDataType(Datatype::F8E5M2);
    k.EnableOutputDataType(Datatype::F8E8M0);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

DeviceFeaturesKey RMSKernelFsv16::get_required_device_features_key(const Params& /*params*/) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_reduce();
    k.requires_reqd_subgroup_size();
    return k;
}

// Each fsv16 block stores 16 consecutive features at stride 1 and XBLK adjacent X elements sit at stride 16, so one
// SIMD16 sub-group can block-read/write XBLK spatial columns at once; this needs whole blocks and no feature padding.
bool RMSKernelFsv16::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    const auto& params = static_cast<const rms_params&>(p);
    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];
    if (params.has_dynamic_tensors() || GetNormalizationAxis(params) != Tensor::DataChannelName::FEATURE || in.GetLayout() != out.GetLayout()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (in.Feature().v == 0 || in.Feature().v % subgroup_size != 0 || in.Feature().v != out.Feature().v) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (in.Feature().pad.before != 0 || in.Feature().pad.after != 0 || out.Feature().pad.before != 0 || out.Feature().pad.after != 0) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (params.elementwise_affine) {
        const auto n = params.inputs[1].LogicalSize();
        if (n != 1 && n != in.Feature().v) {
            DO_NOT_USE_THIS_KERNEL(p.layerID);
        }
    }
    return true;
}

RMSKernelBase::DispatchData RMSKernelFsv16::SetDefault(const rms_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];
    const auto t = get_fsv16_tiling(params);
    dispatchData.dataSize = input.Feature().v;
    dispatchData.dataCount = (input.X().v / t.xblk) * input.Y().v * input.Z().v * input.Batch().v;
    dispatchData.lws = {1, t.fsplit * subgroup_size, 1};
    dispatchData.gws = {dispatchData.dataCount, t.fsplit * subgroup_size, 1};
    dispatchData.itemsNum = t.blocks_per_sg;
    return dispatchData;
}

JitConstants RMSKernelFsv16::GetJitConstants(const rms_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    const auto& in = params.inputs[0];
    const auto t = get_fsv16_tiling(params);
    jit.AddConstants({
        MakeJitConstant("RMS_GAMMA_IS_SCALAR", params.elementwise_affine && params.inputs[1].LogicalSize() == 1),
        MakeJitConstant("SUB_GROUP_SIZE", subgroup_size),
        MakeJitConstant("XBLK", t.xblk),
        MakeJitConstant("FSPLIT", t.fsplit),
        MakeJitConstant("BLOCKS_PER_SG", t.blocks_per_sg),
        MakeJitConstant("F_BLOCKS", in.Feature().v / subgroup_size),
        MakeJitConstant("X_BLOCKS", in.X().v / t.xblk),
        MakeJitConstant("CACHE_INPUT", t.cache),
        MakeJitConstant("INV_DATA_SIZE", 1.0F / static_cast<float>(in.Feature().v)),
    });
    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (in.GetDims().size() == 5) {
            idx_order = {"(b_idx)", "(f_base + lane)", "(z_idx)", "(y_idx)", "(x0 + xi)"};
        } else {
            idx_order = {"(b_idx)", "(f_base + lane)", "(y_idx)", "(x0 + xi)"};
        }
        auto conf = FusedOpsConfiguration("", idx_order, "normalized", params.outputs[0].GetDType(), 1);
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jit;
}

KernelsData RMSKernelFsv16::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority RMSKernelFsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}
}  // namespace kernel_selector
