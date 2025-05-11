// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
static const std::vector<size_t> preferred_sizes = {8, 4, 2, 1};

ParamsKey ReorderWeightsOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT32);
    k.EnableOutputWeightsType(WeightsType::INT8);
    k.EnableOutputWeightsType(WeightsType::F16);
    k.EnableOutputWeightsType(WeightsType::F32);
    k.EnableOutputWeightsType(WeightsType::INT32);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableInputWeightsLayout(WeightsLayout::ioyx);
    k.EnableInputWeightsLayout(WeightsLayout::oyxi);
    k.EnableInputWeightsLayout(WeightsLayout::oyix);
    k.EnableInputWeightsLayout(WeightsLayout::oxiy);
    k.EnableInputWeightsLayout(WeightsLayout::iyxo);
    k.EnableInputWeightsLayout(WeightsLayout::yxio);
    k.EnableInputWeightsLayout(WeightsLayout::oizyx);
    k.EnableInputWeightsLayout(WeightsLayout::iozyx);
    k.EnableInputWeightsLayout(WeightsLayout::goiyx);
    k.EnableInputWeightsLayout(WeightsLayout::gioyx);
    k.EnableInputWeightsLayout(WeightsLayout::goizyx);
    k.EnableInputWeightsLayout(WeightsLayout::giozyx);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_isv16_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_zyx_isv16_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::g_os_is_yx_isv16_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::g_os_is_zyx_isv16_osv16);

    k.EnableOutputWeightsLayout(WeightsLayout::os_iyx_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::os_iyx_osv32);
    k.EnableOutputWeightsLayout(WeightsLayout::os_iyx_osv32__ai32);
    k.EnableOutputWeightsLayout(WeightsLayout::g_os_iyx_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::g_os_iyx_osv32);

    k.EnableOutputWeightsLayout(WeightsLayout::is_os_yx_isv16_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::is_os_zyx_isv16_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::g_is_os_yx_isv16_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::g_is_os_zyx_isv16_osv16);

    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_osv16_isv16);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_zyx_osv32_isv16);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_zyx_osv64_isv16);

    k.EnableOutputWeightsLayout(WeightsLayout::g_os_zyx_is_osv16_isv16);
    k.EnableOutputWeightsLayout(WeightsLayout::g_os_zyx_is_osv16_isv32);
    k.EnableOutputWeightsLayout(WeightsLayout::g_os_zyx_is_osv32_isv16);
    k.EnableOutputWeightsLayout(WeightsLayout::g_os_zyx_is_osv32_isv32);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

DeviceFeaturesKey ReorderWeightsOpt::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;

    bool requires_blocked_read_write_char = false;
    bool requires_blocked_read_write_short = false;
    bool requires_blocked_read_write = false;
    const auto& casted_params = static_cast<const reorder_weights_params&>(params);

    std::vector<WeightsType> tensor_types = {casted_params.input.GetDType(), casted_params.output.GetDType() };
    for (auto& type : tensor_types) {
        if (type == WeightsType::F16) {
            requires_blocked_read_write_short = true;
        } else if (type == WeightsType::F32) {
            requires_blocked_read_write = true;
        } else if (type == WeightsType::UINT8 || type == WeightsType::INT8) {
            requires_blocked_read_write_char = true;
        }
    }

    if (requires_blocked_read_write)
        k.requires_blocked_read_write();

    if (requires_blocked_read_write_short)
        k.requires_blocked_read_write_short();

    if (requires_blocked_read_write_char)
        k.requires_blocked_read_write_char();

    k.requires_subgroups();

    return k;
}

static inline std::pair<size_t, size_t> GetSliceSizes(WeightsLayout l) {
    if (l == WeightsLayout::os_is_yx_isv16_osv16 || l == WeightsLayout::os_is_zyx_isv16_osv16 ||
        l == WeightsLayout::g_os_is_yx_isv16_osv16 || l == WeightsLayout::g_os_is_zyx_isv16_osv16 ||
        l == WeightsLayout::is_os_zyx_isv16_osv16 || l == WeightsLayout::is_os_yx_isv16_osv16 ||
        l == WeightsLayout::os_is_yx_osv16_isv16 || l == WeightsLayout::g_os_zyx_is_osv16_isv16 ||
        l == WeightsLayout::g_is_os_yx_isv16_osv16 || l == WeightsLayout::g_is_os_zyx_isv16_osv16)
        return {16, 16};
    else if (l == WeightsLayout::os_iyx_osv16 || l == WeightsLayout::g_os_iyx_osv16)
        return {1, 16};
    else if (l == WeightsLayout::os_iyx_osv32 || l == WeightsLayout::g_os_iyx_osv32 || l == WeightsLayout::os_iyx_osv32__ai32)
        return {1, 32};
    else if (l == WeightsLayout::os_is_zyx_osv32_isv16 || l == WeightsLayout::g_os_zyx_is_osv32_isv16)
        return {16, 32};
    else if (l == WeightsLayout::os_is_zyx_osv64_isv16)
        return {16, 64};
    else if (l == WeightsLayout::g_os_zyx_is_osv16_isv32)
        return {32, 16};
    else if (l == WeightsLayout::g_os_zyx_is_osv32_isv32)
        return {32, 32};
    else
        return {1, 1};
}

static inline bool IsOsvFirst(WeightsLayout l) {
    if (l == WeightsLayout::os_is_yx_isv16_osv16 || l == WeightsLayout::os_is_zyx_isv16_osv16 ||
        l == WeightsLayout::g_os_is_yx_isv16_osv16 || l == WeightsLayout::g_os_is_zyx_isv16_osv16 ||
        l == WeightsLayout::os_iyx_osv16 || l == WeightsLayout::g_os_iyx_osv16||
        l == WeightsLayout::os_iyx_osv32 || l == WeightsLayout::g_os_iyx_osv32 ||
        l == WeightsLayout::os_iyx_osv32__ai32 || l == WeightsLayout::is_os_yx_isv16_osv16 ||
        l == WeightsLayout::is_os_zyx_isv16_osv16 || l == WeightsLayout::g_is_os_yx_isv16_osv16 ||
        l == WeightsLayout::g_is_os_zyx_isv16_osv16)
        return true;
    else
        return false;
}

static inline size_t GetOptimalSize(size_t val, std::vector<size_t> optimal_sizes) {
    for (auto& s : optimal_sizes)
        if (val % s == 0)
            return s;
    return 1;
}

ReorderWeightsOpt::DispatchData ReorderWeightsOpt::SetDefault(
    const reorder_weights_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.output;
    const auto output_layout = output.GetLayout();
    const auto subgroup_size = 16;
    const auto ifm_block_supported = (output_layout != WeightsLayout::os_iyx_osv16 &&
                                      output_layout != WeightsLayout::os_iyx_osv32 &&
                                      output_layout != WeightsLayout::g_os_iyx_osv16 &&
                                      output_layout != WeightsLayout::g_os_iyx_osv32 &&
                                      output_layout != WeightsLayout::os_iyx_osv32__ai32);

    const auto osv_first = IsOsvFirst(output_layout);
    const auto ofm_block = (osv_first) ? subgroup_size : GetOptimalSize(output.OFM().v, preferred_sizes);
    const auto ifm_block = (osv_first) ? ifm_block_supported ? GetOptimalSize(output.IFM().v, preferred_sizes) : 1
                                       : subgroup_size;

    if (osv_first) {
        dispatchData.gws = { output.G().v * (output.IFM().v / ifm_block),
                             output.Z().v * output.Y().v * output.X().v,
                             Align(output.OFM().v, ofm_block) };
    } else {
        dispatchData.gws = { output.G().v * (output.OFM().v / ofm_block),
                             output.Z().v * output.Y().v * output.X().v,
                             Align(output.IFM().v, ifm_block) };
    }

    dispatchData.lws = { 1, 1, 16 };

    return dispatchData;
}

JitConstants ReorderWeightsOpt::GetJitConstants(const reorder_weights_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    const auto& output = params.output;
    const auto subgroup_size = 16;
    const auto ifm_block_supported = (output.GetLayout() != WeightsLayout::os_iyx_osv16 &&
                                      output.GetLayout() != WeightsLayout::os_iyx_osv32 &&
                                      output.GetLayout() != WeightsLayout::g_os_iyx_osv16 &&
                                      output.GetLayout() != WeightsLayout::g_os_iyx_osv32 &&
                                      output.GetLayout() != WeightsLayout::os_iyx_osv32__ai32);

    const auto slice_sizes = GetSliceSizes(output.GetLayout());
    const auto osv_first = IsOsvFirst(output.GetLayout());
    const auto leftovers = (osv_first) ? output.OFM().v % subgroup_size : output.IFM().v % subgroup_size;
    const auto ofm_block = (osv_first) ? subgroup_size : GetOptimalSize(output.OFM().v, preferred_sizes);
    const auto ifm_block = (osv_first) ? ifm_block_supported ? GetOptimalSize(output.IFM().v, preferred_sizes) : 1
                                       : subgroup_size;

    jit.AddConstant(MakeJitConstant("IFM_SIZE", slice_sizes.first));
    jit.AddConstant(MakeJitConstant("OFM_SIZE", slice_sizes.second));
    jit.AddConstant(MakeJitConstant("OSV_FIRST", osv_first));
    jit.AddConstant(MakeJitConstant("IFM_BLOCK_SIZE", ifm_block));
    jit.AddConstant(MakeJitConstant("OFM_BLOCK_SIZE", ofm_block));

    if (leftovers)
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", leftovers));

    return jit;
}

bool ReorderWeightsOpt::Validate(const Params& params) const {
    const auto& p = static_cast<const reorder_weights_params&>(params);
    const auto& input = p.input;
    const auto& output = p.output;

    if (input.GroupedLayout() != output.GroupedLayout()) {
        return false;
    }

    if (input.GetDims().size() != output.GetDims().size()) {
        return false;
    }

    return true;
}

KernelsData ReorderWeightsOpt::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

KernelsPriority ReorderWeightsOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
