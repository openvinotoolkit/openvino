// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_gemv.h"
#include "fully_connected_kernel_bf_tiled.h"

#include "common_types.h"
#include "kernel_selector_utils.h"
#include "swiglu/swiglu_kernel_base.h"

using namespace kernel_selector::fc_kernel_bf_tiled_utils;
static constexpr size_t simd = 16;

namespace kernel_selector {
ParamsKey FullyConnected_GEMV::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT4);
    k.EnableInputWeightsType(WeightsType::UINT4);
    k.EnableInputWeightsType(WeightsType::UINT2);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDynamicShapesSupport();
    k.EnableWeightsCompression();
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableDifferentTypes();

    return k;
}

DeviceFeaturesKey FullyConnected_GEMV::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_broadcast();
    k.requires_blocked_read_write();
    k.requires_blocked_read_write_char();
    k.requires_blocked_read_write_short();

    return k;
}

bool FullyConnected_GEMV::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    const auto& input = fc_params.inputs[0];
    const auto& output = fc_params.outputs[0];
    const auto& weights = fc_params.weights;

    if (!fc_params.compressed) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    const size_t scale_group_size = weights.IFM().v / fc_params.decompression_scale.Feature().v;
    if (scale_group_size == 0 || scale_group_size % 16 != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Data type re-check: only support f16:{int4,uint4,uint2}:f16
    if (input.GetDType() != Datatype::F16 || (output.GetDType() != Datatype::F16 && output.GetDType() != Datatype::F32) ||
        (weights.GetDType() != WeightsType::INT4 && weights.GetDType() != WeightsType::UINT4 &&
         weights.GetDType() != WeightsType::UINT2)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (weights.GetDType() == WeightsType::UINT2) {
        // u2 is implemented for the OSV16 layout only: one per-lane 16-byte block read covers
        // 64 K values there, which the OSV32/OSV64 k-pair packings do not line up with.
        const auto wl_u2 = weights.GetLayout();
        if (wl_u2 != WeightsLayout::oiyx && wl_u2 != WeightsLayout::os_iyx_osv16) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
        if (weights.IFM().v % 64 != 0 || weights.OFM().v % 16 != 0) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
        // Grouped scales must also land on a whole number of 64-value windows; the
        // channel-wise case takes the SINGLE_GROUP_NUM path and re-splits K by 128.
        const size_t sgs_u2 = weights.IFM().v / fc_params.decompression_scale.Feature().v;
        if (sgs_u2 != weights.IFM().v && sgs_u2 % 64 != 0) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
    }

    // Only support vector data as input, the data size should be aligned by 16 elements
    auto input_size = get_input_bf_size(fc_params);
    if (input_size.first > 1 || input_size.second == 0 || input_size.second % 16 != 0 || weights.IFM().v % 16 != 0) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    auto wl = weights.GetLayout();
    auto wo = weights.OFM().v;

    const auto& fc_input = fc_params.inputs[0];
    if (is_swiglu_fused(fc_params)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (input_size.first != 0 && fc_input.is_dynamic()) {
        if (input_size.first != 1) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
        if ((wl != WeightsLayout::os_is_yx_osv32_isv2 || wo % 32 != 0) &&
            (wl != WeightsLayout::os_is_yx_osv64_isv2 || wo % 64 != 0) &&
            (wl != WeightsLayout::os_iyx_osv16 || wo % 16 != 0)) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }
    }

    if (input.GetLayout() == DataLayout::bfyx) {
        // Padding on input is not supported.
        if (input.X().pad.Total() != 0)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        if (input.Y().pad.Total() != 0)
            DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // We don't support 4d output
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx && fc_params.outputs[0].X().v > 1)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    return true;
}

FullyConnected_GEMV::DispatchData FullyConnected_GEMV::SetDefault(const fully_connected_params& params,
                                                                  int,
                                                                  int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params);

    std::vector<size_t> global = {params.weights.OFM().v, 1, 16};
    if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
        global[0] = params.weights.OFM().v;
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
        global[0] = params.weights.OFM().v / 2;
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
        global[0] = params.weights.OFM().v / 4;
    }

    dispatchData.gws = global;
    dispatchData.lws = {16, 1, 16};

    return dispatchData;
}

KernelsPriority FullyConnected_GEMV::GetKernelsPriority(const Params& params) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    auto priority = FORCE_PRIORITY_9;
    if (!params.is_shape_agnostic) {
        auto output_size = get_output_aligned_bf_size(fc_params, false);
        if (output_size.first == 1 && output_size.second % 16 == 0) {
            priority = FORCE_PRIORITY_2;
        }
    }
    return priority;
}

JitConstants FullyConnected_GEMV::GetJitConstants(const fully_connected_params& params,
                                                  const FullyConnectedKernelBase::DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    // TODO: SWIGLU support
    // if (is_swiglu_fused(params)) {
    //     auto split_length = params.fused_ops[0].GetOpParams<swiglu_fuse_params>()->split_length;
    //     auto gate_idx = params.fused_ops[0].GetOpParams<swiglu_fuse_params>()->gate_idx;
    //     jit.AddConstant(MakeJitConstant("SWIGLU_LENGTH", split_length));
    //     jit.AddConstant(MakeJitConstant("SWIGLU_SPLIT_TO_GLU_IDX", gate_idx));
    // }

    if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
        jit.AddConstant(MakeJitConstant("FILTER_LAYOUT_OS_IS_YX_TYPE", 0));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
        jit.AddConstant(MakeJitConstant("FILTER_LAYOUT_OS_IS_YX_TYPE", 1));
    } else if (params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
        jit.AddConstant(MakeJitConstant("FILTER_LAYOUT_OS_IS_YX_TYPE", 2));
    } else {
        OPENVINO_THROW("GEMV doesn't support this weights layout: ", params.weights.GetLayout());
    }

    if (params.weights.GetDType() == WeightsType::UINT4) {
        jit.AddConstant(MakeJitConstant("WEI_UINT4", 1));
        jit.AddConstant(MakeJitConstant("WEI_U2", 0));
    } else if (params.weights.GetDType() == WeightsType::INT4) {
        jit.AddConstant(MakeJitConstant("WEI_UINT4", 0));
        jit.AddConstant(MakeJitConstant("WEI_U2", 0));
    } else if (params.weights.GetDType() == WeightsType::UINT2) {
        // unsigned like u4, but 4 values per byte; WEI_UINT4 stays 0 so the signed-int4
        // sign-extension branch is not taken either
        jit.AddConstant(MakeJitConstant("WEI_UINT4", 0));
        jit.AddConstant(MakeJitConstant("WEI_U2", 1));
    } else {
        OPENVINO_THROW("GEMV only supports INT4, UINT4 and UINT2, doesn't support ",
                       static_cast<size_t>(params.weights.GetDType()));
    }

    jit.AddConstant(MakeJitConstant("SIMD", simd));
    jit.AddConstant(MakeJitConstant("WEIGHTS_K", params.weights.IFM().v));
    jit.AddConstant(MakeJitConstant("WEIGHTS_N", params.weights.OFM().v));

    auto activation_dt = GetActivationType(params);
    // Activation is computed in F32, so we need to convert it to F32
    if (activation_dt == Datatype::F16) {
        activation_dt = Datatype::F32;
    }
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));

    if (!params.fused_ops.empty() && !is_swiglu_fused(params)) {
        std::vector<std::string> idx_order = {"0", "0", "(cur_n + 16 * i)", "0"};
        if (params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) {
            idx_order = {"0", "0", "(cur_n + i)", "0"};
        }
        FusedOpsConfiguration conf_vec = {"_VEC", idx_order, "sum_value[i]", activation_dt, 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_vec}));
    }
    return jit;
}

KernelsData FullyConnected_GEMV::GetTunedKernelsDataByIndex(const Params& params, const int autoTuneIndex) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto output_f = get_output_aligned_bf_size(fc_params, false).second;

    WeightsLayout weights_layout = WeightsLayout::os_iyx_osv16;
    if (fc_params.weights.GetDType() == WeightsType::UINT2) {
        // u2 has one implemented layout here; skip the int4 heuristics entirely so a
        // horizontal or swiglu-shaped weight cannot land on an osv32/osv64 packing the
        // u2 path does not decode.
        weights_layout = WeightsLayout::os_iyx_osv16;
    } else if (is_swiglu_fused(fc_params)) {
        weights_layout = WeightsLayout::os_is_yx_osv32_isv2;
    } else if (fc_params.compressed && fc_params.inputs[0].GetDType() == Datatype::F16 &&
               (fc_params.weights.GetLayout() == WeightsLayout::oiyx ||
                fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) &&
               (fc_params.weights.GetDType() == WeightsType::INT4 ||
                fc_params.weights.GetDType() == WeightsType::UINT4) &&
               is_weight_horizontal(fc_params, output_f)) {
        // Large N + small K case (horizontal weight) to use osv64_isv2
        weights_layout = WeightsLayout::os_is_yx_osv64_isv2;
    } else if (fc_params.compressed && fc_params.inputs[0].GetDType() == Datatype::F16 &&
               (fc_params.weights.GetDType() == WeightsType::INT4 ||
                fc_params.weights.GetDType() == WeightsType::UINT4) &&
               (fc_params.weights.GetLayout() == WeightsLayout::oiyx ||
                fc_params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) &&
               is_weight_vertical(fc_params, output_f)) {
        // Large K + Small N case (vertical weight)  to use osv16
        weights_layout = WeightsLayout::os_iyx_osv16;
    } else if (fc_params.compressed &&
               fc_params.inputs[0].GetDType() == Datatype::F16
               // ioyx => os_is_yx_osv32_isv2 is not supported yet
               && (fc_params.weights.GetLayout() == WeightsLayout::oiyx ||
                   fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) &&
               (fc_params.weights.GetDType() == WeightsType::INT4 ||
                fc_params.weights.GetDType() == WeightsType::UINT4)) {
        weights_layout = WeightsLayout::os_iyx_osv16;
    }

    // Keeping an already-reordered layout must not drag u2 onto osv32/osv64, which it
    // cannot decode. Validate() already rejects those, so this is belt and braces.
    if (fc_params.weights.GetDType() != WeightsType::UINT2 &&
        ((fc_params.weights.GetLayout() == WeightsLayout::os_iyx_osv16) ||
         (fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) ||
         (fc_params.weights.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2))) {
        weights_layout = fc_params.weights.GetLayout();
    }

    KernelsData kernels_data;
    kernels_data = GetCommonKernelsData(params,
                                        fc_params.inputs[0].GetLayout(),
                                        weights_layout,
                                        EXE_MODE_DEFAULT,
                                        autoTuneIndex,
                                        0);
    return kernels_data;
}

KernelsData FullyConnected_GEMV::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    KernelsData kds = GetTunedKernelsDataByIndex(params, -1);
    if (!kds.empty()) {
        res.emplace_back(kds[0]);
    }

    return res;
}

}  // namespace kernel_selector
