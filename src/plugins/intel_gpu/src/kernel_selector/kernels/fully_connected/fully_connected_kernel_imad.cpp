// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "fully_connected_kernel_imad.h"

// IMAD Fully_Connected primitive implementation.
// Limitations are:
// 1. Input = F x 1 x 1 with Filter = 1 x 1
// 2. No data padding

namespace kernel_selector {

ParamsKey FullyConnectedKernelIMAD::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);

    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey FullyConnectedKernelIMAD::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_reqd_subgroup_size();
    k.requires_subgroup_shuffle();
    k.requires_blocked_read_write();

    return k;
}

FullyConnectedKernelIMAD::Parent::DispatchData FullyConnectedKernelIMAD::SetDefault(const fully_connected_params& params, int, int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params);

    if (!params.has_dynamic_tensors()) {
        auto tuning_data = GetTuningParams(params);
        if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
            dispatchData.gws[0] = RoundUp(params.outputs[0].Y().v, tuning_data.sub_group_size * tuning_data.tile_ofm) /
                                  tuning_data.tile_ofm * tuning_data.slm_div_factor;
            dispatchData.gws[1] = params.outputs[0].Batch().v;
            dispatchData.gws[2] = params.outputs[0].Feature().v / tuning_data.tile_batch;
        } else {
            dispatchData.gws[0] = RoundUp(params.outputs[0].Feature().v, tuning_data.sub_group_size * tuning_data.tile_ofm) /
                                  tuning_data.tile_ofm * tuning_data.slm_div_factor;
            dispatchData.gws[1] = params.outputs[0].Batch().v / tuning_data.tile_batch;
            dispatchData.gws[2] = 1;
        }

        dispatchData.lws[0] = tuning_data.work_group_size;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

bool FullyConnectedKernelIMAD::Validate(const Params& params) const {
    if (!Parent::Validate(params)) {
        return false;
    }

    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    const auto& in = fc_params.inputs[0];
    const auto& wei = fc_params.weights;
    auto out_l = fc_params.outputs[0].GetLayout();

    // Dynamic kernel doesn't support dynamic weights
    if (fc_params.is_shape_agnostic && in.is_dynamic()) {
        if ((out_l == DataLayout::bfyx && in.Y().v == 0) ||
            (out_l == DataLayout::bf && in.Feature().v == 0))
            return false;
    }

    if ((in.X().pad.before != 0) || (in.X().pad.after != 0) ||
        (in.Y().pad.before != 0) || (in.Y().pad.after != 0)) {
        // Padding is not supported
        return false;
    }

    if (out_l == DataLayout::bfyx) {
        // We don't support 4d output
        if (in.X().v > 1)
            return false;
    } else {
        if (in.X().v * in.Y().v * wei.X().v * wei.Y().v != 1) {
            // Currently only Input = F x 1 x 1 with Filter = 1 x 1 is supported
            return false;
        }
    }

    return true;
}

float FullyConnectedKernelIMAD::EstimateOccupancy(const fully_connected_params& params, size_t tile_ofm, size_t tile_batch, size_t slm_div_factor) const {
    FullyConnectedTuningData tuning_data;

    auto of_num = params.outputs[0].Feature().v;
    auto ob_num = params.outputs[0].Batch().v;

    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        ob_num *= of_num;
        of_num = params.outputs[0].Y().v;
    }

    size_t blocks_f = RoundUp(of_num, tuning_data.sub_group_size) / tile_ofm * slm_div_factor / tuning_data.sub_group_size;
    size_t blocks_b = ob_num / tile_batch;

    auto threads = blocks_f * blocks_b;

    return static_cast<float>(threads) / static_cast<float>(params.engineInfo.maxThreadsPerDevice);
}

FullyConnectedKernelIMAD::FullyConnectedTuningData FullyConnectedKernelIMAD::GetTuningParams(const fully_connected_params& params) const {
    FullyConnectedTuningData tuning_data;

    auto of_num = params.outputs[0].Feature().v;
    auto ob_num = params.outputs[0].Batch().v;
    auto if_num = params.inputs[0].Feature().v;
    auto ib_num = params.inputs[0].Batch().v;
    auto tile_batch_max_size = ob_num;

    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        tile_batch_max_size = of_num;
        ob_num *= of_num;
        of_num = params.outputs[0].Y().v;
        ib_num *= if_num;
        if_num = params.inputs[0].Y().v;
    }

    // In most cases SIMD8 works faster than SIMD16
    tuning_data.sub_group_size = IsSIMDSizeSupported(params.engineInfo, 8) ? 8 : 16;

    if (!params.is_shape_agnostic) {
        auto mk_size = if_num * ib_num;
        auto mn_size = of_num * ob_num;

        // Known cases where simd16 works better than simd8
        bool simd16_is_faster = mk_size >= 1000 * 1024 && mn_size >= 1000 * 1024;
        simd16_is_faster |= mk_size == 128 * 768 && mn_size == 128 * 3072;

        // Some specific HW doesn't support SIMD8, force SIMD16 to respect this HW
        // For other SIMD16 exceptions check that if_num is divided by 64 (SIMD16 * ISV4) because
        // if there are leftovers then SIMD8 is more preferrable
        if (!IsSIMDSizeSupported(params.engineInfo, 8) || (simd16_is_faster && if_num % 64 == 0)) {
            tuning_data.sub_group_size = 16;
        }
    }
    tuning_data.tile_ofm = 2;

    tuning_data.tile_batch = tuning_data.sub_group_size == 8 ? 16 : 8;
    if (!params.has_dynamic_tensors()) {
        while (tile_batch_max_size % tuning_data.tile_batch != 0)
            tuning_data.tile_batch--;
    }

    size_t sub_group_pack_size = tuning_data.sub_group_size * tuning_data.pack_size;
    tuning_data.in_f_blocks_number = CeilDiv(if_num, sub_group_pack_size);

    tuning_data.slm_div_factor = 1;
    tuning_data.work_group_size = tuning_data.slm_div_factor * tuning_data.sub_group_size;
    tuning_data.work_groups_number = tuning_data.in_f_blocks_number / tuning_data.slm_div_factor;

    return tuning_data;
}

JitConstants FullyConnectedKernelIMAD::GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    auto tuning_data = GetTuningParams(params);

    auto of_num = params.outputs[0].Feature().v;
    auto if_num = params.inputs[0].Feature().v;
    auto output_3d = params.outputs[0].GetLayout() == DataLayout::bfyx;

    if (output_3d) {
        of_num = params.outputs[0].Y().v;
        if_num = params.inputs[0].Y().v;
    }

    auto has_ifm_leftovers = (if_num % (tuning_data.pack_size * tuning_data.sub_group_size)) != 0;
    auto has_ofm_leftovers = (of_num % (tuning_data.tile_ofm * tuning_data.sub_group_size)) != 0;

    jit.AddConstant(MakeJitConstant("SLM_DIV_FACTOR", tuning_data.slm_div_factor));
    jit.AddConstant(MakeJitConstant("SIMD_SIZE", tuning_data.sub_group_size));
    jit.AddConstant(MakeJitConstant("PACK_SIZE", tuning_data.pack_size));
    jit.AddConstant(MakeJitConstant("WORK_GROUP_SIZE", tuning_data.work_group_size));
    jit.AddConstant(MakeJitConstant("WORK_GROUPS_NUMBER", tuning_data.work_groups_number));
    jit.AddConstant(MakeJitConstant("TILE_OFM", tuning_data.tile_ofm));
    jit.AddConstant(MakeJitConstant("TILE_BATCH", tuning_data.tile_batch));
    jit.AddConstant(MakeJitConstant("HAS_IFM_LEFTOVERS", has_ifm_leftovers));
    jit.AddConstant(MakeJitConstant("HAS_OFM_LEFTOVERS", has_ofm_leftovers));
    jit.AddConstant(MakeJitConstant("OUTPUT_3D", output_3d));
    jit.AddConstant(MakeJitConstant("OF_NUMBER", of_num));
    jit.AddConstant(MakeJitConstant("IF_NUMBER", if_num));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);

        std::vector<std::string> idx_order_slm_split, idx_order_batch_scalar, idx_order_batch_vec;
        if (output_3d) {
            idx_order_slm_split = { "batch", "skip_f", "feature", "0" };
            idx_order_batch_scalar = { "batch", "skip_f", "(feature + SIMD_SIZE * of_idx)", "0" };
            idx_order_batch_vec = { "batch", "(skip_f + ob_idx)", "(feature + SIMD_SIZE * of_idx)", "0" };
        } else {
            idx_order_slm_split = { "batch", "feature", "0", "0" };
            idx_order_batch_scalar = { "batch", "(feature + SIMD_SIZE * of_idx)", "0", "0" };
            idx_order_batch_vec = { "(batch + ob_idx)", "(feature + SIMD_SIZE * of_idx)", "0", "0" };
        }

        FusedOpsConfiguration conf_slm_split = { "_SLM_SPLIT",
                                                 idx_order_slm_split,
                                                 "dequantized",
                                                 input_dt,
                                                 1 };
        FusedOpsConfiguration conf_batch_scalar = { "_BATCH_SCALAR",
                                                    idx_order_batch_scalar,
                                                    "dequantized[of_idx]",
                                                    input_dt,
                                                    1 };
        FusedOpsConfiguration conf_batch_vec = { "_BATCH_VEC",
                                                 idx_order_batch_vec,
                                                 "dequantized[ob_idx][of_idx]",
                                                 input_dt,
                                                 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_slm_split, conf_batch_scalar, conf_batch_vec }));
    }

    return jit;
}

KernelsData FullyConnectedKernelIMAD::GetKernelsData(const Params& params) const {
    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto tuning_data = GetTuningParams(fc_params);
    auto& input = fc_params.inputs[0];

    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    input.GetLayout(),
                                                    tuning_data.sub_group_size == 8 ?
                                                    WeightsLayout::os_is_yx_osv8_isv4 :
                                                    WeightsLayout::os_is_yx_osv16_isv4,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

KernelsPriority FullyConnectedKernelIMAD::GetKernelsPriority(const Params& params) const {
    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto tuning_data = GetTuningParams(fc_params);
    auto output_3d = fc_params.outputs[0].GetLayout() == DataLayout::bfyx;

    float estimated_time = FORCE_PRIORITY_1;
    if (output_3d) {
        estimated_time = tuning_data.tile_batch > 1 ? FORCE_PRIORITY_2 : FORCE_PRIORITY_4;
    } else {
        estimated_time = tuning_data.tile_batch > 1 ? FORCE_PRIORITY_1 : FORCE_PRIORITY_7;
    }

    return estimated_time;
}
}  // namespace kernel_selector
