// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_base.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <vector>
#include <algorithm>

namespace kernel_selector {
JitConstants FullyConnectedKernelBase::GetJitConstants(const fully_connected_params& params,
                                                       const FullyConnectedKernelBase::DispatchData&) const {
    JitConstants jit = WeightBiasKernelBase::GetJitConstants(params);
    const auto& input = params.inputs[0];
    if (input.is_dynamic()) {
        DimensionAccessHelperJit dims(input);
        jit.AddConstant(MakeJitConstant("INPUT0_ELEMENTS_COUNT", toVectorMulString({dims.x(), dims.y(), dims.z(), dims.w(), dims.f()})));
    } else {
        const auto x_size = input.LogicalSize() / input.Batch().v;
        jit.AddConstant(MakeJitConstant("INPUT0_ELEMENTS_COUNT", x_size));
    }

    if (params.compressed) {
        jit.AddConstants({MakeJitConstant("COMPRESSED_WEIGHTS", 1)});
        if (params.weights.GetDType() == WeightsType::INT8 || params.weights.GetDType() == WeightsType::UINT8) {
            jit.AddConstants({MakeJitConstant("COMPRESSED_WEIGHTS_INT8", 1)});
        } else if (params.weights.GetDType() == WeightsType::INT4 || params.weights.GetDType() == WeightsType::UINT4) {
            jit.AddConstants({MakeJitConstant("COMPRESSED_WEIGHTS_INT4", 1)});
        }

        const size_t scale_groups_num = params.decompression_scale.Feature().v;
        const size_t scale_group_size = params.weights.IFM().v / params.decompression_scale.Feature().v;
        jit.AddConstants({MakeJitConstant("DECOMPRESSION_SCALE_TERM", 1)});
        jit.AddConstants({MakeJitConstant("DECOMPRESSION_SCALE", params.decompression_scale)});
        jit.AddConstants({MakeJitConstant("DECOMPRESSION_SCALE_GROUPS_NUM", scale_groups_num)});
        jit.AddConstants({MakeJitConstant("DECOMPRESSION_SCALE_GROUP_SIZE", scale_group_size)});
        if (params.has_decompression_zp) {
            jit.AddConstants({MakeJitConstant("DECOMPRESSION_ZP_TERM", 1)});
            if (params.scalar_zp) {
                jit.AddConstants({MakeJitConstant("DECOMPRESSION_ZP_VALUE", params.zp_value)});
                jit.AddConstants({MakeJitConstant("DECOMPRESSION_ZP_SCALAR", 1)});
            } else {
                const size_t zp_groups_num = params.decompression_zero_point.Feature().v;
                const size_t zp_group_size = params.weights.IFM().v / params.decompression_zero_point.Feature().v;
                jit.AddConstants({MakeJitConstant("DECOMPRESSION_ZP", params.decompression_zero_point)});
                jit.AddConstants({MakeJitConstant("DECOMPRESSION_ZP_GROUPS_NUM", zp_groups_num)});
                jit.AddConstants({MakeJitConstant("DECOMPRESSION_ZP_GROUP_SIZE", zp_group_size)});
            }
        }
    }

    return jit;
}

FullyConnectedKernelBase::DispatchData FullyConnectedKernelBase::SetDefault(const fully_connected_params& params,
                                                                            int, int /*kernel_number*/) const {
    DispatchData dispatchData;

    // Determine global work sizes.
    dispatchData.gws = { params.outputs[0].LogicalSize(), 1, 1 };

    // Find largest positive local work size that is divider for global work size.
    dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
    while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
        --dispatchData.lws[0];
    }
    dispatchData.lws[1] = dispatchData.lws[2] = 1;

    return dispatchData;
}

void FullyConnectedKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const fully_connected_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData FullyConnectedKernelBase::GetCommonKernelsData(const Params &params,
                                                           DataLayout dl,
                                                           WeightsLayout wl,
                                                           const std::string exeMode,
                                                           int autoTuneIndex,
                                                           int kernel_number) const {
    if (!Validate(params)) {
        return KernelsData();
    }

    const auto& orgParams = static_cast<const fully_connected_params&>(params);

    bool bProperInput = orgParams.inputs[0].GetLayout() == dl;
    if (!bProperInput && !orgParams.inputs[0].PitchesDifferFromLogicalDims()) {
        bProperInput = (dl == DataLayout::fb && orgParams.inputs[0].GetLayout() == DataLayout::fyxb) ||
                       (dl == DataLayout::bf && orgParams.inputs[0].GetLayout() == DataLayout::bfyx);
    }

    KernelData kd = KernelData::Default<fully_connected_params>(params);
    GetUpdateDispatchDataFunc(kd);
    fully_connected_params& newParams = *static_cast<fully_connected_params*>(kd.params.get());

    if (!bProperInput) {
        newParams.inputs[0] = newParams.inputs[0].TransformIgnorePadding(dl);
        kd.reorderInput = true;
    }

    bool succeed = UpdateWeightsParams(newParams,
                                       wl,
                                       kd.weightsReorderParams,
                                       GetSupportedKey());

    if (!succeed) {
        return {};
    }

    kd.kernels.resize(1);

    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, kernel_number);

    const DispatchData dispatchData = SetDefault(newParams, autoTuneIndex, kernel_number);
    auto cldnn_jit = GetJitConstants(newParams, dispatchData);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    int inputs_count = 1;
    if (newParams.compressed) {
        inputs_count++;
        if (newParams.has_decompression_zp && !newParams.scalar_zp)
            inputs_count++;
    }

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     exeMode,
                     true,
                     !orgParams.bias.empty(),
                     inputs_count,
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     orgParams.is_shape_agnostic);

    // TODO Pass estimated time only through DispatchData
    kd.autoTuneIndex = autoTuneIndex;
    return {kd};
}

std::string FullyConnectedKernelBase::GetAutoTuneOptions(int autoTuneIndex) const {
    if ((autoTuneIndex >= 0) && (autoTuneIndex < static_cast<int>(autoTuneOptions.size()))) {
        return autoTuneOptions[autoTuneIndex];
    }

    return EXE_MODE_DEFAULT;
}

KernelsData FullyConnectedKernelBase::GetTunedKernelsDataByIndex(const Params &params,
                                                                 DataLayout dl,
                                                                 WeightsLayout wl,
                                                                 const int autoTuneIndex) const {
    return GetCommonKernelsData(params,
                                dl,
                                wl,
                                GetAutoTuneOptions(autoTuneIndex),
                                autoTuneIndex);
}


JitConstants FullyConnectedKernelBase::GetFusedPrimitivesJitConstants(const fully_connected_params&, const DispatchData&) const {
    return {};
}

bool FullyConnectedKernelBase::Validate(const Params& p) const {
    const fully_connected_params& params = static_cast<const fully_connected_params&>(p);

    if (params.GetType() != KernelType::FULLY_CONNECTED) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

Datatype FullyConnectedKernelBase::GetAccumulatorType(const fully_connected_params& params) const {
    if (params.quantization != QuantizationType::NONE)
        return Datatype::INT32;

    auto in_dt = params.inputs[0].GetDType();
    auto wei_dt = params.weights.GetDType();

    auto quantized_inputs = in_dt == Datatype::UINT8 || in_dt == Datatype::INT8;
    auto quantized_weights = wei_dt == WeightsType::UINT8 || wei_dt == WeightsType::INT8;

    // This case should be always false, because quantization type is not NONE
    if (quantized_inputs && quantized_weights)
        return Datatype::INT32;

    // If we either weights or input is quantized, then we use fp32 accumulator to avoid fp16 overflow
    if ((quantized_inputs || quantized_weights) && !params.compressed)
        return Datatype::F32;

    return in_dt;
}

Datatype FullyConnectedKernelBase::GetActivationType(const fully_connected_params& params) const {
    auto in_dt = params.inputs[0].GetDType();
    auto wei_dt = params.weights.GetDType();
    auto out_dt = params.outputs[0].GetDType();

    auto quantized_inputs = in_dt == Datatype::UINT8 || in_dt == Datatype::INT8;
    auto quantized_weights = wei_dt == WeightsType::UINT8 || wei_dt == WeightsType::INT8;

    if (params.quantization != QuantizationType::NONE || quantized_inputs || quantized_weights)
        return Datatype::F32;

    auto output_is_int8 = out_dt == Datatype::UINT8 || out_dt == Datatype::INT8;
    auto input_is_fp = in_dt == Datatype::F32 || in_dt == Datatype::F16;

    if (output_is_int8 && input_is_fp)
        return in_dt;

    return GetUnitType(params);
}

}  // namespace kernel_selector
