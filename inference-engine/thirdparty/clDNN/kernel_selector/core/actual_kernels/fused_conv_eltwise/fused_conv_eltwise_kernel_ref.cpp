/*
// Copyright (c) 2019 Intel Corporation
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
*/

#include "fused_conv_eltwise_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey fused_conv_eltwise_kernel_ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfzyx_f16);
    k.EnableOutputLayout(DataLayout::bfzyx_f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableNonBiasTerm();
    k.EnableBiasPerFeature();
    k.EnableBatching();
    k.EnableFusedConvEltwInt8Quantization();
    k.EnableFusedConvEltwOutputCalibration();
    k.DisableTuning();
    k.EnableFusedConvEltwiseRWOutOpt();
    return k;
}

Datatype fused_conv_eltwise_kernel_ref::GetUnitType(const base_params& params) const {
    const fused_conv_eltwise_params& cp = static_cast<const fused_conv_eltwise_params&>(params);
    (void)cp;
    // if (cp.inputs[0].GetDType() == Datatype::F32)
    //     return Datatype::F32;
    // FIXME: Proper logic.
    return Datatype::INT32;
}

bool fused_conv_eltwise_kernel_ref::Validate(const Params& p, const optional_params& o) const {
    if (!fused_conv_eltwise_kernel_base::Validate(p, o) || !FusedConvolutionEltwiseCheckInput(p, o)) {
        return false;
    }
    const fused_conv_eltwise_params& cp = static_cast<const fused_conv_eltwise_params&>(p);
    (void)cp;

    // TODO: ConvolutionKernel_bfyx_Ref::Supports call?

    return true;
}

JitConstants fused_conv_eltwise_kernel_ref::GetJitConstants(const fused_conv_eltwise_params& params,
                                                            const DispatchData& kd) const {
    auto jit = fused_conv_eltwise_kernel_base::GetJitConstants(params, kd);

    // Create an ACTIVATION macro accepting type parameter - we don't have a
    // single UNIT_TYPE for the whole kernel.
    //
    // TODO: This gives both ACTIVATION and ACTIVATION_TYPED. Should we
    // factor that out into a virtual function to avoid creation of similar
    // yet distinct macros?
    jit.Merge(MakeActivationJitConstants(params.conv.activations, "_CONV_TYPED", true));
    jit.Merge(MakeActivationJitConstants(params.activations, "_ELTW_TYPED", true));
    // Needs to be done on host to get _MAX_VAL/_MIN_VAL/TO_TYPE macros
    // available (will be used in the activations).
    //
    // TODO: Should it be done for all the kernels? Might even be done
    // directly in the OpenCL include, as opposite to jitting. On the other
    // hand, going through jit ensures we are in sync with the
    // MakeTypeJitConstants implementation.
    jit.Merge(MakeTypeJitConstants(Datatype::F32, "float"));
    jit.Merge(MakeTypeJitConstants(Datatype::INT32, "int"));

    if (params.non_conv_scale != 1.0f)
        jit.AddConstant(MakeJitConstant("NON_CONV_SCALE", params.non_conv_scale));
    return jit;
}
KernelsData fused_conv_eltwise_kernel_ref::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}
fused_conv_eltwise_kernel_base::DispatchData fused_conv_eltwise_kernel_ref::SetDefault(
    const fused_conv_eltwise_params& arg,
    int autoTuneIndex) const {
    DispatchData runInfo = fused_conv_eltwise_kernel_base::SetDefault(arg, autoTuneIndex);

    // FIXME: This code copies one of the "if" condition branches from the
    // fused_conv_eltwise_kernel_base::SetDefault. However, the right place
    // to calculate the GWS/LWS is HERE, not in the base class, because it's
    // the propery of the implementation, not the property of the layout.
    //
    // This comment is here to ensure that nobody would remove the
    // calculation from here. Remove it once the base class/architecture is
    // properly fixed and different aspects of the "SetDefault" method are
    // properly decoupled.
    const auto& out = arg.output;
    std::vector<size_t> global;
    global = {out.X().v, out.Y().v, out.Feature().v * out.Batch().v};
    auto local = GetOptimalLocalWorkGroupSizes(global);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}
}  // namespace kernel_selector
