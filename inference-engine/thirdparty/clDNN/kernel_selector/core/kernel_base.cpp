// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_base.h"

#include <sstream>

namespace kernel_selector {
const primitive_db KernelBase::db;
thread_local size_t KernelBase::counter = 0;

std::string toString(const kernel_selector::CommonDispatchData& dispatchData) {
    auto gws = dispatchData.gws;
    auto lws = dispatchData.lws;
    std::stringstream os;
    os << "GWS(" << gws.size() << "): ";
    for (auto e : gws) {
        os << e << " ";
    }
    os << "LWS(" << lws.size() << "): ";
    for (auto e : lws) {
        os << e << " ";
    }
    return os.str();
}

void KernelBase::CheckDispatchData(const std::string& kernelName, const kernel_selector::CommonDispatchData& dispatchData) {
    if (dispatchData.gws.size() != 3 || dispatchData.lws.size() != 3)
        throw std::runtime_error("ERROR: Invalid dispatch data for kernel: " + kernelName + ": " +
                                 ": LWS and GWS size is expected to be equal to 3. Actual: " +
                                 toString(dispatchData));

    if (dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2] > 256) {
        throw std::runtime_error("ERROR: Invalid dispatch data for kernel: " + kernelName +
                                 ": LWS cannot be greater than 256. Actual: " +
                                 toString(dispatchData));
    }
    for (size_t i = 0; i < dispatchData.gws.size(); i++) {
        if (dispatchData.gws[i] == 0 || dispatchData.lws[i] == 0)
            throw std::runtime_error("ERROR: Invalid dispatch data for kernel: " + kernelName +
                                     ": Dispatch data cannot contain zeros. Actual: " +
                                     toString(dispatchData));

        if (dispatchData.gws[i] % dispatchData.lws[i] != 0)
            throw std::runtime_error("ERROR: Invalid dispatch data for kernel: " + kernelName +
                                     ": GWS must be divisible by corresponding LWS. Actual: " +
                                     toString(dispatchData));
    }
}

static bool IsTypeUsedIn(Datatype type, const base_params& params) {
    return params.output.GetDType() == type ||
           std::any_of(params.inputs.begin(), params.inputs.end(), [=](const DataTensor& input) -> bool {
               return input.GetDType() == type;
           });
}

Datatype KernelBase::GetUnitType(const base_params& params) const {
    Datatype types_prioritized[] =
        {Datatype::INT8, Datatype::F16, Datatype::INT32, Datatype::INT64, Datatype::UINT8, Datatype::UINT32};

    for (Datatype type : types_prioritized)
        if (IsTypeUsedIn(type, params))
            return type;

    return Datatype::F32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeBaseParamsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
JitConstants KernelBase::MakeBaseParamsJitConstants(const base_params& params) const {
    auto unitType = GetUnitType(params);

    JitConstants jit{
        MakeJitConstant("OUTPUT", params.output),
        MakeJitConstant("FP64_SUPPORTED", params.engineInfo.bFP64Support),
        MakeJitConstant("FP16_SUPPORTED", params.engineInfo.bFP16Support),
        MakeJitConstant("FP16_UNIT_USED", IsTypeUsedIn(Datatype::F16, params)),
        MakeJitConstant("INT8_UNIT_USED", IsTypeUsedIn(Datatype::INT8, params)),
        MakeJitConstant("INT32_UNIT_USED", IsTypeUsedIn(Datatype::INT32, params)),
        MakeJitConstant("INT64_UNIT_USED", IsTypeUsedIn(Datatype::INT64, params)),
        MakeJitConstant("UINT8_UNIT_USED", IsTypeUsedIn(Datatype::UINT8, params)),
        MakeJitConstant("UINT32_UNIT_USED", IsTypeUsedIn(Datatype::UINT32, params)),
    };

    // for activation function
    jit.Merge(MakeUnitTypeJitConstants(unitType));
    jit.Merge(MakeActivationJitConstants(params.activations, unitType));

    for (size_t i = 0; i < params.inputs.size(); i++) {
        jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i), params.inputs[i]));
    }

#ifndef NDEBUG
    jit.AddConstant(MakeJitConstant("LayerID", params.layerID));
#endif
    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeBaseParamsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
JitConstants KernelBase::MakeFusedOpsJitConstants(const kernel_selector::base_params &params,
                                                  const std::vector<FusedOpsConfiguration> &conf) const {
    JitConstants jit = {};

    if (conf.empty())
        return jit;

    try {
        for (auto& c : conf) {
            std::string fused_ops;
            std::string fused_ops_preload;
            std::string fused_ops_calc;
            std::string in_name = c.input_var_name;
            std::string out_name = "";
            Datatype in_type = c.input_dt;
            bool can_all_use_preload = true;

            for (size_t i = 0; i < params.fused_ops.size(); i++) {
                auto fused_dep_codegen = FusedOpsCodeGenerator(params.fused_ops[i]);
                jit.Merge(fused_dep_codegen.MakeLoadJitConstants(c, params.output));
                jit.Merge(fused_dep_codegen.MakeOpJitConstants(c, in_name, in_type, out_name));

                bool can_use_preload = fused_dep_codegen.CanPreloadData(c);
                can_all_use_preload &= can_use_preload;
                bool can_preload_eltwise = true;
                if (params.fused_ops[i].GetType() == FusedOpType::ELTWISE &&
                    c.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE)
                    can_preload_eltwise = false;
                fused_ops += "\\\n\tFUSED_OP" + std::to_string(i) + "_LOAD" + c.suffix;
                fused_ops += "\\\n\tFUSED_OP" + std::to_string(i) + "_ACTION" + c.suffix;
                if (can_use_preload && can_preload_eltwise)
                    fused_ops_preload += "\\\n\tFUSED_OP" + std::to_string(i) + "_LOAD" + c.suffix;
                if (c.allow_for_partial_preload && (!can_use_preload || !can_preload_eltwise))
                    fused_ops_calc += "\\\n\tFUSED_OP" + std::to_string(i) + "_LOAD" + c.suffix;
                fused_ops_calc += "\\\n\tFUSED_OP" + std::to_string(i) + "_ACTION" + c.suffix;
            }

            jit.AddConstant(MakeJitConstant("FUSED_OPS" + c.suffix, fused_ops));
            jit.AddConstant(MakeJitConstant("FUSED_OPS_PRELOAD" + c.suffix, fused_ops_preload));
            jit.AddConstant(MakeJitConstant("FUSED_OPS_CALC" + c.suffix, fused_ops_calc));
            jit.AddConstant(MakeJitConstant("FUSED_OPS_RESULT" + c.suffix, out_name));

            bool can_any_use_preload = !fused_ops_preload.empty();
            jit.AddConstant(MakeJitConstant("FUSED_OPS_CAN_USE_PRELOAD" + c.suffix,
                can_all_use_preload || (c.allow_for_partial_preload && can_any_use_preload)));
        }

        jit.Merge(MakeFusedOpsDeclsJitConstants(params, conf));
    } catch (std::exception& ex) {
        throw std::runtime_error("Fused op code generation for node " + params.layerID + " failed with error: " + ex.what());
    }

    return jit;
}

JitConstants KernelBase::MakeFusedOpsDeclsJitConstants(const kernel_selector::base_params &params,
                                                       const std::vector<FusedOpsConfiguration> &conf) const {
    JitConstants jit = {};

    if (conf.empty())
        return jit;

    std::string input_decls = "";
    for (size_t i = 0; i < params.fused_ops.size(); i++) {
        auto fused_dep_codegen = FusedOpsCodeGenerator(params.fused_ops[i]);
        std::string op_type = fused_dep_codegen.GetTypeStr();

        jit.Merge(fused_dep_codegen.MakeFusedTensorJitConstants(conf[0]));
        jit.Merge(fused_dep_codegen.MakeInputDeclsJitConstants(conf[0]));
        if (!params.fused_ops[i].tensors.empty()) {
            std::string optional_comma = (!input_decls.empty() ? "," : "");
            input_decls += optional_comma + "\\\n\tFUSED_OP" + std::to_string(i) + "_DECLS";
        }
    }

    jit.AddConstant(MakeJitConstant("FUSED_OPS_DECLS", input_decls));
    jit.AddConstant(MakeJitConstant("HAS_FUSED_OPS", true));
    jit.AddConstant(MakeJitConstant("HAS_FUSED_OPS_DECLS", !input_decls.empty()));

    return jit;
}

bool KernelBase::IsFusedPrimitiveSupported(const fused_operation_desc& fused_op) const {
    for (auto& supported_op : GetSupportedFusedOps()) {
        if (fused_op.GetType() == supported_op)
            return true;
    }

    return false;
}

std::vector<KernelBase::FusedOpType> KernelBase::GetSupportedFusedOps() const {
    return {};
}


}  // namespace kernel_selector
