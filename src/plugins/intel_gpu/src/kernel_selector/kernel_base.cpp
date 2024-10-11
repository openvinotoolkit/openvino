// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_base.h"

#include <sstream>

namespace kernel_selector {
const primitive_db KernelBase::db;

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

void KernelBase::CheckDispatchData(const std::string& kernelName, const kernel_selector::CommonDispatchData& dispatchData,
                                   const size_t maxWorkGroupSize) {
    if (dispatchData.gws.size() != 3 || dispatchData.lws.size() != 3)
        throw std::runtime_error("ERROR: Invalid dispatch data for kernel: " + kernelName + ": " +
                                 ": LWS and GWS size is expected to be equal to 3. Actual: " +
                                 toString(dispatchData));

    if (dispatchData.lws[0] * dispatchData.lws[1] * dispatchData.lws[2] > maxWorkGroupSize) {
        throw std::runtime_error("ERROR: Invalid dispatch data for kernel: " + kernelName +
                                 ": LWS cannot be greater than " + std::to_string(static_cast<int>(maxWorkGroupSize)) + ". Actual: " +
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
    // TODO: multiple output support
    return params.outputs[0].GetDType() == type ||
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
JitConstants KernelBase::MakeBaseParamsJitConstants(const base_params& params, bool add_tensor_definitions) const {
    auto unitType = GetUnitType(params);

    JitConstants jit{
        MakeJitConstant("FP64_SUPPORTED", params.engineInfo.supports_fp64),
        MakeJitConstant("FP16_SUPPORTED", params.engineInfo.supports_fp16),
        MakeJitConstant("FP16_UNIT_USED", IsTypeUsedIn(Datatype::F16, params)),
        MakeJitConstant("INT8_UNIT_USED", IsTypeUsedIn(Datatype::INT8, params)),
        MakeJitConstant("INT32_UNIT_USED", IsTypeUsedIn(Datatype::INT32, params)),
        MakeJitConstant("INT64_UNIT_USED", IsTypeUsedIn(Datatype::INT64, params)),
        MakeJitConstant("UINT8_UNIT_USED", IsTypeUsedIn(Datatype::UINT8, params)),
        MakeJitConstant("UINT32_UNIT_USED", IsTypeUsedIn(Datatype::UINT32, params)),
    };

    // for activation function
    jit.Merge(MakeUnitTypeJitConstants(unitType));
    // Changed data type from unit type to output data type to fix the issue case that
    // the activation function makes cl kernel build error when the output data type
    // and unit type are different and activation param is existed
    bool convert_input_to_output_dt = (params.outputs[0].GetDType() == Datatype::F32 && params.inputs[0].GetDType() == Datatype::F16);
    // If input is FP16 and output is FP32, convert input to float before running activation function.
    jit.Merge(MakeActivationJitConstants(params.activations, params.outputs[0].GetDType(), "", false, false, convert_input_to_output_dt));

    if (add_tensor_definitions) {
        for (size_t i = 0; i < params.inputs.size(); i++) {
            jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i), params.inputs[i]));
        }

        // NOTE : until all cl kernels legacy is resolved, the outputs are to be OUTPUT, OUTPUT1, OUTPUT2, ...
        jit.AddConstant(MakeJitConstant("OUTPUT", params.outputs[0]));
        for (size_t i = 1; i < params.outputs.size(); i++) {
            jit.AddConstant(MakeJitConstant("OUTPUT" + toCodeString(i), params.outputs[i]));
        }

        if (params.is_shape_agnostic) {
            jit.AddConstant(MakeJitConstant("IS_DYNAMIC", 1));
            jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_ARG", "__global const int* shape_info,"));
            jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_TENSOR", "shape_info,"));
        } else {
            jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_ARG", ""));
            jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_TENSOR", ""));
        }
    }

#ifndef NDEBUG
    jit.AddConstant(MakeJitConstant("LayerID", params.layerID));
#endif
    return jit;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// IsSIMDSizeSupported
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool KernelBase::IsSIMDSizeSupported(const EngineInfo &info, size_t simd_size) const {
    auto supported_sizes = info.supportedSimdSizes;
    return std::find(supported_sizes.begin(), supported_sizes.end(), simd_size) != supported_sizes.end();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MakeFusedOpsJitConstants
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
JitConstants KernelBase::MakeFusedOpsJitConstants(const kernel_selector::base_params &params,
                                                  const std::vector<FusedOpsConfiguration> &conf) const {
    JitConstants jit = {};
    // TODO: multiple output support

    if (conf.empty())
        return jit;

    if (std::all_of(params.fused_ops.cbegin(), params.fused_ops.cend(),
        [](fused_operation_desc desc) { return desc.GetType() == KernelType::REORDER; })) {
        return jit;
    }

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
                // Reorder is not processed by jitter
                if (params.fused_ops[i].GetType() == FusedOpType::REORDER)
                    continue;

                auto fused_dep_codegen = FusedOpsCodeGenerator(params.fused_ops[i]);
                jit.Merge(fused_dep_codegen.MakeLoadJitConstants(c, params.outputs[0]));
                jit.Merge(fused_dep_codegen.MakeOpJitConstants(c, in_name, in_type, out_name));

                bool can_use_preload = fused_dep_codegen.CanPreloadData(c);
                can_all_use_preload &= can_use_preload;
                bool can_preload_eltwise = true;
                if (params.fused_ops[i].GetType() == FusedOpType::ELTWISE &&
                    c.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE)
                    can_preload_eltwise = false;
                fused_ops += "\\\n\tFUSED_OP" + toCodeString(i) + "_LOAD" + c.suffix;
                fused_ops += "\\\n\tFUSED_OP" + toCodeString(i) + "_ACTION" + c.suffix;
                if (can_use_preload && can_preload_eltwise)
                    fused_ops_preload += "\\\n\tFUSED_OP" + toCodeString(i) + "_LOAD" + c.suffix;
                if (c.allow_for_partial_preload && (!can_use_preload || !can_preload_eltwise))
                    fused_ops_calc += "\\\n\tFUSED_OP" + toCodeString(i) + "_LOAD" + c.suffix;
                fused_ops_calc += "\\\n\tFUSED_OP" + toCodeString(i) + "_ACTION" + c.suffix;
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
    std::string input_args = "";

    for (size_t i = 0; i < params.fused_ops.size(); i++) {
        auto fused_dep_codegen = FusedOpsCodeGenerator(params.fused_ops[i]);
        std::string op_type = fused_dep_codegen.GetTypeStr();

        jit.Merge(fused_dep_codegen.MakeFusedTensorJitConstants(conf[0]));
        jit.Merge(fused_dep_codegen.MakeInputDeclsJitConstants(conf[0]));
        if (!params.fused_ops[i].tensors.empty()) {
            std::string optional_comma = (!input_decls.empty() ? "," : "");
            input_decls += optional_comma + "\\\n\tFUSED_OP" + toCodeString(i) + "_DECLS";
            input_args += optional_comma + "\\\n\tFUSED_OP" + toCodeString(i) + "_ARGS";
        }
    }

    jit.AddConstant(MakeJitConstant("FUSED_OPS_DECLS", input_decls));
    jit.AddConstant(MakeJitConstant("FUSED_OPS_ARGS", input_args));
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

DeviceFeaturesKey KernelBase::get_common_subgroups_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;

    bool requires_blocked_read_write_char = false;
    bool requires_blocked_read_write_short = false;
    bool requires_blocked_read_write = false;
    const auto& casted_params = static_cast<const base_params&>(params);

    std::vector<Datatype> tensor_types;
    for (auto& t : casted_params.inputs) {
        tensor_types.push_back(t.GetDType());
    }
    for (auto& t : casted_params.outputs) {
        tensor_types.push_back(t.GetDType());
    }

    for (auto& type : tensor_types) {
        if (type == Datatype::F16) {
            requires_blocked_read_write_short = true;
        } else if (type == Datatype::F32) {
            requires_blocked_read_write = true;
        } else if (type == Datatype::UINT8 || type == Datatype::INT8) {
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
    k.requires_reqd_subgroup_size();

    return k;
}

}  // namespace kernel_selector
