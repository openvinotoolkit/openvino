// Copyright (c) 2016-2019 Intel Corporation
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


#include "kernel_base.h"

namespace kernel_selector {
const primitive_db KernelBase::db;
size_t KernelBase::counter = 0;

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
        MakeJitConstant("GRADIENT", params.gradient),
    };

    // for activation function
    jit.Merge(MakeUnitTypeJitConstants(unitType));
    jit.Merge(MakeActivationJitConstants(params.activations));

    for (size_t i = 0; i < params.inputs.size(); i++) {
        jit.AddConstant(MakeJitConstant("INPUT" + toCodeString(i), params.inputs[i]));
    }

    jit.AddConstant(MakeJitConstant("LayerID", params.layerID));

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

    std::string input_decls = "";
    std::vector<std::string> fused_ops;
    std::vector<std::string> names;
    for (const auto &c : conf) {
        fused_ops.emplace_back("");
        names.push_back(c.input_var_name);
    }

    for (size_t i = 0; i < params.fused_ops.size(); i++) {
        auto& fused_dep = params.fused_ops[i];
        for (size_t j = 0; j < conf.size(); j++) {
            std::string out_var = "";
            jit.Merge(fused_dep.MakeLoadJitConstants(conf[j]));
            jit.Merge(fused_dep.MakeOpJitConstants(conf[j], names[j], out_var));
            names[j] = out_var;

            fused_ops[j] += "\\\n\tFUSED_OP" + std::to_string(i) + "_LOAD" + conf[j].suffix;
            fused_ops[j] += "\\\n\tFUSED_OP" + std::to_string(i) + "_ACTION" + conf[j].suffix;
        }
    }

    jit.Merge(MakeFusedOpsDeclsJitConstants(params, conf));

    for (size_t j = 0; j < conf.size(); j++) {
        jit.AddConstant(MakeJitConstant("FUSED_OPS" + conf[j].suffix, fused_ops[j]));
        jit.AddConstant(MakeJitConstant("FINAL_NAME" + conf[j].suffix, names[j]));
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
        auto& fused_dep = params.fused_ops[i];
        std::string op_type = fused_dep.GetTypeStr();

        jit.Merge(fused_dep.MakeFusedTensorJitConstants(conf[0]));
        jit.Merge(fused_dep.MakeInputDeclsJitConstants(conf[0]));
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

}  // namespace kernel_selector
