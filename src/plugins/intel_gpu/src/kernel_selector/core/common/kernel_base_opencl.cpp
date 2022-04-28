// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_base_opencl.h"
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#if defined __INTEL_COMPILER
#pragma warning disable : 177
#endif

namespace kernel_selector {
namespace {

class CodeBuilder {
    std::ostringstream oss;
    std::string code;
    std::vector<std::string> defined_macroses;

    CodeBuilder& register_macro(const std::string& name) {
        assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) == 0);

        defined_macroses.push_back(name);
        return *this;
    }

public:
    CodeBuilder& set_code(const std::string& c) {
        assert(code.empty());
        code = c;
        return *this;
    }

    CodeBuilder& add_line(const std::string& line) {
        oss << line << "\n";
        return *this;
    }

    CodeBuilder& decoration_macro(const std::string& name,
                                  const std::string& prefix,
                                  const std::string& postfix,
                                  const std::string& name_prefix = std::string()) {
        oss << "#define " << name << "(name) " << prefix << " " + name_prefix + "_##" + "name"
            << (postfix.empty() ? "" : "##_") << postfix << std::endl;
        return register_macro(name);
    }

    CodeBuilder& value_macro(const std::string& name, const std::string& value) {
        oss << "#define " << name << " " << value << std::endl;
        return register_macro(name.substr(0, name.find('(')));
    }

    std::string str() {
        std::ostringstream os;
        os << oss.str();
        os << code << std::endl;
        return os.str();
    }
};
}  // namespace

std::string KernelBaseOpenCL::GetEntryPoint(const std::string& templateName,
                                            const std::string& layerID,
                                            const Params& params,
                                            const optional_params& options,
                                            const size_t partID) const {
    std::string kernelID = layerID;

    if (kernelID.empty() || !options.meaningfulKernelsNames) {
        kernelID = templateName;
    }

    std::replace(kernelID.begin(), kernelID.end(), '.', '_');
    std::replace(kernelID.begin(), kernelID.end(), '/', '_');

    // UniqueID = program_id + processing_index + additional weight/reorder tag
    kernelID += "_" + params.uniqueID + "_" + std::to_string(partID);

    return kernelID;
}

std::pair<std::string, std::string> KernelBaseOpenCL::CreateJit(const std::string& template_name,
                                          const JitConstants& constants,
                                          const std::string& kernel_id) const {
    class CodeBuilder code;
    std::string undefs;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + template_name + " ")
        .add_line("// Kernel name: " + kernel_id)
        .value_macro("KERNEL(name)", "__kernel void " + kernel_id)
        .decoration_macro("FUNC", "", kernel_id)
        .decoration_macro("FUNC_CALL", "", kernel_id);

    undefs += "#undef KERNEL\n";
    undefs += "#undef FUNC\n";
    undefs += "#undef FUNC_CALL\n";

    for (auto& definition : constants.GetDefinitions()) {
        code.value_macro(definition.first, definition.second);
        undefs += "#ifdef " + definition.first.substr(0, definition.first.find('(')) + "\n";
        undefs += "#undef " + definition.first.substr(0, definition.first.find('(')) + "\n";
        undefs += "#endif\n";
    }

    std::string jit = code.str();
    // std::cout << jit << std::endl;
    std::pair<std::string, std::string> jit_undefs(jit, undefs);

    return jit_undefs;
}

Arguments KernelBaseOpenCL::GetArgsDesc(uint32_t num_of_input,
                                          bool use_weights,
                                          bool use_bias,
                                          uint32_t number_of_inputs_for_fused_prim,
                                          uint32_t num_of_output) const {
    Arguments args;

    for (uint32_t i = 0; i < num_of_input; i++) {
        args.push_back({ArgumentDescriptor::Types::INPUT, i});
    }

    for (uint32_t i = 0; i < num_of_output; i++) {
        args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
    }

    if (use_weights) {
        args.push_back({ArgumentDescriptor::Types::WEIGHTS, 0});
    }

    if (use_bias) {
        args.push_back({ArgumentDescriptor::Types::BIAS, 0});
    }

    for (uint32_t i = 0; i < number_of_inputs_for_fused_prim; i++) {
        args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, i});
    }

    return args;
}

std::shared_ptr<KernelString> KernelBaseOpenCL::GetKernelString(const std::string& name,
                                                                  const std::pair<std::string, std::string>& jit,
                                                                  const std::string& entry_point,
                                                                  const EngineInfo& engine_info,
                                                                  const std::string& exe_mode) const {
    std::shared_ptr<KernelString> kernel_string = std::make_shared<KernelString>();

    auto codes = db.get(name);

    if (codes.size()) {
        kernel_string->str = codes[0];
        kernel_string->jit = jit.first;
        kernel_string->undefs = jit.second;
        kernel_string->options = exe_mode + " -cl-mad-enable";
        if (engine_info.bOptHintsSupport)
            kernel_string->options += " -DOPT_HINTS_SUPPORTED=1";
        if (engine_info.bLocalBlockIOSupport)
            kernel_string->options += " -Dcl_intel_subgroup_local_block_io -DLOCAL_BLOCK_IO_SUPPORTED=1";
        kernel_string->entry_point = entry_point;
        kernel_string->batch_compilation = true;
    }

    return kernel_string;
}

uint32_t KernelBaseOpenCL::GetFusedPrimitiveInputsCount(const Params &params) const {
    auto p = dynamic_cast<const base_params&>(params);
    uint32_t fused_deps_total = 0;
    for (auto fused_op : p.fused_ops) {
        fused_deps_total += static_cast<uint32_t>(fused_op.dep_size);
    }

    return fused_deps_total;
}

void KernelBaseOpenCL::FillCLKernelData(clKernelData& kernel,
                                        const CommonDispatchData& dispatchData,
                                        const EngineInfo& engine_info,
                                        const std::string& kernelMapName,
                                        const std::pair<std::string, std::string>& jit,
                                        const std::string& entryPoint,
                                        const std::string& exeMode,
                                        bool weights,
                                        bool bias,
                                        int number_of_inputs,
                                        uint32_t number_of_inputs_for_fused_prims,
                                        int number_of_outputs) const {
    KernelBase::CheckDispatchData(kernelMapName, dispatchData, engine_info.maxWorkGroupSize);
    kernel.code.kernelString = GetKernelString(kernelMapName, jit, entryPoint, engine_info, exeMode);
    kernel.params.workGroups.global = dispatchData.gws;
    kernel.params.workGroups.local = dispatchData.lws;
    kernel.params.arguments = GetArgsDesc(number_of_inputs, weights, bias, number_of_inputs_for_fused_prims, number_of_outputs);
}
}  // namespace kernel_selector
