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

#include "common_kernel_base.h"
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

std::string common_kernel_base::GetEntryPoint(const std::string& templateName,
                                              const std::string& layerID,
                                              const optional_params& options) const {
    std::string kernelID = layerID;

    if (kernelID.empty() || !options.meaningfulKernelsNames) {
        kernelID = templateName;
    }

    std::replace(kernelID.begin(), kernelID.end(), '.', '_');
    std::replace(kernelID.begin(), kernelID.end(), '/', '_');

    kernelID += "_" + std::to_string(UniqeID());

    return kernelID;
}

std::string common_kernel_base::CreateJit(const std::string& template_name,
                                          const JitConstants& constants,
                                          const std::string& kernel_id) const {
    class CodeBuilder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + template_name + " ")
        .add_line("// Kernel name: " + kernel_id)
        .value_macro("KERNEL(name)", "__kernel void " + kernel_id)
        .decoration_macro("FUNC", "", kernel_id)
        .decoration_macro("FUNC_CALL", "", kernel_id);

    for (auto& definition : constants.GetDefinitions()) {
        code.value_macro(definition.first, definition.second);
    }

    std::string jit = code.str();

    return jit;
}

Arguments common_kernel_base::GetArgsDesc(uint32_t num_of_input,
                                          bool use_weights,
                                          bool use_bias,
                                          uint32_t number_of_inputs_for_fused_prim) const {
    Arguments args;

    for (uint32_t i = 0; i < num_of_input; i++) {
        args.push_back({ArgumentDescriptor::Types::INPUT, i});
    }

    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

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

std::shared_ptr<KernelString> common_kernel_base::GetKernelString(const std::string& name,
                                                                  const std::string& jit,
                                                                  const std::string& entry_point,
                                                                  const EngineInfo& engine_info,
                                                                  const std::string& exe_mode) const {
    std::shared_ptr<KernelString> kernel_string = std::make_shared<KernelString>();

    auto codes = db.get(name);

    if (codes.size()) {
        kernel_string->str = codes[0];
        kernel_string->jit = jit;
        kernel_string->options = exe_mode + " -cl-mad-enable";
        if (engine_info.bOptHintsSupport)
            kernel_string->options += " -DOPT_HINS_SUPPORTED=1";
        if (engine_info.bLocalBlockIOSupport)
            kernel_string->options += " -Dcl_intel_subgroup_local_block_io";
        kernel_string->entry_point = entry_point;
        kernel_string->batch_compilation = true;
    }

    return kernel_string;
}

static void Check_RunInfoData(const std::string& kernelName, const kernel_selector::CommonDispatchData& runInfo) {
    if (runInfo.lws0 * runInfo.lws1 * runInfo.lws2 > 256) {
        std::cout << "ERROR: dispatch data for kernel: " << kernelName << " LWS cannot be greater than 256!\n"
                  << std::endl;
    }
    if (runInfo.gws0 == 0 || runInfo.gws1 == 0 || runInfo.gws2 == 0 || runInfo.lws0 == 0 || runInfo.lws1 == 0 ||
        runInfo.lws2 == 0) {
        std::cout << "ERROR: dispatch data for kernel: " << kernelName << " dispatch data cannot contain zeros!"
                  << std::endl;
    }
    if (runInfo.gws0 % runInfo.lws0 != 0) {
        std::cout << "ERROR: dispatch data for kernel: " << kernelName << " is incorrect: GWS0: " << runInfo.gws0
                  << " LWS0: " << runInfo.lws0 << std::endl;
    }
    if (runInfo.gws1 % runInfo.lws1 != 0) {
        std::cout << "ERROR: dispatch data for kernel: " << kernelName << " is incorrect: GWS1: " << runInfo.gws1
                  << " LWS1: " << runInfo.lws1 << std::endl;
    }
    if (runInfo.gws2 % runInfo.lws2 != 0) {
        std::cout << "ERROR: dispatch data for kernel: " << kernelName << " is incorrect: GWS2: " << runInfo.gws2
                  << " LWS2: " << runInfo.lws2 << std::endl;
    }
}

uint32_t common_kernel_base::GetFusedPrimitiveInputsCount(const Params &params) const {
    auto p = dynamic_cast<const base_params&>(params);
    uint32_t fused_deps_total = 0;
    for (auto fused_op : p.fused_ops) {
        fused_deps_total += static_cast<uint32_t>(fused_op.dep_size);
    }

    return fused_deps_total;
}

void common_kernel_base::FillCLKernelData(clKernelData& kernel,
                                          const CommonDispatchData& runInfo,
                                          const EngineInfo& engine_info,
                                          const std::string& kernelMapName,
                                          const std::string& jit,
                                          const std::string& entryPoint,
                                          const std::string& exeMode,
                                          bool weights,
                                          bool bias,
                                          int number_of_inputs,
                                          uint32_t number_of_inputs_for_fused_prims) const {
    Check_RunInfoData(kernelMapName, runInfo);
    kernel.workGroups.global = {runInfo.gws0, runInfo.gws1, runInfo.gws2};
    kernel.workGroups.local = {runInfo.lws0, runInfo.lws1, runInfo.lws2};
    kernel.kernelString = GetKernelString(kernelMapName, jit, entryPoint, engine_info, exeMode);
    kernel.arguments =
        GetArgsDesc(number_of_inputs, weights, bias, number_of_inputs_for_fused_prims);
}
}  // namespace kernel_selector
