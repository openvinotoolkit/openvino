// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_generator.hpp"

#include <cctype>

#include "intel_gpu/runtime/kernel_args.hpp"
#include "kernels_db.hpp"

namespace ov::intel_gpu::cm {

std::string KernelGenerator::build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& kernel_id) const {
    CodeBuilder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + std::string(template_name) + " ")
        .add_line("// Kernel name: " + kernel_id);

    for (auto& jit_constant : jit_constants) {
        code.value_macro(jit_constant.name, jit_constant.value);
    }

    code.add_line(std::string(SourcesDB::get_kernel_template(template_name)));

    for (auto& jit_constant : jit_constants) {
        code.undef_macro(jit_constant.name);
    }

    return code.str();
}

KernelData KernelGenerator::get_kernel_data(const kernel_impl_params& params) const {
    auto jit = get_jit_constants(params);

    KernelData kd;
    kd.code.kernel_string = std::make_shared<KernelString>();
    kd.code.kernel_string->language = kernel_language::CM;
    kd.code.kernel_string->entry_point = get_entry_point(params);
    kd.code.kernel_string->jit = "";
    kd.code.kernel_string->undefs = "";
    kd.code.kernel_string->options = get_build_options(params);
    kd.code.kernel_string->batch_compilation = true;
    kd.code.kernel_string->has_microkernels = false;
    kd.code.kernel_string->str = build_code(m_kernel_name, jit, kd.code.kernel_string->entry_point);

    kd.params.arguments = get_arguments_desc(params);
    kd.update_dispatch_data_func = get_dispatch_data_func();
    kd.need_args_update = true;

    return kd;
}

std::string KernelGenerator::get_entry_point(const kernel_impl_params& params) const {
    return m_kernel_name + m_stage_suffix + "_" + std::to_string(params.hash()) + (params.is_dynamic() ? "__sa" : "");
}

std::string KernelGenerator::get_build_options(const kernel_impl_params& params) const {
    return " -cmc ";
}

JitConstants KernelGenerator::get_jit_constants(const kernel_impl_params& params) const {
    return {};
}

Arguments KernelGenerator::get_arguments_desc(const kernel_impl_params& params) const {
    Arguments args;

    if (params.is_dynamic())
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});

    for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
        args.push_back({ArgumentDescriptor::Types::INPUT, i});
    }

    for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
        args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
    }

    return args;
}

}  // namespace ov::intel_gpu::cm
