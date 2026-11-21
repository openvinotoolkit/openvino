// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// See experimental_kernel_generator.hpp. Replicates the upstream
// `ov::intel_gpu::cm::KernelGenerator::get_kernel_data()`/`build_code()` logic but resolves the kernel
// template string through the experimental SourcesDB so that no experimental kernel is registered in
// the upstream SourcesDB.

#include "experimental_kernel_generator.hpp"

#include <string>

#include "common_utils/jitter.hpp"
#include "kernels_db.hpp"

namespace ov::intel_gpu::experimental::cm {

namespace {

[[maybe_unused]] std::string build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& entry_point) {
    CodeBuilder code;
    code.add_line("\n//====================================================")
        .add_line("// Experimental Kernel template: " + std::string(template_name) + " ")
        .add_line("// Kernel name: " + entry_point);

    for (const auto& jit_constant : jit_constants) {
        code.value_macro(jit_constant.name, jit_constant.value);
    }

    code.add_line(std::string(ExperimentalSourcesDB::get_kernel_template(template_name)));

    for (const auto& jit_constant : jit_constants) {
        code.undef_macro(jit_constant.name);
    }

    return code.str();
}

}  // namespace

KernelData ExperimentalKernelGenerator::get_kernel_data(const RuntimeParams& params) const {
    auto jit = get_jit_constants(params);

    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language = KernelLanguage::CM;
    kd.code->entry_point = get_entry_point(params);
    kd.code->jit = "";
    kd.code->undefs = "";
    kd.code->options = get_build_options(params);
    kd.code->batch_compilation = true;
    kd.code->has_microkernels = false;
    kd.code->str = build_code(m_kernel_name, jit, kd.code->entry_point);

    kd.params.arguments = get_arguments_desc(params);
    kd.params.layerID = params.desc->id;
    kd.update_dispatch_data_func = get_dispatch_data_func();
    kd.need_args_update = true;
    kd.need_dispatch_data_update = true;

    return kd;
}

}  // namespace ov::intel_gpu::experimental::cm