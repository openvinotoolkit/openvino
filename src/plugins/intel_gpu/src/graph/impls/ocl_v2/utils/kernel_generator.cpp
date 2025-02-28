// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_generator.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "kernels_db.hpp"
#include "common_utils/jitter.hpp"

#include <cctype>

namespace ov::intel_gpu::ocl {

JitConstants SingleKernelGenerator::make_tensors_jit_constants(const kernel_impl_params& params) const {
    JitConstants jit_constants;

    const auto& in_offsets_map = params.in_port_to_shape_info_offset;
    const auto& out_offsets_map = params.out_port_to_shape_info_offset;

    for (size_t i = 0; i < params.input_layouts.size(); i++) {
        jit_constants.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[i], in_offsets_map.at(i)));
    }

    jit_constants.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], out_offsets_map.at(0)));
    for (size_t i = 1; i < params.output_layouts.size(); i++) {
        jit_constants.add(make_layout_jit_constants("OUTPUT" + to_code_string(i), params.output_layouts[i], out_offsets_map.at(i)));
    }

    return jit_constants;
}

JitConstants SingleKernelGenerator::make_base_jit_constants(const kernel_impl_params& params) const {
    JitConstants jit_constants;

    auto entry_point = get_entry_point(params);
    jit_constants.add(make_jit_constant("KERNEL(name)", "__kernel void " + entry_point));
    jit_constants.add(make_jit_constant("KERNEL_ID", entry_point));

    if (params.is_dynamic()) {
        jit_constants.add(make_jit_constant("IS_DYNAMIC", 1));
        jit_constants.add(make_jit_constant("OPTIONAL_SHAPE_INFO_ARG", "__global const int* shape_info,"));
        jit_constants.add(make_jit_constant("OPTIONAL_SHAPE_INFO_TENSOR", "shape_info,"));
    } else {
        jit_constants.add(make_jit_constant("OPTIONAL_SHAPE_INFO_ARG", ""));
        jit_constants.add(make_jit_constant("OPTIONAL_SHAPE_INFO_TENSOR", ""));
    }

    return jit_constants;
}

std::string SingleKernelGenerator::build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& kernel_id) const {
    CodeBuilder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + std::string(template_name) + " ")
        .add_line("// Kernel name: " + kernel_id)
        .decoration_macro("FUNC", "", kernel_id)
        .decoration_macro("FUNC_CALL", "", kernel_id)
        .decoration_macro("CONST_ARRAY_DECL", "__constant size_t ", kernel_id + " []")
        .decoration_macro("CONST_ARRAY_REF", "", kernel_id);

    for (auto& jit_constant : jit_constants) {
        code.value_macro(jit_constant.name, jit_constant.value);
    }

    try {
        code.add_line(std::string(OCLSourcesDB::get_kernel_template(template_name)));
    } catch (std::exception&) {
        OPENVINO_THROW("[GPU] Couldn't find kernel template: ", template_name);
    }

    for (auto& jit_constant : jit_constants) {
        code.undef_macro(jit_constant.name);
    }

    return code.str();
}

KernelData SingleKernelGenerator::get_kernel_data(const kernel_impl_params& params) const {
    auto jit = get_jit_constants(params);

    KernelData kd;
    kd.code.kernel_string = std::make_shared<KernelString>();
    kd.code.kernel_string->language = kernel_language::OCLC_V2;
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

std::string SingleKernelGenerator::get_entry_point(const kernel_impl_params& params) const {
    auto entry_point = std::string(m_kernel_name) + m_stage_suffix;

    entry_point += "_" + std::to_string(params.hash());
    entry_point += params.is_dynamic() ? "__sa" : "";

    return entry_point;
}

std::string SingleKernelGenerator::get_build_options(const kernel_impl_params& params) const {
    std::string options;
    const auto& device_info = params.get_program().get_engine().get_device_info();
    if (device_info.vendor_id == cldnn::INTEL_VENDOR_ID) {
        options = " -cl-mad-enable";
        if (device_info.supports_local_block_io)
            options += " -Dcl_intel_subgroup_local_block_io -DLOCAL_BLOCK_IO_SUPPORTED=1";
    }

#if CL_TARGET_OPENCL_VERSION >= 200
        options += " -cl-std=CL2.0";
#endif

    return options;
}

JitConstants SingleKernelGenerator::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = make_base_jit_constants(params);
    jit.merge(make_tensors_jit_constants(params));
    jit.add(make_activation_jit_constants(activation_func::none, ov::element::dynamic, "", false, false));
    return jit;
}

Arguments SingleKernelGenerator::get_arguments_desc(const kernel_impl_params& params) const {
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

}  // namespace ov::intel_gpu::ocl
