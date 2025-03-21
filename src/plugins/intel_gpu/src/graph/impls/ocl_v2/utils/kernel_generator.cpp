// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_generator.hpp"

#include <cctype>

#include "common_utils/kernel_generator_base.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "jitter.hpp"
#include "kernels_db.hpp"

namespace ov::intel_gpu::ocl {

JitConstants KernelGenerator::make_tensors_jit_constants(const RuntimeParams& params) {
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

JitConstants KernelGenerator::make_base_jit_constants(const RuntimeParams& params) const {
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

std::string KernelGenerator::build_code(std::string_view template_name, const JitConstants& jit_constants, const std::string& entry_point) {
    CodeBuilder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + std::string(template_name) + " ")
        .add_line("// Kernel name: " + entry_point)
        .decoration_macro("FUNC", "", entry_point)
        .decoration_macro("FUNC_CALL", "", entry_point)
        .decoration_macro("CONST_ARRAY_DECL", "__constant size_t ", entry_point + " []")
        .decoration_macro("CONST_ARRAY_REF", "", entry_point);

    for (const auto& jit_constant : jit_constants) {
        code.value_macro(jit_constant.name, jit_constant.value);
    }

    code.add_line(std::string(SourcesDB::get_kernel_template(template_name)));

    for (const auto& jit_constant : jit_constants) {
        code.undef_macro(jit_constant.name);
    }

    return code.str();
}

KernelData KernelGenerator::get_kernel_data(const RuntimeParams& params) const {
    auto jit = get_jit_constants(params);

    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language = KernelLanguage::OCLC_V2;
    kd.code->entry_point = get_entry_point(params);
    kd.code->jit = "";  // jit and undefa are a part of the code now
    kd.code->undefs = "";
    kd.code->options = get_build_options(params);
    kd.code->batch_compilation = true;
    kd.code->has_microkernels = false;
    kd.code->str = build_code(m_kernel_name, jit, kd.code->entry_point);

    kd.params.arguments = get_arguments_desc(params);
    kd.update_dispatch_data_func = get_dispatch_data_func();
    kd.need_args_update = true;
    kd.need_dispatch_data_update = true;

    return kd;
}

std::string KernelGenerator::get_entry_point(const RuntimeParams& params) const {
    return m_kernel_name + "_" + m_stage_suffix + "_" + std::to_string(params.hash()) + (params.is_dynamic() ? "__sa" : "");
}

std::string KernelGenerator::get_build_options(const RuntimeParams& params) const {
    std::string options;
    const auto& device_info = params.get_program().get_engine().get_device_info();
    if (device_info.vendor_id == cldnn::INTEL_VENDOR_ID) {
        options = " -cl-mad-enable";
    }

#if CL_TARGET_OPENCL_VERSION >= 200
    options += " -cl-std=CL2.0";
#endif

    return options;
}

JitConstants KernelGenerator::get_jit_constants(const RuntimeParams& params) const {
    auto jit = make_base_jit_constants(params);
    jit.add(make_tensors_jit_constants(params));
    return jit;
}

Arguments KernelGenerator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;

    if (params.is_dynamic()) {
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    }

    for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
        args.push_back({ArgumentDescriptor::Types::INPUT, i});
    }

    for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
        args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
    }

    return args;
}

void KernelGenerator::add_fused_ops_arguments(Arguments& args, const RuntimeParams& params) {
    if (params.has_fused_primitives()) {
        size_t num_fused_deps = 0;
        for (const auto& fd : params.fused_desc) {
            for (const auto& in_d : fd.inputs) {
                if (in_d.m_type == cldnn::FusedInputType::EXTERNAL) {
                    num_fused_deps++;
                }
            }
        }
        for (size_t i = 0; i < num_fused_deps; i++) {
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, static_cast<uint32_t>(i)});
        }
    }
}

}  // namespace ov::intel_gpu::ocl
