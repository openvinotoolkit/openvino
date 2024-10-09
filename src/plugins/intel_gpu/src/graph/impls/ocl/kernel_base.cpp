// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_base.hpp"
#include "jitter.hpp"
#include <cctype>

namespace ov {
namespace intel_gpu {
namespace ocl {

class CodeBuilder {
    std::ostringstream oss;
    std::string code;
    std::vector<std::string> defined_macroses;

    CodeBuilder& register_macro(const std::string& name) {
        assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) == 0);
        defined_macroses.push_back(name);
        return *this;
    }

    CodeBuilder& unregister_macro(const std::string& name) {
        assert(std::count(defined_macroses.begin(), defined_macroses.end(), name) != 0);
        defined_macroses.erase(std::remove_if(defined_macroses.begin(), defined_macroses.end(), [&](const std::string& v) { return v == name; }));
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

    CodeBuilder& undef_macro(const std::string& name) {
        oss << "#undef " << name.substr(0, name.find('(')) << std::endl;
        return unregister_macro(name.substr(0, name.find('(')));
    }

    std::string str() {
        oss << std::endl;
        return oss.str();
    }
};

const primitive_db KernelGeneratorBase::db;

JitConstants SingleKernelGenerator::make_base_jit_constants(const program_node& node, const kernel_impl_params& params) const {
    JitConstants jit_constants;

    auto entry_point = get_entry_point(node, params);
    jit_constants.add(make_jit_constant("KERNEL(name)", "__kernel void " + entry_point));
    jit_constants.add(make_jit_constant("KERNEL_ID", entry_point));

    size_t shape_info_idx = 0;
    for (size_t i = 0; i < params.input_layouts.size(); i++) {
        jit_constants.add(make_layout_jit_constants("INPUT" + to_code_string(i), params.input_layouts[i], shape_info_idx));
        if (params.input_layouts[i].is_dynamic())
            shape_info_idx++;
    }

    jit_constants.add(make_layout_jit_constants("OUTPUT", params.output_layouts[0], shape_info_idx++));
    if (params.input_layouts[0].is_dynamic())
        shape_info_idx++;
    for (size_t i = 1; i < params.output_layouts.size(); i++) {
        jit_constants.add(make_layout_jit_constants("OUTPUT" + to_code_string(i), params.output_layouts[i], shape_info_idx++));
        if (params.input_layouts[i].is_dynamic())
            shape_info_idx++;
    }

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

std::string SingleKernelGenerator::build_code(const std::string& template_name, const JitConstants& jit_constants, const std::string& kernel_id) const {
    CodeBuilder code;
    code.add_line("\n//====================================================")
        .add_line("// Kernel template: " + template_name + " ")
        .add_line("// Kernel name: " + kernel_id)
        .decoration_macro("FUNC", "", kernel_id)
        .decoration_macro("FUNC_CALL", "", kernel_id)
        .decoration_macro("CONST_ARRAY_DECL", "__constant size_t ", kernel_id + " []")
        .decoration_macro("CONST_ARRAY_REF", "", kernel_id);

    for (auto& jit_constant : jit_constants) {
        code.value_macro(jit_constant.name, jit_constant.value);
    }

    try {
        code.add_line(db.get(template_name)[0]);
    } catch (std::exception&) {
        OPENVINO_THROW("[GPU] Couldn't find kernel template: ", template_name);
    }

    for (auto& jit_constant : jit_constants) {
        code.undef_macro(jit_constant.name);
    }

    return code.str();
}

KernelData SingleKernelGenerator::get_kernel_data(const program_node& node, const kernel_impl_params& params) const {
    KernelData kd;
    auto kernel_str = std::make_shared<KernelString>();
    auto entry_point = get_entry_point(node, params);
    auto jit = get_jit_constants(node, params);
    auto dispatch_data_f = get_dispatch_data_func(params);
    jit.add(m_jit_constants);

    kernel_str->entry_point = entry_point;
    kernel_str->jit = "";
    kernel_str->undefs = "";
    kernel_str->options = get_build_options(node, params);
    kernel_str->batch_compilation = false;
    kernel_str->has_microkernels = false;
    kernel_str->str = build_code(get_name(), jit, entry_point);
    kd.code.kernelString = kernel_str;
    kd.params.workGroups = dispatch_data_f(params).work_groups;
    kd.params.arguments = get_arguments_desc(node, params);
    kd.internal_buffers = get_interanl_buffers(node, params);
    kd.update_dispatch_data_func = dispatch_data_f;

    return kd;
}

std::string SingleKernelGenerator::get_entry_point(const program_node& node, const kernel_impl_params& params) const {
    std::string entry_point = get_name();

    entry_point += "_" + std::to_string(params.hash());
    entry_point += "__sa";

    return entry_point;
}

std::string SingleKernelGenerator::get_build_options(const program_node& node, const kernel_impl_params& params) const {
    std::string options;
    const auto& device_info = node.get_program().get_engine().get_device_info();
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

void SingleKernelGenerator::add_common_jit_constants(const JitConstants& jit_constants) {
    m_jit_constants.add(jit_constants);
}

JitConstants SingleKernelGenerator::get_jit_constants(const program_node& node, const kernel_impl_params& params) const {
    auto jit = make_base_jit_constants(node, params);
    jit.add(make_activation_jit_constants(activation_func::none, ov::element::undefined, "", false, false));
    return jit;
}

Arguments SingleKernelGenerator::get_arguments_desc(const program_node& node, const kernel_impl_params& params) const {
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

DispatchData SingleKernelGenerator::get_dispatch_data(const kernel_impl_params& params) const {
    auto f = get_dispatch_data_func(params);

    return f(params);
}

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
