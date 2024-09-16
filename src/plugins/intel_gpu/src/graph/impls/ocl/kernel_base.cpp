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

JitConstants KernelGeneratorBase::make_base_jit_constants(const program_node& node, const kernel_impl_params& params) const {
    JitConstants jit_constants;

    auto entry_point = get_entry_point(node, params);
    jit_constants.add(make_jit_constant("KERNEL(name)", "__kernel void " + entry_point));
    jit_constants.add(make_jit_constant("KERNEL_ID", entry_point));

    for (size_t i = 0; i < params.input_layouts.size(); i++) {
        jit_constants.add(make_jit_constants("INPUT" + to_code_string(i), params.input_layouts[i]));
    }

    jit_constants.add(make_jit_constants("OUTPUT", params.output_layouts[0]));
    for (size_t i = 1; i < params.output_layouts.size(); i++) {
        jit_constants.add(make_jit_constants("OUTPUT" + to_code_string(i), params.output_layouts[i]));
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

std::string KernelGeneratorBase::build_code(const std::string& template_name, const JitConstants& jit_constants, const std::string& kernel_id) const {
    CodeBuilder code;
    std::string undefs;
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

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
