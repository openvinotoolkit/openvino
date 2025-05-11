// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov::intel_cpu {

std::string jit_emitter_pretty_name(const std::string& pretty_func) {
    // Example:
    //      pretty_func := void ov::intel_cpu::jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in, const
    //      std::vector<size_t>& out) const begin := -----------| end :=
    //      ---------------------------------------------------| result := ov::intel_cpu::jit_load_memory_emitter
    // Signatures:
    //      GCC:   void foo() [with T = {type}]
    //      clang: void foo() [T = {type}]
    //      MSVC:  void __cdecl foo<{type}>(void)
    auto parenthesis = pretty_func.find('(');
    if (parenthesis == std::string::npos || parenthesis == 0) {
        return pretty_func;
    }
    if (pretty_func[parenthesis - 1] == '>') {  // To cover template on MSVC
        parenthesis--;
        size_t counter = 1;
        while (counter != 0 && parenthesis > 0) {
            parenthesis--;
            if (pretty_func[parenthesis] == '>') {
                counter++;
            }
            if (pretty_func[parenthesis] == '<') {
                counter--;
            }
        }
    }
    auto end = pretty_func.substr(0, parenthesis).rfind("::");
    if (end == std::string::npos || end == 0) {
        return pretty_func;
    }

    auto begin = pretty_func.substr(0, end).rfind(' ');
    if (begin == std::string::npos || begin == 0) {
        return pretty_func;
    }
    begin++;
    return end > begin ? pretty_func.substr(begin, end - begin) : pretty_func;
}

ov::element::Type get_arithmetic_binary_exec_precision(const std::shared_ptr<ov::Node>& n) {
    std::vector<ov::element::Type> input_precisions;
    for (const auto& input : n->inputs()) {
        input_precisions.push_back(input.get_source_output().get_element_type());
    }

    OPENVINO_ASSERT(std::all_of(input_precisions.begin(),
                                input_precisions.end(),
                                [&input_precisions](const ov::element::Type& precision) {
                                    return precision == input_precisions[0];
                                }),
                    "Binary Eltwise op has unequal input precisions");

    return input_precisions[0];
}

}  // namespace ov::intel_cpu
