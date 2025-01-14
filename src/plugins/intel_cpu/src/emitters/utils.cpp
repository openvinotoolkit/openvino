// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

namespace ov {
namespace intel_cpu {

std::string jit_emitter_pretty_name(const std::string& pretty_func) {
#define SAFE_SYMBOL_FINDING(idx, find)        \
    auto idx = (find);                        \
    if (idx == std::string::npos || idx == 0) \
        return pretty_func;
    // Example:
    //      pretty_func := void ov::intel_cpu::jit_load_memory_emitter::emit_impl(const std::vector<size_t>& in, const
    //      std::vector<size_t>& out) const begin := -----------| end :=
    //      ---------------------------------------------------| result := ov::intel_cpu::jit_load_memory_emitter
    // Signatures:
    //      GCC:   void foo() [with T = {type}]
    //      clang: void foo() [T = {type}]
    //      MSVC:  void __cdecl foo<{type}>(void)
    SAFE_SYMBOL_FINDING(parenthesis, pretty_func.find("("))
    if (pretty_func[parenthesis - 1] == '>') {  // To cover template on MSVC
        parenthesis--;
        size_t counter = 1;
        while (counter != 0 && parenthesis > 0) {
            parenthesis--;
            if (pretty_func[parenthesis] == '>')
                counter++;
            if (pretty_func[parenthesis] == '<')
                counter--;
        }
    }
    SAFE_SYMBOL_FINDING(end, pretty_func.substr(0, parenthesis).rfind("::"))
    SAFE_SYMBOL_FINDING(begin, pretty_func.substr(0, end).rfind(" "))
    begin++;
#undef SAFE_SYMBOL_FINDING
    return end > begin ? pretty_func.substr(begin, end - begin) : pretty_func;
}

}  // namespace intel_cpu
}  // namespace ov
